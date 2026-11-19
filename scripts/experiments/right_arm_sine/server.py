"""Local Tianji right-arm sine tracking experiment. Run as a module from repo root."""
from copy import deepcopy
import argparse
import json
import math
import secrets
import signal
import threading
import time
from contextlib import contextmanager, nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from manimux.embodiments.arm.tianji.kinematics import j67_ok
from manimux.embodiments.arm.tianji.position_lead import JointPositionLead

from manimux.cli import _load_config, _runtime_lock
from manimux.embodiments.arm.tianji.arm import TianjiArmSettings, TianjiController
from manimux.embodiments.robot import RobotModel
from manimux.runtime.execution import ExecutionPipeline
from manimux.runtime.executors import SmoothExecutor
from manimux.runtime.safety import SafetyGuard
from manimux.types import ActionHorizon, RobotCommand, RobotState

CONFIG = Path("manimux/configs/experiments/throw_ball/pi05/tianji_pi05_native_h50_29999_rtc_retime_gripper.yaml")
DATA = Path(".local/right-arm-sine")
DATA.mkdir(parents=True, exist_ok=True)
cfg = _load_config(CONFIG)
body = yaml.safe_load(Path(cfg["robot"]["config"]).read_text())
hw = body["hardware"]
bounds = body["components"]["right_arm"]["hardware"]
model = RobotModel.from_config(cfg["robot"]["config"]).kinematics.models["right_arm"]
settings = TianjiArmSettings(bounds["joint_limits"], hw["velocity_ratio"], hw["acceleration_ratio"],
                             measured_joint_limits=bounds["measured_joint_limits"])
gate = threading.Lock()
stop = threading.Event()
status = {"phase": "idle"}
preview = None
heartbeat = 0.
token = secrets.token_urlsafe(24)


class ProbeController(TianjiController):
    def feedforward(self, period_ms):
        self._begin_batch()
        self._check(self._robot.set_PD_vel_est_step("B", period_ms), "right velocity feedforward")
        self._check(self._robot.send_cmd(), "submit right velocity feedforward")

    def sample(self):
        joints, _, data = self._read()
        output = data["outputs"][1]
        inputs = data["inputs"][1]
        diagnostics = {k: inputs.get(k) for k in (
            "joint_cmd_pos", "in_frame_serial", "frame_miss_cnt", "sys_cyc", "sys_cyc_miss_cnt",
        )}
        return joints["right"], np.asarray(output["fb_joint_cmd"]), int(output["frame_serial"]), diagnostics


@contextmanager
def connected():
    with _runtime_lock(cfg, CONFIG, mode="right-arm-sine"):
        arm = ProbeController(settings={"right": settings}, ip=cfg["robot"]["options"]["hardware"]["ip"],
                              stale_timeout_s=.1, max_tracking_error_deg=hw["max_tracking_error_deg"])
        arm.connect()
        try:
            yield arm
        finally:
            arm.close()


def save(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, allow_nan=False))
    tmp.replace(path)


def capture():
    with connected() as arm:
        q, _, serial, _ = arm.sample()
    anchor = dict(q=q.tolist(), serial=serial, captured=time.time())
    save(DATA / "origin.json", anchor)
    return anchor


def tcp(q):
    return model.fk(np.r_[q, 0.])


def plan(opts):
    opts = dict(opts, ka_s2=JointPositionLead.validate_ka(opts.get("ka_s2", 0)),
                accel_cap_deg=JointPositionLead.validate_offset(
                    opts.get("accel_cap_deg", JointPositionLead.default_accel_cap_deg)),
                lead_ms=JointPositionLead.validate_time(opts.get("lead_ms", 0)),
                max_offset_deg=JointPositionLead.validate_offset(
                    opts.get("max_offset_deg", JointPositionLead.default_offset_deg)))
    live = opts["mode"] == "hardware"
    if opts["mode"] not in ("hardware", "simulation"):
        raise ValueError("Unknown mode")
    path_type = opts.setdefault("path_type", "x_sine")
    if path_type not in ("x_sine", "y_sine", "y_only"):
        raise ValueError("Unknown trajectory")
    hz, seconds, span = (float(opts[k]) for k in ("hz", "seconds", "span"))
    amp = 0. if path_type == "y_only" else float(opts["amp"])
    opts["amp"] = amp
    if hz not in (50, 100, 125, 200, 250) or not 4 <= seconds <= 40 or not 20 <= span <= 160 or (path_type != "y_only" and not 5 <= amp <= 60):
        raise ValueError("Settings outside supported range")
    if opts["ff"] is not None and type(opts["ff"]) is not bool:
        raise ValueError("Invalid feedforward selection")
    if live:
        if not (DATA / "origin.json").exists():
            raise ValueError("Capture the right arm origin first")
        q0 = np.array(json.loads((DATA / "origin.json").read_text())["q"])
    else:
        q0 = np.radians(body["home"]["joints_deg"]["right_arm"])
    start = tcp(q0)
    # Sample one spatial curve independently of playback rate. Retrace exactly.
    u = np.linspace(0, 1, 501)
    curve = np.repeat(start[None], len(u), axis=0)
    travel_axis = 0 if path_type == "x_sine" else 1
    curve[:,travel_axis,3] += u * span / 1000
    if path_type != "y_only":
        curve[:,1-travel_axis,3] += np.sin(u * 2*np.pi) * amp / 1000
    knots = [q0.copy()]
    for pose in curve[1:]:
        flange_target = model.flange_target(pose, np.array([0.]))
        q = knots[-1].copy()
        for _ in range(6):
            flange, jac = model.arm.flange_and_jacobian(q)
            error = np.r_[flange_target[:3,3]-flange[:3,3],
                          Rotation.from_matrix(flange_target[:3,:3] @ flange[:3,:3].T).as_rotvec()]
            if np.linalg.norm(error) < 1e-7:
                break
            q += jac.T @ np.linalg.solve(jac @ jac.T + np.eye(6)*1e-7, error)
        actual = tcp(q)
        if (np.linalg.norm(actual[:3,3]-pose[:3,3]) > .0003
                or Rotation.from_matrix(actual[:3,:3] @ pose[:3,:3].T).magnitude() > .002
                or not j67_ok(np.degrees(q)) or abs(q-knots[-1]).max() > .025):
            raise ValueError(f"Unreachable or discontinuous path at {len(knots)/5:.1f}%")
        knots.append(q)
    phase = np.linspace(0, 2*np.pi, round(seconds*hz)+1)
    pos = (1-np.cos(phase))/2
    joints = np.column_stack([np.interp(pos, u, col) for col in np.array(knots).T])
    if not all(j67_ok(row) for row in np.degrees(joints)):
        raise ValueError("J6/J7 interference bound reached")
    if np.any(joints < settings.joint_limits[0]) or np.any(joints > settings.joint_limits[1]):
        raise ValueError("Trajectory exceeds command joint limits")
    vmax = abs(np.diff(joints, axis=0)*hz).max()
    amax = abs(np.diff(joints,n=2,axis=0)*hz**2).max()
    if vmax > np.pi/2 or amax > 80:
        raise ValueError("Joint rate/acceleration limit exceeded. Increase duration or reduce excursion.")
    return dict(id=secrets.token_hex(8), opts=opts, q=joints, q0=q0, xy=curve[:,:2,3]*1000,
                peak=math.degrees(vmax), hz=int(hz))


class RightJointExecution:
    """Right-arm slice of the throw-ball chunk-IK rollout's post-IK execution."""
    group = "right_arm"
    velocity_source = "smooth_executor_output"

    def __init__(self, hz, measured, lead_ms=0, max_offset_deg=JointPositionLead.default_offset_deg, *,
                 ka_s2=0, accel_cap_deg=JointPositionLead.default_accel_cap_deg):
        config = cfg["executor"]
        ik = cfg["execution"]["ik"]
        if config["type"] != "smooth" or ik["mode"] != "chunk":
            raise ValueError("Probe requires the selected rollout's smooth chunk-IK configuration")
        dt_ns = round(1e9 / hz)
        smooth = deepcopy(config["smooth"])
        # Fixed tool coordinate preserves the rollout layout; only joints 1..7 go to Marvin.
        indices = smooth["gripper"]["group_indices"]
        smooth["gripper"]["group_indices"] = {self.group: indices[self.group]}
        executor = SmoothExecutor(smooth, dt_ns / 1e9)
        self.pipeline = ExecutionPipeline(executor, kinematics=None, ik_options=ik,
                                          control_dt_ns=dt_ns, motion_limits=config["motion_limits"])
        self.dt_ns = dt_ns
        safety = {k: {self.group: v[self.group]} if isinstance(v, dict) and v else v
                  for k, v in config["command_safety"].items()}
        self.guard = SafetyGuard({self.group: 8}, smooth["position_limit_abs"],
                                 control_dt_s=dt_ns / 1e9, **safety)
        state = RobotState({self.group: np.r_[measured, 0.]}, time.monotonic_ns(), 0)
        executor.reset(state)
        self.guard.reset(state)
        max_velocity = safety["max_velocity"].get(self.group)
        self.lead = JointPositionLead(lead_ms, dt_ns, max_offset_deg, ka_s2=ka_s2, accel_cap_deg=accel_cap_deg,
                                      max_velocity_rad_s=None if max_velocity is None else max_velocity[:7])
        self.issue_guard = None
        if self.lead.enabled:
            self.issue_guard = SafetyGuard({self.group: 8}, smooth["position_limit_abs"],
                                           control_dt_s=dt_ns / 1e9, **safety)
            self.issue_guard.reset(state)
        self.last_command = state.groups
        self.velocity_rad_s = np.zeros(7)

    def step(self, q_rows, measured, now_ns, serial):
        state = RobotState({self.group: np.r_[measured, 0.]}, now_ns, serial)
        self.guard.validate_state(state)
        ref = ActionHorizon(now_ns, self.dt_ns, "sine",
                            {self.group: np.column_stack((q_rows, np.zeros(len(q_rows))))})
        cmd = self.pipeline.step(now_ns, state, ref, last_command=self.last_command).command
        self.guard.validate_command(cmd)
        self.velocity_rad_s = self.pipeline.executor.output_velocity(self.group)[:7]
        self.last_command = cmd.groups
        return cmd.groups[self.group][:7]

    def position_target(self, nominal, now_ns, *, last_issued=None, missed_tick=False):
        target = self.lead.apply(nominal, now_ns, last_issued=last_issued,
                                 velocity_rad_s=self.velocity_rad_s, missed_tick=missed_tick)
        if self.issue_guard is not None:
            if not j67_ok(np.degrees(target)):
                raise ValueError("Position lead exceeds J6/J7 interference bound")
            if np.any(target < settings.joint_limits[0]) or np.any(target > settings.joint_limits[1]):
                raise ValueError("Position lead exceeds command joint limits")
            command = RobotCommand({self.group: np.r_[target, 0.]}, now_ns, "sine")
            self.issue_guard.validate_command(command)
        return target


def execute(p):
    global status
    opts, hz = p["opts"], p["hz"]
    live = opts["mode"] == "hardware"
    rows, sample_times = [], []
    run_id = f"{time.time_ns()}-{opts['mode']}"
    status = dict(phase="running", id=run_id, samples=0, raw_rows=rows)
    result = dict(id=run_id, opts=opts, origin_rad=p["q0"].tolist(), xy=p["xy"].tolist(), rows=rows,
                  columns=["elapsed_s","host_cmd_deg","feedback_deg","controller_cmd_deg","frame_serial",
                           "host_tcp_mm","feedback_tcp_mm","send_s","newly_issued_deg","trajectory_time_s","scheduled_slot","executor_reference_deg","marvin_inputs",
                           "nominal_cmd_deg","nominal_tcp_mm","new_nominal_cmd_deg","accel_offset_before_total_cap_deg",
                           "new_command_lead_recovery", "nominal_velocity_deg_s", "nominal_acceleration_deg_s2"])
    period_ns = round(1e9 / hz)
    slope_intervals = math.ceil(JointPositionLead.velocity_slope_window_ns / period_ns)
    result["position_lead"] = dict(time_ms=opts["lead_ms"], velocity_source=RightJointExecution.velocity_source,
                                   acceleration_source="velocity_linear_slope_nominal_ticks",
                                   accel_window_samples=slope_intervals + 1,
                                   max_offset_deg=opts["max_offset_deg"], ka_s2=opts["ka_s2"],
                                   accel_cap_deg=opts["accel_cap_deg"], accel_window_ms=slope_intervals * period_ns / 1e6)
    result["execution_path"] = "rollout_post_ik_v1"
    result["rollout_config"] = str(CONFIG)
    result["timing_basis"] = "monotonic_elapsed"
    skipped_slots = 0
    endpoint_send = None
    first_send = None
    motion_intervals = 0
    error = None
    try:
        with connected() if live else nullcontext(None) as arm:
            measured = p["q0"].copy()
            ff_attempted = False
            try:
                if live:
                    measured, _, _, _ = arm.sample()
                    if abs(measured-p["q0"]).max() > np.radians(.5):
                        raise ValueError("Right arm is away from captured origin. Capture and preview again.")
                    if stop.is_set() or time.monotonic()-heartbeat > 2:
                        raise RuntimeError("Stopped before enabling")
                    arm.send_commands({"right": measured.copy()})
                    if opts["ff"] is not None:
                        ff_attempted = True
                        arm.feedforward(round(1000/hz) if opts["ff"] else 0)
                execution = RightJointExecution(hz, measured, opts["lead_ms"], opts["max_offset_deg"],
                                                ka_s2=opts["ka_s2"], accel_cap_deg=opts["accel_cap_deg"])
                prev_accel_deg = [0.] * 7
                prev_velocity = prev_acceleration = [0.] * 7
                prev = measured.copy()
                prev_nominal = measured.copy()
                start = time.monotonic()
                diagnostics = None
                last_send = None
                slot = 0
                duration = float(opts["seconds"])
                # Include a short endpoint hold for settling measurements.
                while True:
                    deadline = start + slot/hz
                    if stop.wait(max(0, deadline-time.monotonic())) or time.monotonic()-heartbeat > 2:
                        raise RuntimeError("Operator stop or browser heartbeat lost")
                    now = time.monotonic()
                    if live and now-deadline > max(.015,2/hz):
                        raise RuntimeError("Send schedule overrun")
                    if live:
                        measured, ctl, serial, diagnostics = arm.sample()
                    phase_s = min(duration, time.monotonic() - start)
                    indices = np.minimum(duration, phase_s + np.arange(execution.pipeline.horizon_steps)/hz)
                    indices *= (len(p["q"]) - 1) / duration
                    lo = np.minimum(indices.astype(int), len(p["q"])-2)
                    q_rows = p["q"][lo] + (indices-lo)[:, None] * (p["q"][lo+1]-p["q"][lo])
                    if not live:
                        measured += (prev-measured)*(1-np.exp(-1/hz/.055))
                        ctl, serial = np.degrees(prev), slot
                    now_ns = round(now * 1e9)
                    nominal = execution.step(q_rows, measured, now_ns, serial)
                    missed_tick = bool(rows and slot != rows[-1][10] + 1)
                    q = execution.position_target(nominal, now_ns, last_issued=prev, missed_tick=missed_tick)
                    if live:
                        arm.send_commands({"right": q})
                    sent = time.monotonic()
                    # Compare feedback with the last issued target at feedback collection.
                    rows.append([now-start,np.degrees(prev).tolist(),np.degrees(measured).tolist(),
                                 ctl.tolist(),serial,(tcp(prev)[:3,3]*1000).tolist(),
                                 (tcp(measured)[:3,3]*1000).tolist(),sent-start,np.degrees(q).tolist(),phase_s,slot,np.degrees(q_rows[0]).tolist(),diagnostics,
                                 np.degrees(prev_nominal).tolist(),(tcp(prev_nominal)[:3,3]*1000).tolist(),
                                 np.degrees(nominal).tolist(),prev_accel_deg,execution.lead.recovery_start_ns is not None,
                                 prev_velocity, prev_acceleration])
                    prev_accel_deg = np.degrees(execution.lead.accel_offset).tolist()
                    prev_velocity = np.degrees(execution.velocity_rad_s).tolist()
                    prev_acceleration = np.degrees(execution.lead.acceleration_rad_s2).tolist()
                    if last_send is not None:
                        sample_times.append(1000*(sent-last_send))
                    last_send = sent
                    prev = q.copy()
                    prev_nominal = nominal.copy()
                    status = dict(phase="running", id=run_id, samples=len(rows), last=rows[-1], raw_rows=rows)
                    if first_send is None:
                        first_send = sent
                    if endpoint_send is None and phase_s >= duration:
                        endpoint_send = sent
                        motion_intervals = len(rows) - 1
                    if endpoint_send is not None and sent - endpoint_send >= .3:
                        break
                    # Stay on the original clock. Skip expired deadlines and leave
                    # at least half a period before issuing another command.
                    next_slot = max(slot+1, math.ceil((sent-start)*hz + .5))
                    skipped_slots += next_slot - slot - 1
                    slot = next_slot
            finally:
                if live and ff_attempted:
                    try:
                        arm.stop()
                    finally:
                        arm.feedforward(0)
    except Exception as exc:
        error = str(exc)
    finally:
        try:
            result["skipped_slots"] = skipped_slots
            if endpoint_send is not None:
                result["motion_duration_s"] = endpoint_send - first_send
                result["motion_send_hz"] = motion_intervals / result["motion_duration_s"]
            result.update(error=error, interval_ms_p50_p95_p99=np.percentile(sample_times,[50,95,99]).tolist() if sample_times else [])
            save(DATA / f"{run_id}.json", result)
            status = dict(phase="finished", id=run_id, error=error, samples=len(rows))
        finally:
            gate.release()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, data, code=200, mime="application/json"):
        raw = json.dumps(data, allow_nan=False).encode() if mime=="application/json" else data.encode()
        self.send_response(code)
        self.send_header("Content-Type", mime+"; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        import re
        if self.headers.get("Host") != self.server.host:
            return self.reply("Invalid host",403)
        if self.path == "/":
            html = Path(__file__).with_name("index.html").read_text()
            return self.reply(html.replace("{{TOKEN}}",token), mime="text/html")
        from urllib.parse import parse_qs, urlsplit
        url = urlsplit(self.path)
        if url.path == "/state":
            snapshot = status.copy()
            raw_rows = snapshot.pop("raw_rows", None)
            query = parse_qs(url.query)
            if raw_rows is not None and "offset" in query:
                try:
                    offset = int(query["offset"][0])
                except ValueError:
                    return self.reply("Invalid sample offset", 400)
                end = snapshot["samples"]
                if query.get("id", [""])[0] != snapshot["id"] or not 0 <= offset <= end:
                    offset = 0
                snapshot.update(offset=offset, rows=raw_rows[offset:end])
            return self.reply(dict(status=snapshot, origin=(DATA/"origin.json").exists(),
                                   runs=sorted((x.stem for x in DATA.glob("[0-9]*.json")),reverse=True)))
        if re.fullmatch(r"/run/[0-9]+-(simulation|hardware)", self.path):
            path = DATA / (self.path.split("/")[-1]+".json")
            if path.exists():
                return self.reply(json.loads(path.read_text()))
        self.reply("Missing",404)

    def do_POST(self):
        global preview, heartbeat
        if self.headers.get("Host") != self.server.host or self.headers.get("X-Probe") != token:
            return self.reply("Invalid request",403)
        try:
            size = int(self.headers.get("Content-Length",0))
            if not 0 < size <= 2048:
                raise ValueError("Invalid request length")
            obj = json.loads(self.rfile.read(size))
            if self.path == "/heartbeat":
                heartbeat = time.monotonic()
                return self.reply(True)
            if self.path == "/stop":
                stop.set()
                return self.reply(True)
            if not gate.acquire(False):
                return self.reply("Experiment busy",409)
            transferred = False
            try:
                if self.path == "/origin":
                    preview = None
                    return self.reply(capture())
                if self.path == "/preview":
                    preview = None
                    p = plan(obj)
                    preview = p
                    return self.reply(dict(id=p["id"],xy=p["xy"].tolist(),peak=p["peak"],path_type=p["opts"]["path_type"],
                                           velocity_source=RightJointExecution.velocity_source,
                                           lead_ms=p["opts"]["lead_ms"], max_offset_deg=p["opts"]["max_offset_deg"],
                                           ka_s2=p["opts"]["ka_s2"], accel_cap_deg=p["opts"]["accel_cap_deg"]))
                if self.path == "/start" and preview and preview["id"] == obj.get("id"):
                    p, preview = preview, None
                    stop.clear()
                    heartbeat = time.monotonic()
                    threading.Thread(target=execute,args=(p,),daemon=False).start()
                    transferred = True
                    return self.reply(True)
                raise ValueError("Preview the selected settings first")
            finally:
                if not transferred:
                    gate.release()
        except Exception as exc:
            self.reply(str(exc),400)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",type=int,default=8766)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1",args.port),Handler)
    server.host = f"127.0.0.1:{server.server_port}"
    print(f"http://{server.host}",flush=True)
    def shutdown(*_):
        stop.set()
        threading.Thread(target=server.shutdown,daemon=True).start()
    signal.signal(signal.SIGTERM,shutdown)
    signal.signal(signal.SIGINT,shutdown)
    try:
        server.serve_forever()
    finally:
        stop.set()
        server.server_close()


if __name__ == "__main__":
    main()
