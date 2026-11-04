"""ManiMux-owned per-row IK and high-rate motor control, independent of Isaac."""

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import threading
import time

from manimux.embodiments.asyncsim import AsyncSimClient
from manimux.policies import PolicyCapabilities
from manimux.policies.worker import PolicyWorkerClient
from manimux.runtime.asyncsim import AsyncSimRuntime
from manimux.runtime.motor_ik import MotorIKClient, limit_grippers, validate_motor_config
from manimux.types import ActionContext, InferenceRequest

MOTOR_STREAM = "proprio.motor_joint_state"
TERMINAL = {"succeeded", "failed", "finished"}


def source_slot(sim_ts, start_ts, dt=.004):
    return max(0, math.floor((sim_ts - start_ts + 1e-9) / dt))


class MotorPublisher:
    """Only this loop samples motor targets; inference and IK never run here."""

    def __init__(self, endpoint, episode_id, planner, *, client=None):
        self.client = client or AsyncSimClient(endpoint, timeout_s=30)
        self.episode_id, self.planner = episode_id, planner
        self.robots = planner.config["robots"]
        self.stop = threading.Event(); self.active = threading.Event(); self.done = threading.Event()
        self.ready = threading.Event(); self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name="manimux-motor-250hz", daemon=True)
        self.counter = 0; self.error = None; self.last_groups = None
        self.commands = []; self.solves = []; self.events = []
        self.chunk = self.pending = None
        self.last_slot = -1

    def start(self):
        self.thread.start()
        if not self.ready.wait(60):
            raise TimeoutError("motor connection did not become ready")
        self.check()

    def check(self):
        if self.error is not None:
            raise RuntimeError("motor control failed: " + repr(self.error)) from self.error

    def issue(self, groups, sim_ts, plan_id, client, *, row, tick, measured):
        with self.lock:
            self.counter += 1
            limited = limit_grippers(groups, measured, self.robots)
            command = {"episode_id": self.episode_id, "command_seq": self.counter,
                "issued_sim_ts": sim_ts, "apply_after_sim_ts": sim_ts,
                "expires_at_sim_ts": sim_ts + 35, "plan_id": plan_id,
                "action_space": "joint_position", "groups": limited}
            started = time.monotonic_ns()
            ack = client.submit_command(command)
            if not ack["accepted"]:
                raise RuntimeError("motor command rejected: " + ack.get("reason", "unknown"))
            self.last_groups = deepcopy(limited)
            self.commands.append({**command, "accepted": True, "wall_ns": started,
                "ack_wall_ns": time.monotonic_ns(), "source_index": row, "interpolation_tick": tick,
                "measured_motor_state": deepcopy(measured)})
            return ack

    def job(self, chunk, row, sim_ts, measured):
        return {"plan_id": chunk.plan_id, "source_index": row, "seed_sim_ts": sim_ts,
            "groups": {k: v[row].tolist() for k, v in chunk.groups.items()},
            "measured": deepcopy(measured), "held": deepcopy(self.last_groups or measured)}

    def begin(self, chunk, first, sim_ts, main_client):
        if self.active.is_set() or self.pending is not None:
            raise RuntimeError("previous motor plan is not drained")
        if chunk.dt_ns != 40_000_000 or chunk.action_space != "eef_pose":
            raise ValueError("motor runtime requires unchanged 25Hz EEF source rows")
        self.issue(first["sequence"][0], sim_ts, chunk.plan_id, main_client, row=0, tick=0, measured=first["measured"])
        self.chunk, self.current = chunk, first
        self.start_ts, self.last_slot, self.requested_row = sim_ts, 0, 0
        self.solves.append(first); self.done.clear(); self.active.set()

    def _run(self):
        try:
            self.client.connect(); self.client.attach(self.episode_id)
            self.client.subscribe([MOTOR_STREAM]); self.ready.set()
            next_health = 0.
            while not self.stop.is_set():
                if not self.active.wait(.01):
                    continue
                if self.stop.is_set():
                    break
                if time.monotonic() >= next_health:
                    health = self.client.health(); next_health = time.monotonic() + .2
                    if health.get("last_error"):
                        raise RuntimeError(health["last_error"])
                    if health["episode"]["state"] in TERMINAL:
                        self.active.clear(); self.done.set(); break
                snapshot = self.client.read_snapshot([MOTOR_STREAM])
                now = snapshot["sim_ts"]
                packet = snapshot["packets"][MOTOR_STREAM]["packet"]
                if packet is None:
                    time.sleep(.001); continue
                measured = packet["payload"].get(0, packet["payload"].get("0"))
                slot = source_slot(now, self.start_ts)
                if self.pending is not None:
                    solved = self.planner.poll()
                    if solved is not None:
                        self.solves.append(solved)
                        if source_slot(now, self.start_ts, .04) == solved["source_index"]:
                            self.current = solved
                        else:
                            self.events.append({"kind": "late_ik_discarded", "sim_ts": now,
                                                "source_index": solved["source_index"]})
                        self.pending = None
                row = source_slot(now, self.start_ts, .04)
                if row >= self.chunk.horizon_steps:
                    if self.pending is None:
                        self.active.clear(); self.done.set()
                    else:
                        time.sleep(.001)
                    continue
                if row > self.requested_row and self.pending is None:
                    if row > self.requested_row + 1:
                        self.events.append({"kind": "source_rows_skipped", "count": row-self.requested_row-1})
                    self.pending = self.job(self.chunk, row, packet["capture_ts"], measured)
                    self.planner.submit(self.pending); self.requested_row = row
                if slot > self.last_slot:
                    if slot > self.last_slot + 1:
                        self.events.append({"kind": "motor_control_periods_skipped",
                            "count": slot-self.last_slot-1, "sim_ts": now})
                    tick = min(9, max(0, slot - self.current["source_index"] * 10))
                    groups = self.current["sequence"][tick]
                    try:
                        self.issue(groups, now, self.chunk.plan_id, self.client,
                                   row=self.current["source_index"], tick=tick, measured=measured)
                    except RuntimeError:
                        health = self.client.health()
                        if health["episode"]["state"] not in TERMINAL or health.get("last_error"):
                            raise
                        self.active.clear(); self.done.set(); break
                    self.last_slot = slot
                time.sleep(.001)
        except Exception as error:
            self.error = error; self.ready.set(); self.done.set()

    def close(self):
        self.stop.set(); self.active.set(); self.thread.join(timeout=65)
        if self.thread.is_alive():
            raise RuntimeError("motor thread failed to stop")
        self.client.close()

    def audit(self, paused):
        intervals = [(b["wall_ns"]-a["wall_ns"])/1e6 for a,b in zip(self.commands,self.commands[1:])
                     if a["plan_id"] == b["plan_id"]]
        return {"action_space": "joint_position", "command_semantics": "physical_motor_positions",
            "manimux_ik": True, "manimux_interpolation": True, "robodojo_interpolation": False,
            "controller_location": "manimux_runtime", "control_hz_target": 250,
            "policy_action_hz": 25, "physics_during_inference": not paused,
            "ik_seed": "current_measured_joint_state", "interpolation_rule": "linear_80pct_then_hold",
            "planner": "manimux.policy_adapter.robodojo_compat.RoboDojoCompatibleDecoder",
            "runtime_pid": os.getpid(), "ik_worker_pid": self.planner.pid,
            "robots": self.robots, "gripper_control": "measured_20pct_range_limit_and_mimic_in_runtime",
            "commands": self.commands, "solves": self.solves, "events": self.events,
            "wall_command_intervals_ms": intervals}


class CompatibleMotorRuntime(AsyncSimRuntime):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        if config["robot"]["control_hz"] != 250 or config["inference"]["algorithm"] != "serial":
            raise ValueError("compatible motor runtime requires 250Hz/serial")
        if self.ablation_mode != "fresh_paused":
            raise ValueError("runtime motor pilot is scoped to the aligned/paused comparison")
        self.motor_config = validate_motor_config(json.loads(
            Path(config["robot"]["options"]["motor_control_config"]).read_text()))

    def run_policy(self, *, seed=None, max_steps=None, timeout_s=None, recording_directory=None):
        worker = self.worker or PolicyWorkerClient(self.config["policy"], self.session_id)
        planner = MotorIKClient(self.motor_config, Path(recording_directory or ".") / "motor-ik.log")
        publisher = None; events = []; accepted = rejected = 0; inference_ms = 0.
        started = time.monotonic(); episode = None
        def timeout():
            if timeout_s is not None and time.monotonic()-started > timeout_s:
                raise TimeoutError("runtime motor episode exceeded wall timeout")
        try:
            worker.start(); self._check_capabilities(worker); planner.start()
            print(json.dumps({"event": "runtime_motor_ik_ready", "runtime_pid": os.getpid(),
                "ik_worker_pid": planner.pid, "control_hz_target": 250, "policy_action_hz": 25}), flush=True)
            episode = self.start(seed=seed, recording_directory=recording_directory)
            publisher = MotorPublisher(self.client.endpoint, episode["episode_id"], planner)
            publisher.start()
            request_seq = 0
            while True:
                timeout(); publisher.check()
                health = self.client.health()
                if health.get("last_error"):
                    raise RuntimeError(health["last_error"])
                if health["episode"]["state"] in TERMINAL:
                    break
                initial = self.client.read_snapshot(self.streams)
                if any(v["packet"] is None for v in initial["packets"].values()):
                    time.sleep(.01); continue
                request_seq += 1
                wire = self.client.policy_snapshot(self.streams, mode=self.ablation_mode,
                    request_seq=request_seq, timeout_s=min(3600., self.config["policy"]["timeout_s"]+60))
                observation = self.observe(wire)
                now_ns = observation.state.monotonic_ns
                view = self.adapter.build_observation(observation)
                deadline = time.monotonic_ns()+round(self.config["policy"]["timeout_s"]*1e9)
                request = self.adapter.prepare_request(InferenceRequest(self.session_id,request_seq,now_ns,
                    deadline,view,self._instruction))
                events.append({"kind":"inference_submitted","request_seq":request_seq,
                    "ablation_mode":self.ablation_mode,"input_snapshot_ts":wire["sim_ts"],
                    "input_streams":{k:{"capture_ts":v["packet"]["capture_ts"],"age_s":v["age_s"]}
                                     for k,v in wire["packets"].items()}})
                worker.submit_latest(request)
                response = None
                while response is None:
                    timeout()
                    if not worker.is_alive or time.monotonic_ns()>deadline:
                        raise RuntimeError("policy worker stopped or exceeded inference deadline")
                    response = worker.poll(); time.sleep(.005)
                if response.error or response.request_seq != request_seq or response.session_id != self.session_id:
                    raise RuntimeError(f"invalid policy response: {response.error}")
                inference_ms += response.inference_ms
                chunk = self.adapter.decode_action(response.raw_action, ActionContext(request_seq,now_ns,
                    response.finished_time_ns,execution_time_ns=now_ns,max_source_steps=32))
                events.append({"kind":"plan_committed","plan_id":chunk.plan_id,
                    "request_seq":request_seq,"groups":{k:v.tolist() for k,v in chunk.groups.items()}})
                first = planner.decode(publisher.job(chunk,0,wire["sim_ts"],wire["packets"][MOTOR_STREAM]["packet"]["payload"].get(
                    0,wire["packets"][MOTOR_STREAM]["packet"]["payload"].get("0"))))
                self.client.resume_with_next_command()
                publisher.begin(chunk,first,wire["sim_ts"],self.client); accepted += 1
                if accepted == 1:
                    print(json.dumps({"event": "runtime_motor_control_started", "sim_ts": wire["sim_ts"],
                        "action_space": "joint_position", "controller_location": "manimux_runtime"}), flush=True)
                while not publisher.done.wait(.1):
                    timeout(); publisher.check()
                publisher.check()
            audit = {"episode_id":episode["episode_id"],"snapshots":self._snapshot_audit,
                "manimux_events":events,"actions":publisher.commands,
                "execution_contract":publisher.audit(True)}
            return {"session_id":self.session_id,"episode_id":episode["episode_id"],
                "steps":publisher.counter,"accepted_plans":accepted,"rejected_plans":rejected,
                "rejection_reasons":[],"asyncsim_result":self.client.result(),
                "asyncsim_metrics":self.client.metrics(),"audit":audit,
                "backend_metadata":getattr(worker,"capabilities",PolicyCapabilities()).backend_metadata,
                "inference_time_s":inference_ms/1000,"wall_elapsed_s":time.monotonic()-started,
                "packets":tuple(self._selected_packets.values())}
        finally:
            try:
                if publisher is not None:
                    publisher.close()
            finally:
                try:
                    planner.close()
                finally:
                    worker.close(); self.close()
