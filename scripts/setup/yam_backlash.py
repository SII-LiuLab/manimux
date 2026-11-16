"""Receive-only YAM joint display and paired manual alignment measurements."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import signal
import threading
import time
import uuid
from collections import deque
from contextlib import suppress
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
POSITION_STEP_RAD = 25.0 / 65535
FRESH_S = 0.2


def utc_now():
    return datetime.now(UTC).isoformat()


class Measurements:
    """Keep actual receive times and persist every capture before acknowledging it."""

    def __init__(
        self, channels, mode, out=None, window_s=0.5, clock=time.monotonic, *, arm_role="follower"
    ):
        if arm_role not in ("follower", "leader"):
            raise ValueError("Unknown arm role")
        self.channels = channels
        self.mode = mode
        self.arm_role = arm_role
        self.clock = clock
        self.window_s = window_s
        self.lock = threading.RLock()
        self.started = clock()
        self.session_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
        self.source_error = None
        self.captures = []
        self.pairs = []
        self.pending = {}
        self.samples = {f"{arm}:J{j}": deque(maxlen=20000) for arm in channels for j in range(1, 7)}
        self.history = {key: deque(maxlen=300) for key in self.samples}
        self.sequences = dict.fromkeys(self.samples, 0)
        self.out = out / self.session_id if out is not None else None
        self.metadata = {
            "session_id": self.session_id,
            "created_at": utc_now(),
            "mode": mode,
            "arm_role": arm_role,
            "channels": channels,
            "window_s": window_s,
            "position_step_rad": POSITION_STEP_RAD,
            "position_step_deg": math.degrees(POSITION_STEP_RAD),
            "angle_meaning": "YAM CAN motor feedback; not an independent output angle measurement",
            "delta_definition": "decreasing approach mean minus increasing approach mean",
            "acquisition": "manual status queries" if mode == "read" else mode,
            "time_basis": "synthetic host monotonic time"
            if mode == "demo"
            else "host monotonic converted from kernel CAN receive timestamps",
        }
        if self.out is not None:
            self.out.mkdir(parents=True, exist_ok=False)
            (self.out / "WORKSPACE.md").write_text(
                f"# YAM {arm_role} manual alignment measurements\n\n"
                f"Created and updated: {utc_now()}. Canonical project: {ROOT}.\n"
                "Purpose: retain this standalone joint-feedback measurement session.\n"
                f"Arm role: {arm_role}. Source: {mode}. "
                "Demo samples are synthetic, not hardware evidence.\n"
                "session.json records provenance; events.jsonl retains captures, pairings and "
                "discarded pending selections. The UI exports CSV and JSON.\n"
                "No controller commands or calibration changes are performed here.\n"
                "Keep: unique measurement evidence; archive only after exporting/reviewing.\n",
                encoding="utf-8",
            )
            (self.out / "session.json").write_text(json.dumps(self.metadata, indent=2) + "\n")

    def error(self, message):
        with self.lock:
            self.source_error = message

    def ingest(self, arm, decoded, received):
        key = f"{arm}:J{decoded['joint']}"
        position = decoded["position_rad"]
        if key not in self.samples or not math.isfinite(position) or not math.isfinite(received):
            return
        with self.lock:
            samples = self.samples[key]
            if received > self.clock() + 0.01 or (samples and received <= samples[-1]["t"]):
                return
            self.sequences[key] += 1
            sample = {**decoded, "t": received, "sequence": self.sequences[key]}
            samples.append(sample)
            while samples and samples[0]["t"] < received - max(2, self.window_s + 1):
                samples.popleft()
            history = self.history[key]
            if not history or received - history[-1]["t"] >= 0.05:
                history.append({"t": received, "deg": math.degrees(position)})

    def window(self, key, now):
        values = [s for s in self.samples[key] if now - self.window_s <= s["t"] <= now]
        latest = self.samples[key][-1] if self.samples[key] else None
        reason = None
        if self.source_error:
            reason = self.source_error
        elif latest is None or now - latest["t"] > FRESH_S:
            reason = "No fresh CAN feedback"
        elif len(values) < 3 or values[-1]["t"] - values[0]["t"] < self.window_s * 0.8:
            reason = "Waiting for a complete measurement window"
        elif any(b["t"] - a["t"] > FRESH_S for a, b in zip(values, values[1:], strict=False)):
            reason = "Feedback gap inside measurement window"
        elif any(s["status"] not in (0, 1) for s in values):
            reason = "Motor reports a fault in the measurement window"
        positions = [s["position_rad"] for s in values]
        return {
            "ready": reason is None,
            "reason": reason,
            "samples": len(values),
            "mean_deg": math.degrees(sum(positions) / len(positions)) if positions else None,
            "span_deg": math.degrees(max(positions) - min(positions)) if positions else None,
        }, values

    def snapshot(self):
        with self.lock:
            now = self.clock()
            joints = []
            for key, samples in self.samples.items():
                arm, joint = key.split(":J")
                latest = samples[-1] if samples else None
                age = max(0, now - latest["t"]) if latest else None
                window, _ = self.window(key, now)
                joints.append(
                    {
                        "key": key,
                        "arm": arm,
                        "channel": self.channels[arm],
                        "joint": int(joint),
                        "motor_type": "DM4340" if int(joint) <= 3 else "DM4310",
                        "position_rad": latest["position_rad"] if latest else None,
                        "position_deg": math.degrees(latest["position_rad"]) if latest else None,
                        "status": latest["status"] if latest else None,
                        "age_s": age,
                        "fresh": age is not None and age <= FRESH_S,
                        "sequence": self.sequences[key],
                        "window": window,
                    }
                )
            return {
                "mode": self.mode,
                "arm_role": self.arm_role,
                "source_label": {
                    "demo": "Synthetic preview",
                    "passive": "Passive CAN feedback",
                    "read": "Manual status queries; no background motor queries or motor control",
                }[self.mode],
                "source_error": self.source_error,
                "session_id": self.session_id,
                "window_s": self.window_s,
                "joints": joints,
                "captures": list(self.captures),
                "pairs": list(self.pairs),
                "pending": dict(self.pending),
                "history": {
                    key: [
                        {"t": s["t"] - self.started, "deg": s["deg"]}
                        for s in values
                        if now - s["t"] <= 15
                    ]
                    for key, values in self.history.items()
                },
            }

    def persist(self, event):
        if self.out is not None:
            with (self.out / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

    def capture(self, key, direction, note=""):
        if key not in self.samples or direction not in ("increasing", "decreasing"):
            raise ValueError("Select a known joint and approach direction")
        if not isinstance(note, str) or len(note) > 500:
            raise ValueError("Note must be text with at most 500 characters")
        with self.lock:
            window, values = self.window(key, self.clock())
            if not window["ready"]:
                raise ValueError(window["reason"])
            arm, joint = key.split(":J")
            capture = {
                "id": len(self.captures) + 1,
                "joint_key": key,
                "arm": arm,
                "joint": int(joint),
                "direction": direction,
                "mean_deg": window["mean_deg"],
                "mean_rad": sum(s["position_rad"] for s in values) / len(values),
                "span_deg": window["span_deg"],
                "samples": len(values),
                "at": utc_now(),
                "note": note.strip(),
                "source": self.mode,
                "duration_s": values[-1]["t"] - values[0]["t"],
                "raw_samples": [dict(s) for s in values],
            }
            pair = None
            previous = self.pending.get(key)
            if previous and previous["direction"] != direction:
                by_direction = {direction: capture, previous["direction"]: previous}
                delta = (
                    by_direction["decreasing"]["mean_deg"] - by_direction["increasing"]["mean_deg"]
                )
                pair = {
                    "id": len(self.pairs) + 1,
                    "joint_key": key,
                    "arm": arm,
                    "joint": int(joint),
                    **by_direction,
                    "delta_deg": delta,
                    "abs_delta_deg": abs(delta),
                    "at": utc_now(),
                }
            # One durable event contains both sides of the state transition.
            self.persist({"type": "capture", "capture": capture, "pair": pair})
            self.captures.append(capture)
            if pair:
                self.pairs.append(pair)
                self.pending.pop(key)
            else:
                self.pending[key] = capture

    def discard(self, key):
        if key not in self.samples:
            raise ValueError("Unknown joint")
        with self.lock:
            if key in self.pending:
                self.persist(
                    {
                        "type": "discard_pending",
                        "joint_key": key,
                        "at": utc_now(),
                        "capture_id": self.pending[key]["id"],
                    }
                )
                self.pending.pop(key)

    def export(self):
        with self.lock:
            return {
                "metadata": dict(self.metadata),
                "captures": list(self.captures),
                "pairs": list(self.pairs),
                "pending": dict(self.pending),
            }

    def csv(self):
        output = io.StringIO(newline="")
        fields = [
            "session_id",
            "source",
            "arm_role",
            "channel",
            "pair_id",
            "arm",
            "joint",
            "increasing_deg",
            "decreasing_deg",
            "delta_deg",
            "abs_delta_deg",
            "increasing_span_deg",
            "decreasing_span_deg",
            "increasing_samples",
            "decreasing_samples",
            "at",
            "increasing_note",
            "decreasing_note",
        ]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for pair in self.export()["pairs"]:
            row = {key: pair[key] for key in ("arm", "joint", "delta_deg", "abs_delta_deg", "at")}
            row.update(
                session_id=self.session_id,
                source=self.mode,
                arm_role=self.arm_role,
                channel=self.channels[pair["arm"]],
                pair_id=pair["id"],
            )
            for direction in ("increasing", "decreasing"):
                for suffix, source in (
                    ("deg", "mean_deg"),
                    ("span_deg", "span_deg"),
                    ("samples", "samples"),
                    ("note", "note"),
                ):
                    value = pair[direction][source]
                    # Notes remain data when opened in spreadsheet software.
                    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                        value = "'" + value
                    row[f"{direction}_{suffix}"] = value
            writer.writerow(row)
        return output.getvalue()


class DemoSource:
    def __init__(self, measurements):
        self.measurements = measurements
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        self.thread.start()

    def run(self):
        started = time.monotonic()
        while not self.stop.is_set():
            now = time.monotonic()
            for arm in self.measurements.channels:
                for joint in range(1, 7):
                    angle = math.radians(
                        (joint - 3) * 12
                        + (5 if arm == "right" else 0)
                        + 0.15 * math.sin((now - started) * 0.8 + joint)
                    )
                    raw = round((angle + 12.5) / POSITION_STEP_RAD)
                    self.measurements.ingest(
                        arm,
                        {
                            "joint": joint,
                            "position_rad": raw * POSITION_STEP_RAD - 12.5,
                            "raw_position": raw,
                            "status": 1,
                        },
                        now,
                    )
            self.stop.wait(0.02)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)


def read_on_demand(measurements, source, *, joint_key=None, duration_s=0.0):
    """Request actual samples on a user action, without any persistent polling loop."""
    keys = [joint_key] if joint_key else list(measurements.samples)
    if any(key not in measurements.samples for key in keys):
        raise ValueError("Unknown joint")
    started = time.monotonic()
    while True:
        with measurements.lock:
            before = {key: measurements.sequences[key] for key in keys}
        # A separate browser may have just requested a round. Respect the source's
        # 10 Hz limit rather than treating an untransmitted request as a sample.
        while not source.refresh():
            time.sleep(0.01)
        if time.monotonic() - started >= duration_s:
            break
        time.sleep(0.11)
    deadline = time.monotonic() + 0.15
    while True:
        with measurements.lock:
            missing = [key for key in keys if measurements.sequences[key] <= before[key]]
        if not missing or time.monotonic() >= deadline:
            break
        time.sleep(0.005)
    if missing:
        raise ValueError("No new status reply: " + ", ".join(missing))


def handler_for(measurements, source=None, *, html_path=None):
    acquisition_lock = threading.Lock()
    page = Path(html_path) if html_path is not None else Path(__file__).with_suffix(".html")
    export_prefix = "leader-backlash" if measurements.arm_role == "leader" else "backlash"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, body, content_type="application/json", status=200, filename=None):
            if not isinstance(body, (str, bytes)):
                body = json.dumps(body, allow_nan=False)
            if isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            with suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self.respond(page.read_bytes(), "text/html")
            elif path == "/api/state":
                state = measurements.snapshot()

                # Raw capture samples are preserved in exports, not repeated on each UI poll.
                def compact(value):
                    if isinstance(value, dict):
                        return {k: compact(v) for k, v in value.items() if k != "raw_samples"}
                    if isinstance(value, list):
                        return [compact(v) for v in value]
                    return value

                self.respond(compact(state))
            elif path == "/api/export.json":
                self.respond(
                    measurements.export(),
                    filename=f"{export_prefix}-{measurements.session_id}.json",
                )
            elif path == "/api/export.csv":
                self.respond(
                    measurements.csv(),
                    "text/csv",
                    filename=f"{export_prefix}-{measurements.session_id}.csv",
                )
            else:
                self.respond({"error": "Not found"}, status=404)

        def do_POST(self):
            # Browser requests must originate from this local UI.
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{self.headers.get('Host')}":
                self.respond({"error": "Cross-origin request rejected"}, status=403)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("Invalid request length")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("Expected a JSON object")
                key = payload.get("joint_key", "")
                if not isinstance(key, str):
                    raise ValueError("Invalid joint key")
                if self.path == "/api/capture":
                    with acquisition_lock:
                        if measurements.mode == "read":
                            if key not in measurements.samples or payload.get("direction") not in (
                                "increasing",
                                "decreasing",
                            ):
                                raise ValueError("Select a known joint and approach direction")
                            read_on_demand(
                                measurements,
                                source,
                                joint_key=key,
                                duration_s=measurements.window_s + 0.1,
                            )
                        measurements.capture(key, payload.get("direction"), payload.get("note", ""))
                elif self.path == "/api/refresh":
                    if measurements.mode != "read":
                        raise ValueError("Manual refresh requires --mode read")
                    with acquisition_lock:
                        read_on_demand(measurements, source)
                elif self.path == "/api/discard":
                    measurements.discard(key)
                else:
                    self.respond({"error": "Not found"}, status=404)
                    return
                self.respond({"ok": True})
            except (ValueError, TypeError, RuntimeError) as error:
                self.respond({"error": str(error)}, status=400)
            except OSError:
                self.respond(
                    {"error": "Read/save failed; measurement was not accepted"}, status=500
                )

    return Handler


def resolve_channels(station, arms, arm_role):
    """Resolve explicit role bindings without falling back to the other arm pair."""
    if arm_role not in ("follower", "leader"):
        raise ValueError("Unknown arm role")
    suffix = "_yam_leader" if arm_role == "leader" else "_yam"
    try:
        components = station["robot"]["components"]
        channels = {arm: components[f"{arm}{suffix}"]["channel"] for arm in arms}
    except (KeyError, TypeError) as error:
        raise ValueError(f"Station needs robot.components.<left/right>{suffix}.channel") from error
    if any(not isinstance(channel, str) or not channel for channel in channels.values()):
        raise ValueError("Each selected arm needs a CAN channel in the station file")
    if len(set(channels.values())) != len(channels):
        raise ValueError("Selected arms must use distinct CAN channels")
    if arm_role == "leader":
        follower_channels = {
            components.get(f"{arm}_yam", {}).get("channel") for arm in ("left", "right")
        }
        if set(channels.values()) & follower_channels:
            raise ValueError("Leader bindings must not reuse a configured follower channel")
    return channels


def main(arm_role="follower"):
    if arm_role not in ("follower", "leader"):
        raise ValueError("Unknown arm role")
    leader = arm_role == "leader"
    page = Path(__file__).with_name("yam_leader_backlash.html" if leader else "yam_backlash.html")
    parser = argparse.ArgumentParser(
        description=f"YAM {arm_role} joint readout and alignment measurements"
    )
    parser.add_argument(
        "--mode",
        choices=("demo", "passive", "read"),
        required=True,
        help="demo: synthetic; passive: existing feedback; read: status-only queries",
    )
    parser.add_argument("--local", type=Path, default=ROOT / "manimux/configs/local/station.yaml")
    parser.add_argument("--arms", nargs="+", choices=("left", "right"), default=["left", "right"])
    parser.add_argument("--port", type=int, default=8769 if leader else 8768)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / (".local/backlash-leader" if leader else ".local/backlash"),
    )
    parser.add_argument(
        "--window", type=float, default=0.5, help="Trailing capture window in seconds"
    )
    parser.add_argument(
        "--check", action="store_true", help="Check files and bindings without CAN or HTTP sockets"
    )
    args = parser.parse_args()
    if not math.isfinite(args.window) or not 0.2 <= args.window <= 2:
        parser.error("--window must be between 0.2 and 2 seconds")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    channels = {arm: f"demo_{arm}" for arm in args.arms}
    if args.mode != "demo":
        import yaml

        station = yaml.safe_load(args.local.read_text())
        try:
            channels = resolve_channels(station, args.arms, arm_role)
        except ValueError as error:
            parser.error(str(error))
    if not page.is_file():
        parser.error(f"Missing {page.name}")
    if args.check:
        print(
            json.dumps(
                {
                    "check": "ok",
                    "mode": args.mode,
                    "arm_role": arm_role,
                    "channels": channels,
                    "port": args.port,
                    "hardware_opened": False,
                    "window_s": args.window,
                }
            )
        )
        return
    measurements = Measurements(channels, args.mode, args.out, args.window, arm_role=arm_role)
    if args.mode == "demo":
        source = DemoSource(measurements)
    else:
        if __package__:
            from scripts.setup.yam_backlash_can import PassiveCanSource, ReadQueryCanSource
        else:
            from yam_backlash_can import PassiveCanSource, ReadQueryCanSource
        source_type = ReadQueryCanSource if args.mode == "read" else PassiveCanSource
        source = source_type(channels, measurements.ingest, measurements.error)
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port), handler_for(measurements, source, html_path=page)
    )

    def interrupt(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        source.start()
        print(
            json.dumps(
                {
                    "url": f"http://127.0.0.1:{args.port}",
                    "mode": args.mode,
                    "arm_role": arm_role,
                    "channels": channels,
                    "out": str(measurements.out),
                }
            ),
            flush=True,
        )
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
        server.server_close()


if __name__ == "__main__":
    main()
