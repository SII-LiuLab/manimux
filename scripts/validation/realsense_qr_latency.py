#!/usr/bin/env python3
"""Preview ManiMux RGB capture and measure timestamp-QR-to-host latency.

Run from the repository with ``python -m scripts.validation.realsense_qr_latency``.
No camera is opened by --print-config. Normal execution opens ONE RGB camera.
The QR must be displayed by the same computer (same Unix clock).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import threading
import time

import numpy as np

from manimux.cli import bind_station, read_camera_recipe
from manimux.embodiments.sensor import build_camera
from manimux.servers.camera.server import camera_config

ROOT = Path(__file__).resolve().parents[2]
ALIASES = {"left": "d405_left", "front": "d405_front", "right": "d405_right"}
FIELDS = ("sequence", "received_unix_s", "qr_unix_s", "qr_to_host_ms",
          "cache_age_at_decode_ms", "decode_ms", "status", "actual_exposure_us",
          "sdk_timestamp_ms", "timestamp_domain", "sensor_timestamp_us", "frame_timestamp_us",
          "sdk_arrival_unix_ms", "global_to_sdk_ms", "sdk_to_python_ms", "global_to_python_ms")


def sdk_timing(metadata, received_s):
    """Compare same-frame timestamps; never subtract an unmapped device clock."""
    meta = metadata or {}
    result = dict.fromkeys(("global_to_sdk_ms", "sdk_to_python_ms", "global_to_python_ms"), "")
    arrival = meta.get("sdk_arrival_unix_ms")
    if arrival is not None:
        result["sdk_to_python_ms"] = received_s * 1000 - arrival
    stamp = meta.get("sdk_timestamp_ms")
    if meta.get("timestamp_domain") == "timestamp_domain.global_time" and stamp is not None:
        result["global_to_python_ms"] = received_s * 1000 - stamp
        if arrival is not None:
            result["global_to_sdk_ms"] = arrival - stamp
    return result


def measurement(payload, received_s, age_ms, decode_ms, sequence, metadata=None):
    """Keep invalid timestamps visible, but exclude them from latency statistics."""
    row = dict.fromkeys(FIELDS, "")
    row.update(sequence=sequence, received_unix_s=received_s,
               cache_age_at_decode_ms=age_ms, decode_ms=decode_ms, status="no_qr")
    row.update(metadata or {})
    row.update(sdk_timing(metadata, received_s))
    if payload:
        try:
            stamp = float(payload)
        except ValueError:
            row["status"] = "invalid_payload"
            return row
        if not math.isfinite(stamp) or stamp < 1_000_000_000:
            row["status"] = "invalid_timestamp"
            return row
        row.update(qr_unix_s=stamp, qr_to_host_ms=(received_s - stamp) * 1000)
        row["status"] = "ok" if received_s >= stamp else "negative_clock_difference"
    return row


def resolve(args):
    config = bind_station({"camera_server": read_camera_recipe(args.config)}, args.local)
    name = ALIASES.get(args.camera, args.camera)
    spec = camera_config(config)["sensors"]["cameras"][name]
    if spec["implementation"] != "manimux.embodiments.sensor.realsense:RealSenseSensor":
        raise ValueError("Select a RealSense camera")
    # Explicit user requirement, even if another recipe enables depth.
    spec.update(enable_depth=False, align_depth=False, collect_metadata=True)
    if not spec.get("background", True):
        raise ValueError("This diagnostic requires the ManiMux background capture setting")
    return name, spec


def show_qr(hz, size):
    """Display a same-host Unix timestamp using OpenCV's installed QR encoder."""
    import cv2

    if hz <= 0 or size < 200:
        raise ValueError("QR refresh rate must be positive and size >= 200")
    encoder = cv2.QRCodeEncoder_create()
    title = "Timestamp QR | q/Esc: quit"
    deadline = time.perf_counter()
    started = deadline
    count = 0
    try:
        while True:
            time.sleep(max(0, deadline - time.perf_counter()))
            stamp = f"{time.time():.6f}"
            qr = encoder.encode(stamp)
            qr = cv2.copyMakeBorder(qr, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
            qr = cv2.resize(qr, (size, size), interpolation=cv2.INTER_NEAREST)
            qr = cv2.copyMakeBorder(qr, 0, 40, 0, 0, cv2.BORDER_CONSTANT, value=255)
            cv2.putText(qr, stamp, (12, size + 27), cv2.FONT_HERSHEY_SIMPLEX, .7, 0, 1)
            cv2.imshow(title, qr)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
            count += 1
            if count % max(1, round(hz)) == 0:
                print(f"QR submission rate: {count / (time.perf_counter() - started):.1f} Hz "
                      "(not measured monitor refresh)", flush=True)
            deadline += 1 / hz
            if deadline < time.perf_counter():
                deadline = time.perf_counter()
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="front", help="left/front/right or recipe stream name")
    parser.add_argument("--config", type=Path, default=ROOT / "manimux/configs/embodiment/sensor/cameras/realsense_3_views.yaml")
    parser.add_argument("--local", type=Path, default=ROOT / "manimux/configs/local/station.yaml")
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--show-qr", action="store_true", help="only display timestamp QR; opens no camera")
    parser.add_argument("--hz", type=float, default=60, help="QR display target update rate")
    parser.add_argument("--size", type=int, default=800, help="QR display width in pixels")
    parser.add_argument("--seconds", type=float, default=0, help="0: until q/Esc")
    parser.add_argument("--csv", type=Path, help="new CSV file; existing files are never overwritten")
    args = parser.parse_args()
    if args.show_qr:
        show_qr(args.hz, args.size)
        return
    name, spec = resolve(args)
    print(json.dumps({"camera": name, **spec}, indent=2), flush=True)
    if args.print_config:
        return
    import cv2

    # Check wall-clock stability. Frame receipt Unix time comes from capture(),
    # before metadata reads or decoding; never substitute decoder completion time.
    before = time.monotonic_ns()
    wall_s = time.time()
    after = time.monotonic_ns()
    unix_offset = wall_s - (before + after) / 2e9
    stop = threading.Event()
    lock = threading.Lock()
    state = {"latest": None, "error": None, "attempts": 0, "valid": 0}
    latencies = []
    camera = build_camera(name, spec)
    csv_file = args.csv.open("x", newline="") if args.csv else None
    writer = csv.DictWriter(csv_file, fieldnames=FIELDS) if csv_file else None
    if writer:
        writer.writeheader()

    def decode_loop():
        detector = cv2.QRCodeDetector()
        previous = None
        try:
            while not stop.is_set():
                frame = camera.read_capture()
                if frame.sequence == previous:
                    stop.wait(0.002)
                    continue
                previous = frame.sequence
                start = time.monotonic_ns()
                image = cv2.cvtColor(frame.image, cv2.COLOR_RGB2GRAY)
                payload, _, _ = detector.detectAndDecode(image)
                elapsed = (time.monotonic_ns() - start) / 1e6
                row = measurement(payload, frame.unix_s,
                                  (start - frame.monotonic_ns) / 1e6,
                                  elapsed, frame.sequence, frame.metadata)
                with lock:
                    state["latest"] = row
                    state["attempts"] += 1
                    if row["status"] == "ok":
                        latencies.append(row["qr_to_host_ms"])
                        state["valid"] += 1
                if writer:
                    writer.writerow(row)
                    csv_file.flush()
        except Exception as exc:
            with lock:
                state["error"] = exc
            stop.set()

    worker = None
    try:
        camera.start()
        worker = threading.Thread(target=decode_loop, name="qr_decoder", daemon=True)
        worker.start()
        title = f"RGB only: {name} | q/Esc: quit"
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, spec["width"], spec["height"])
        started = last_print = time.monotonic()
        print("Point this camera at this script's --show-qr window on THIS computer.\n"
              "QR-to-host includes QR generation/display, exposure and USB/capture.\n"
              "It excludes decoder and this preview's display latency. No fixed subtraction.\n"
              "Actual exposure and raw device timestamps are in us; SDK timestamp is in ms.\n"
              "SDK arrival is host Unix ms, truncated to 1 ms in librealsense 2.58.3.\n"
              "Same-frame global->SDK + SDK->Python = global->Python; negative gaps are retained.\n"
              "N/A means unavailable. Raw device timestamps are NOT Unix host timestamps.\n"
              "Preview and decoder read independent latest-frame snapshots; slow decoding skips frames.", flush=True)
        while not stop.is_set():
            frame = camera.read_capture()
            preview = cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR)
            with lock:
                row = state["latest"]
                count, attempts = state["valid"], state["attempts"]
                values = list(latencies) if time.monotonic() - last_print >= 1 else None
            age = (time.monotonic_ns() - frame.monotonic_ns) / 1e6
            lines = [f"RGB frame {frame.sequence} | cache age {age:.1f} ms",
                     f"Decoded {count}/{attempts} sampled frames"]
            if row and row["status"] == "ok":
                lines += [f"Last QR-to-host: {row['qr_to_host_ms']:.1f} ms (frame {row['sequence']})",
                          f"Decode: {row['decode_ms']:.1f} ms (excluded)"]
            else:
                lines.append(f"QR: {row['status'] if row else 'waiting'}")
            meta = frame.metadata or {}
            timing = sdk_timing(meta, frame.unix_s)
            timing_text = " | ".join(
                f"{key}={'N/A' if number == '' else f'{number:.1f}'}"
                for key, number in timing.items())
            def value(key):
                return "N/A" if meta.get(key) is None else meta[key]
            exposure = meta.get("actual_exposure_us")
            exposure_text = "N/A" if exposure is None else f"{exposure} us ({exposure / 1000:.3f} ms)"
            lines += [f"Actual exposure: {exposure_text}",
                      f"SDK ts: {value('sdk_timestamp_ms')} ms",
                      f"Clock: {value('timestamp_domain')}",
                      f"Exposure midpoint (device): {value('sensor_timestamp_us')} us",
                      f"Readout start (device): {value('frame_timestamp_us')} us",
                      f"Host RGB received: {frame.unix_s:.6f} s",
                      f"SDK arrival: {value('sdk_arrival_unix_ms')} ms"]
            lines += [f"{key}: {'N/A' if number == '' else f'{number:.1f} ms'}"
                      for key, number in timing.items()]
            for i, label in enumerate(lines):
                y = 22 + i * 23
                cv2.putText(preview, label, (8, y), cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 0, 0), 3)
                cv2.putText(preview, label, (8, y), cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 255, 0), 1)
            cv2.imshow(title, preview)
            if values is not None:
                stats = "no valid QR" if not values else "p50/p95/min/max = " + "/".join(
                    f"{x:.1f}" for x in np.percentile(values, [50, 95, 0, 100])) + " ms"
                print(f"{count}/{attempts} decoded; {stats}", flush=True)
                print(f"RGB frame={frame.sequence} actual_exposure={exposure_text} "
                      f"sdk_timestamp_ms={value('sdk_timestamp_ms')} domain={value('timestamp_domain')} "
                      f"sensor_timestamp_us={value('sensor_timestamp_us')} "
                      f"frame_timestamp_us={value('frame_timestamp_us')} "
                      f"received_unix_s={frame.unix_s:.6f}", flush=True)
                print(f"RGB frame={frame.sequence} sdk_arrival_unix_ms={value('sdk_arrival_unix_ms')} "
                      f"{timing_text}", flush=True)
                last_print = time.monotonic()
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
            if args.seconds > 0 and time.monotonic() - started >= args.seconds:
                break
            # Detect clock changes instead of silently corrupting timestamp differences.
            if abs(time.time() - time.monotonic_ns() / 1e9 - unix_offset) > .005:
                raise RuntimeError("Host wall clock changed by >5 ms; restart this measurement")
            time.sleep(.005)
        with lock:
            if state["error"]:
                raise RuntimeError("QR/capture worker failed") from state["error"]
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        if worker:
            worker.join()
        camera.close()
        cv2.destroyAllWindows()
        if csv_file:
            csv_file.close()
        if latencies:
            print("Final QR-to-host p50/p95/min/max (ms):",
                  np.percentile(latencies, [50, 95, 0, 100]).round(2).tolist(), flush=True)


if __name__ == "__main__":
    main()
