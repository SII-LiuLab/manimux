#!/usr/bin/env python3
# ruff: noqa: E501
"""Small browser-based live preview for two Orbbec UVC color cameras."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2

LOG = logging.getLogger("orbbec-preview")


PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Orbbec 双摄 · 实时预览</title>
  <style>
    :root { color-scheme: dark; --bg:#090d12; --panel:#111821; --muted:#8c9aaa;
      --line:#26313d; --ok:#45d483; --bad:#ff6473; --accent:#59a8ff; }
    * { box-sizing: border-box; }
    body { margin:0; min-height:100vh; background:radial-gradient(circle at 50% -20%,#1a2c3f,#090d12 58%);
      color:#f4f7fb; font:14px/1.45 Inter,ui-sans-serif,system-ui,sans-serif; }
    main { width:min(1720px,calc(100% - 32px)); margin:24px auto; }
    header { display:flex; align-items:center; justify-content:space-between; gap:16px; margin-bottom:14px; }
    h1 { margin:0; font-size:clamp(20px,3vw,30px); font-weight:650; letter-spacing:-.02em; }
    .sub { margin-top:4px; color:var(--muted); }
    .badge { display:flex; align-items:center; gap:8px; padding:8px 12px; border:1px solid var(--line);
      border-radius:999px; background:rgba(17,24,33,.86); white-space:nowrap; }
    .dot { width:9px; height:9px; border-radius:50%; background:var(--bad); box-shadow:0 0 12px currentColor; }
    .dot.ok { background:var(--ok); }
    .grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }
    .card { padding:12px; border:1px solid var(--line); border-radius:18px; background:rgba(17,24,33,.75);
      box-shadow:0 22px 70px rgba(0,0,0,.25); }
    .card-head { display:flex; align-items:center; justify-content:space-between; gap:12px; padding:2px 2px 10px; }
    .card-title { font-size:17px; font-weight:620; }
    .card-serial { color:var(--muted); font-size:12px; margin-top:2px; }
    .viewer { position:relative; overflow:hidden; border:1px solid var(--line); border-radius:12px;
      background:#020405; aspect-ratio:16/9; }
    .viewer img { display:block; width:100%; height:100%; object-fit:contain; }
    .overlay { position:absolute; inset:0; display:grid; place-items:center; pointer-events:none; }
    .message { padding:10px 14px; border-radius:10px; background:rgba(4,7,10,.78); color:#dbe5ef; }
    .toolbar { display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:12px;
      padding:14px 2px 0; color:var(--muted); }
    .stats { display:flex; flex-wrap:wrap; gap:8px 18px; }
    .stats b { color:#eef4fa; font-weight:560; }
    .actions { display:flex; gap:8px; }
    button,a.button { appearance:none; border:1px solid var(--line); border-radius:9px; padding:8px 12px;
      color:#edf4fb; background:var(--panel); cursor:pointer; text-decoration:none; font:inherit; }
    button:hover,a.button:hover { border-color:var(--accent); }
    @media (max-width:1050px) { .grid{grid-template-columns:1fr} }
    @media (max-width:650px) { main{width:min(100% - 18px,1720px);margin:12px auto} header{align-items:flex-start} }
  </style>
</head>
<body><main>
  <header>
    <div><h1>Orbbec 双摄实时预览</h1><div class="sub">Gemini 305 + Gemini 335 · RGB 1280 × 720</div></div>
    <div class="badge"><span id="dot-all" class="dot"></span><span id="state-all">正在连接</span></div>
  </header>
  <div class="grid">
    <article class="card">
      <div class="card-head">
        <div><div class="card-title">Gemini 305</div><div class="card-serial">CV278640000Z</div></div>
        <div class="badge"><span id="dot-0" class="dot"></span><span id="state-0">正在连接</span></div>
      </div>
      <section id="viewer-0" class="viewer">
        <img id="feed-0" src="/stream/0.mjpg" alt="Gemini 305 live stream">
        <div class="overlay"><div id="message-0" class="message">正在等待第一帧…</div></div>
      </section>
      <div class="toolbar">
        <div class="stats"><span>分辨率 <b id="resolution-0">—</b></span><span>帧率 <b id="fps-0">—</b></span><span>设备 <b id="device-0">—</b></span></div>
        <div class="actions"><a class="button" href="/snapshot/0.jpg" target="_blank">打开单帧</a><button data-fullscreen="viewer-0" type="button">全屏</button></div>
      </div>
    </article>
    <article class="card">
      <div class="card-head">
        <div><div class="card-title">Gemini 335</div><div class="card-serial">CP0N763000LK</div></div>
        <div class="badge"><span id="dot-1" class="dot"></span><span id="state-1">正在连接</span></div>
      </div>
      <section id="viewer-1" class="viewer">
        <img id="feed-1" src="/stream/1.mjpg" alt="Gemini 335 live stream">
        <div class="overlay"><div id="message-1" class="message">正在等待第一帧…</div></div>
      </section>
      <div class="toolbar">
        <div class="stats"><span>分辨率 <b id="resolution-1">—</b></span><span>帧率 <b id="fps-1">—</b></span><span>设备 <b id="device-1">—</b></span></div>
        <div class="actions"><a class="button" href="/snapshot/1.jpg" target="_blank">打开单帧</a><button data-fullscreen="viewer-1" type="button">全屏</button></div>
      </div>
    </article>
  </div>
</main>
<script>
  const $ = id => document.getElementById(id);
  document.querySelectorAll('[data-fullscreen]').forEach(button => button.addEventListener('click', () => $(button.dataset.fullscreen).requestFullscreen?.()));
  [0, 1].forEach(i => {
    $(`feed-${i}`).addEventListener('load', () => $(`message-${i}`).hidden = true);
    $(`feed-${i}`).addEventListener('error', () => { $(`message-${i}`).hidden = false; $(`message-${i}`).textContent = '视频流中断，正在等待恢复…'; });
  });
  async function update() {
    try {
      const s = await fetch('/api/status', {cache:'no-store'}).then(r => r.json());
      $('dot-all').classList.toggle('ok', s.all_online);
      $('state-all').textContent = s.all_online ? '两路实时在线' : `${s.online_count} / ${s.cameras.length} 在线`;
      s.cameras.forEach((camera, i) => {
        $(`dot-${i}`).classList.toggle('ok', camera.online);
        $(`state-${i}`).textContent = camera.online ? '实时在线' : '等待摄像头';
        $(`resolution-${i}`).textContent = camera.width && camera.height ? `${camera.width} × ${camera.height}` : '—';
        $(`fps-${i}`).textContent = camera.measured_fps ? `${camera.measured_fps.toFixed(1)} FPS` : '—';
        $(`device-${i}`).textContent = camera.device;
        if (!camera.online) { $(`message-${i}`).hidden = false; $(`message-${i}`).textContent = camera.error || '正在等待第一帧…'; }
        else { $(`message-${i}`).hidden = true; }
      });
    } catch (_) {
      $('dot-all').classList.remove('ok'); $('state-all').textContent = '服务不可达';
    }
  }
  update(); setInterval(update, 1000);
</script></body></html>"""


class CameraStream:
    def __init__(
        self,
        name: str,
        serial: str,
        device: str,
        width: int,
        height: int,
        fps: int,
        quality: int,
    ) -> None:
        self.name = name
        self.serial = serial
        self.device = device
        self.width = width
        self.height = height
        self.target_fps = fps
        self.quality = quality
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._capture_loop, name=f"orbbec-{serial}-capture", daemon=True
        )
        self.jpeg: bytes | None = None
        self.sequence = 0
        self.started_at = time.monotonic()
        self.last_frame_at = 0.0
        self.measured_fps = 0.0
        self.error: str | None = None
        self.actual_width = 0
        self.actual_height = 0

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=3.0)

    def _open(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.target_fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        return capture

    def _capture_loop(self) -> None:
        capture: cv2.VideoCapture | None = None
        sample_start = time.monotonic()
        sample_frames = 0
        while not self.stop_event.is_set():
            if capture is None or not capture.isOpened():
                if capture is not None:
                    capture.release()
                capture = self._open()
                if not capture.isOpened():
                    self.error = f"无法打开 {self.device}"
                    self.stop_event.wait(1.0)
                    continue
                self.error = None

            ok, frame = capture.read()
            if not ok or frame is None:
                self.error = "摄像头暂时没有返回画面，正在重连"
                capture.release()
                capture = None
                self.stop_event.wait(0.5)
                continue

            encoded, buffer = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
            )
            if not encoded:
                self.error = "JPEG 编码失败"
                continue

            now = time.monotonic()
            sample_frames += 1
            elapsed = now - sample_start
            if elapsed >= 1.0:
                self.measured_fps = sample_frames / elapsed
                sample_start, sample_frames = now, 0

            with self.condition:
                self.jpeg = buffer.tobytes()
                self.sequence += 1
                self.actual_height, self.actual_width = frame.shape[:2]
                self.last_frame_at = now
                self.error = None
                self.condition.notify_all()

        if capture is not None:
            capture.release()

    def wait_for_frame(self, after: int, timeout: float = 2.0) -> tuple[int, bytes | None]:
        with self.condition:
            self.condition.wait_for(
                lambda: self.sequence > after or self.stop_event.is_set(), timeout=timeout
            )
            return self.sequence, self.jpeg

    def status(self) -> dict[str, Any]:
        age = time.monotonic() - self.last_frame_at if self.last_frame_at else None
        return {
            "name": self.name,
            "serial": self.serial,
            "device": self.device,
            "online": age is not None and age < 2.0,
            "width": self.actual_width,
            "height": self.actual_height,
            "target_fps": self.target_fps,
            "measured_fps": round(self.measured_fps, 2),
            "frames": self.sequence,
            "frame_age_ms": round(age * 1000, 1) if age is not None else None,
            "error": self.error,
        }


def make_handler(cameras: list[CameraStream]) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "OrbbecPreview/1.0"

        def log_message(self, format: str, *args: object) -> None:
            LOG.info("%s - %s", self.address_string(), format % args)

        def _send(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/":
                self._send(PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/api/status":
                statuses = [camera.status() for camera in cameras]
                online_count = sum(status["online"] for status in statuses)
                body = {
                    "all_online": online_count == len(cameras),
                    "online_count": online_count,
                    "cameras": statuses,
                }
                self._send(json.dumps(body).encode(), "application/json")
            elif path == "/healthz":
                statuses = [camera.status() for camera in cameras]
                online = all(status["online"] for status in statuses)
                status = HTTPStatus.OK if online else HTTPStatus.SERVICE_UNAVAILABLE
                self._send(json.dumps(statuses).encode(), "application/json", status)
            elif path.startswith("/snapshot/") and path.endswith(".jpg"):
                camera = self._camera_from_path(path, "/snapshot/", ".jpg")
                if camera is None:
                    return
                _, frame = camera.wait_for_frame(-1)
                if frame is None:
                    self._send(b"camera frame unavailable\n", "text/plain", HTTPStatus.SERVICE_UNAVAILABLE)
                else:
                    self._send(frame, "image/jpeg")
            elif path.startswith("/stream/") and path.endswith(".mjpg"):
                camera = self._camera_from_path(path, "/stream/", ".mjpg")
                if camera is not None:
                    self._stream_mjpeg(camera)
            else:
                self._send(b"not found\n", "text/plain", HTTPStatus.NOT_FOUND)

        def _camera_from_path(
            self, path: str, prefix: str, suffix: str
        ) -> CameraStream | None:
            try:
                index = int(path.removeprefix(prefix).removesuffix(suffix))
                if index < 0:
                    raise IndexError
                return cameras[index]
            except (ValueError, IndexError):
                self._send(b"camera not found\n", "text/plain", HTTPStatus.NOT_FOUND)
                return None

        def _stream_mjpeg(self, camera: CameraStream) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.end_headers()
            sequence = -1
            try:
                while not camera.stop_event.is_set():
                    sequence, frame = camera.wait_for_frame(sequence)
                    if frame is None:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                return

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemini-305-device", default="/dev/video6", help="Gemini 305 RGB node")
    parser.add_argument("--gemini-335-device", default="/dev/video34", help="Gemini 335 RGB node")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP bind address")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    camera_specs = [
        ("Gemini 305", "CV278640000Z", args.gemini_305_device),
        ("Gemini 335", "CP0N763000LK", args.gemini_335_device),
    ]
    missing = [device for _, _, device in camera_specs if not Path(device).exists()]
    if missing:
        raise SystemExit(f"Camera device does not exist: {', '.join(missing)}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cameras = [
        CameraStream(name, serial, device, args.width, args.height, args.fps, args.jpeg_quality)
        for name, serial, device in camera_specs
    ]
    server = ThreadingHTTPServer((args.host, args.port), make_handler(cameras))
    server.daemon_threads = True

    def shutdown(_signum: int, _frame: object) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    for camera in cameras:
        camera.start()
    LOG.info("Orbbec dual preview: http://%s:%d", args.host, args.port)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        for camera in cameras:
            camera.stop()
        server.server_close()


if __name__ == "__main__":
    main()
