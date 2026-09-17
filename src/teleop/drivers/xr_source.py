#!/usr/bin/env python3
"""XR pose source — ctypes binding onto libPXREARobotSDK.so, reads PICO
controller data via roboticsservice.

No pybind11 layer needed — the .so exports 4 C functions with no missing
deps, ctypes.CDLL loads it directly.

Startup order (reversed order fails to connect):
    1. Host:  /opt/apps/roboticsservice/runService.sh
    2. PICO:  open XenseVR-Toolkit, check Controller, enable Send

Self-test:
    python3 xr_source.py --dump            # print live pose + rate
    python3 xr_source.py --dump --raw      # also print the raw JSON

See docs/drivers.md for design rationale, empirical data, and known pitfalls.
"""
import collections
import ctypes
import json
import threading
import time

SDK_PATH = '/opt/apps/roboticsservice/SDK/x64/libPXREARobotSDK.so'

# PXREAClientCallbackType
SERVER_CONNECT = 1 << 2
SERVER_DISCONNECT = 1 << 3
DEVICE_FIND = 1 << 4
DEVICE_MISSING = 1 << 5
DEVICE_CONNECT = 1 << 9
DEVICE_STATE_JSON = 1 << 25
DEVICE_CUSTOM_MESSAGE = 1 << 26
FULL_MASK = 0xFFFFFFFF

_TYPE_NAMES = {
    SERVER_CONNECT: 'ServerConnect',
    SERVER_DISCONNECT: 'ServerDisconnect',
    DEVICE_FIND: 'DeviceFind',
    DEVICE_MISSING: 'DeviceMissing',
    DEVICE_CONNECT: 'DeviceConnect',
    DEVICE_STATE_JSON: 'DeviceStateJson',
    DEVICE_CUSTOM_MESSAGE: 'DeviceCustomMessage',
}


class PXREADevStateJson(ctypes.Structure):
    _fields_ = [('devID', ctypes.c_char * 32),
                ('stateJson', ctypes.c_char * 16352)]


_CALLBACK_T = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int,
                               ctypes.c_int, ctypes.c_void_p)
_CB_REF = None          # module-level strong ref, prevents GC-induced segfault


def parse_pose(s):
    """'x,y,z,qx,qy,qz,qw' -> (px,py,pz,qx,qy,qz,qw), meters / unit quaternion.

    Raw device axes, unconverted. See docs/drivers.md for the measured axis
    convention; frame conversion into the robot base happens in retarget.py.
    """
    if not s:
        return None
    v = [float(x) for x in s.split(',')]
    if len(v) != 7:
        raise ValueError('pose 字段应为 7 个数，实得 %d: %r' % (len(v), s))
    return tuple(v)


class XRFrame:
    """One snapshot of XR data."""

    __slots__ = ('dev_id', 'raw', 'host_ns', 'timestamp_ns', 'predict_time',
                 'head', 'left', 'right')

    def __init__(self, dev_id, raw, host_ns):
        self.dev_id = dev_id
        self.raw = raw
        self.host_ns = host_ns                      # host monotonic clock, for alignment
        self.timestamp_ns = raw.get('timeStampNs')  # headset clock
        self.predict_time = raw.get('predictTime')  # microseconds
        self.head = raw.get('Head')
        ctrl = raw.get('Controller') or {}
        self.left = ctrl.get('left')
        self.right = ctrl.get('right')

    def pose(self, hand):
        """hand: 'left' | 'right' | 'head' -> 7-tuple, or None if no data."""
        d = self.head if hand == 'head' else getattr(self, hand)
        if not d:
            return None
        return parse_pose(d.get('pose'))

    def button(self, hand, name, default=0.0):
        """name: trigger / grip / axisX / axisY / axisClick /
        primaryButton / secondaryButton / menuButton"""
        d = getattr(self, hand)
        if not d:
            return default
        return d.get(name, default)

    def __repr__(self):
        return '<XRFrame %s ts=%s L=%s R=%s>' % (
            self.dev_id, self.timestamp_ns,
            'y' if self.left else 'n', 'y' if self.right else 'n')


class XRSource:
    """Thread-safe XR data source.

        src = XRSource()
        src.start()
        f = src.latest(max_age_s=0.1)   # None means stale/no data
    """

    def __init__(self, dev_id=None, on_event=None):
        """dev_id: only accept this device serial; None = accept the first one seen.
        on_event: optional fn(type_name, status, payload) for UI/logging."""
        self._want_dev = dev_id
        self._on_event = on_event
        self._lock = threading.Lock()
        self._frame = None
        self._count = 0
        self._parse_errors = 0
        self._devices = set()
        self._started = False
        self._lib = None
        self._rec_queue = None          # non-None while recording
        self._rec_stop = None
        self._rec_thread = None
        self.rec_written = 0

    # ---------- lifecycle ----------

    def start(self):
        global _CB_REF
        if self._started:
            return
        self._lib = ctypes.CDLL(SDK_PATH)
        self._lib.PXREAInit.restype = ctypes.c_int
        self._lib.PXREADeinit.restype = ctypes.c_int
        self._lib.PXREADeviceControlJson.restype = ctypes.c_int
        self._lib.PXREADeviceControlJson.argtypes = [ctypes.c_char_p,
                                                     ctypes.c_char_p]
        _CB_REF = _CALLBACK_T(self._on_callback)
        rc = self._lib.PXREAInit(None, _CB_REF, FULL_MASK)
        if rc != 0:
            raise RuntimeError(
                'PXREAInit 失败 (rc=%d)：roboticsservice 起来了吗？'
                ' 跑 /opt/apps/roboticsservice/runService.sh' % rc)
        self._started = True

    def stop(self):
        self.stop_recording()
        if self._started and self._lib is not None:
            self._lib.PXREADeinit()
            self._started = False

    # ---------- recording (zero frame loss) ----------

    def start_recording(self, path):
        """Persist every frame as-is. Callback only enqueues; a separate
        writer thread does the disk I/O, so it never blocks the callback."""
        if self._rec_queue is not None:
            raise RuntimeError('已经在录制了')
        self.rec_written = 0
        q = collections.deque()
        self._rec_stop = threading.Event()

        def _writer():
            with open(path, 'w') as fh:
                while True:
                    try:
                        host_ns, dev, raw = q.popleft()
                    except IndexError:
                        if self._rec_stop.is_set():
                            break       # exit only once stopped and queue drained
                        time.sleep(0.002)
                        continue
                    fh.write(json.dumps({'host_ns': host_ns, 'dev': dev,
                                         'state': raw},
                                        ensure_ascii=False) + '\n')
                    self.rec_written += 1
                    if self.rec_written % 200 == 0:
                        fh.flush()

        self._rec_queue = q             # assign last, so writer never sees a stale queue
        self._rec_thread = threading.Thread(target=_writer, daemon=True)
        self._rec_thread.start()

    def stop_recording(self):
        """Stop recording and flush whatever is left in the queue. Returns frames written."""
        if self._rec_queue is None:
            return 0
        self._rec_queue = None          # detach first so the callback stops enqueuing
        self._rec_stop.set()
        if self._rec_thread:
            self._rec_thread.join(timeout=5.0)
        self._rec_thread = None
        return self.rec_written

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    # ---------- callback (SDK thread! keep minimal) ----------

    def _on_callback(self, context, cb_type, status, user_data):
        try:
            if cb_type == DEVICE_STATE_JSON:
                st = ctypes.cast(
                    user_data, ctypes.POINTER(PXREADevStateJson)).contents
                dev = st.devID.decode('utf-8', 'replace')
                if self._want_dev and dev != self._want_dev:
                    return
                raw = json.loads(st.stateJson.decode('utf-8', 'replace'))
                # unwrap the {"functionName":"Tracking","value":"<json>"} envelope;
                # see docs/drivers.md
                if (isinstance(raw, dict) and isinstance(raw.get('value'), str)
                        and raw.get('functionName') == 'Tracking'):
                    raw = json.loads(raw['value'])
                host_ns = time.monotonic_ns()
                frame = XRFrame(dev, raw, host_ns)
                with self._lock:
                    self._frame = frame
                    self._count += 1
                # recording: append-only, disk I/O happens on the writer thread
                q = self._rec_queue
                if q is not None:
                    q.append((host_ns, dev, raw))
                return

            name = _TYPE_NAMES.get(cb_type, 'Type%d' % cb_type)
            payload = ''
            if user_data:
                try:
                    payload = ctypes.cast(
                        user_data, ctypes.c_char_p).value.decode(
                            'utf-8', 'replace')
                except Exception:
                    payload = ''
            if cb_type == DEVICE_FIND:
                with self._lock:
                    self._devices.add(payload)
            elif cb_type == DEVICE_MISSING:
                with self._lock:
                    self._devices.discard(payload)
            if self._on_event:
                self._on_event(name, status, payload)
        except Exception:
            # must never let an exception propagate back into the C layer
            with self._lock:
                self._parse_errors += 1

    # ---------- reads ----------

    def latest(self, max_age_s=None):
        """Return the latest frame; None if older than max_age_s (stale)."""
        with self._lock:
            f = self._frame
        if f is None:
            return None
        if max_age_s is not None:
            if (time.monotonic_ns() - f.host_ns) > max_age_s * 1e9:
                return None
        return f

    def age_s(self):
        """Seconds since the latest frame; inf if no data."""
        with self._lock:
            f = self._frame
        if f is None:
            return float('inf')
        return (time.monotonic_ns() - f.host_ns) / 1e9

    def stats(self):
        with self._lock:
            return {'frames': self._count, 'parse_errors': self._parse_errors,
                    'devices': sorted(self._devices)}

    def wait_for_data(self, timeout_s=10.0):
        """Block until the first frame arrives. Returns True/False."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            if self.latest() is not None:
                return True
            time.sleep(0.02)
        return False

    def control_json(self, dev_id, params):
        """Send a JSON command to the device (PXREADeviceControlJson)."""
        if isinstance(params, dict):
            params = json.dumps(params)
        return self._lib.PXREADeviceControlJson(dev_id.encode(),
                                                params.encode())


def _main():
    import argparse
    ap = argparse.ArgumentParser(description='XR data source self-test')
    ap.add_argument('--dump', action='store_true', help='Print continuously.')
    ap.add_argument('--raw', action='store_true',
                    help='Also print the raw JSON.')
    ap.add_argument('--dev', default=None,
                    help='Only accept this device serial.')
    ap.add_argument('--hz', type=float, default=5.0, help='Print rate.')
    ap.add_argument('--timeout', type=float, default=30.0,
                    help='Seconds to wait for the first frame.')
    ap.add_argument('--record', metavar='FILE',
                    help='Save every frame as-is to a jsonl file (with '
                         'host timestamps, for offline replay).')
    ap.add_argument('--duration', type=float, default=None,
                    help='Auto-stop recording/printing after this many '
                         'seconds.')
    args = ap.parse_args()

    def on_event(name, status, payload):
        print('[event] %-16s status=%s %s' % (name, status, payload))

    src = XRSource(dev_id=args.dev, on_event=on_event)
    src.start()
    print('SDK 已加载，等待设备…（PICO 上要打开应用并按下 Send）')
    if not src.wait_for_data(args.timeout):
        print('❌ %.0fs 内没收到任何位姿。检查：'
              '\n   1) runService.sh 是否在跑（且先于头显应用启动）'
              '\n   2) PICO 与主机是否同网段'
              '\n   3) PICO 面板上 Tracking-Controller 与 Send 是否都打开'
              % args.timeout)
        src.stop()
        return 1

    print('✅ 收到数据：%s' % src.stats())
    if not args.dump and not args.record:
        src.stop()
        return 0

    if args.record:
        src.start_recording(args.record)
        print('⏺  录制中 → %s' % args.record)

    t_start = time.monotonic()
    last_n, last_t = 0, time.monotonic()
    period = 1.0 / args.hz
    try:
        while True:
            if args.duration and (time.monotonic() - t_start) >= args.duration:
                break
            time.sleep(period)
            if not args.dump:
                continue
            f = src.latest(max_age_s=0.2)
            n = src.stats()['frames']
            now = time.monotonic()
            hz = (n - last_n) / (now - last_t)
            last_n, last_t = n, now
            if f is None:
                print('⚠️  stale  age=%.3fs  （XR 链路断了？）' % src.age_s())
                continue
            l, r = f.pose('left'), f.pose('right')
            fmt = lambda p: ('—' if p is None else
                             '%+.3f %+.3f %+.3f | %+.3f %+.3f %+.3f %+.3f' % p)
            print('%6.1f Hz  age=%.3f' % (hz, src.age_s()))
            print('   L %s  trig=%.2f grip=%.2f' %
                  (fmt(l), f.button('left', 'trigger'),
                   f.button('left', 'grip')))
            print('   R %s  trig=%.2f grip=%.2f  A=%s B=%s' %
                  (fmt(r), f.button('right', 'trigger'),
                   f.button('right', 'grip'),
                   f.button('right', 'primaryButton'),
                   f.button('right', 'secondaryButton')))
            if args.raw:
                print('   raw: %s' % json.dumps(f.raw, ensure_ascii=False))
    except KeyboardInterrupt:
        pass
    finally:
        if args.record:
            n = src.stop_recording()
            print('⏹  已录 %d 帧 → %s（SDK 共收到 %d 帧）'
                  % (n, args.record, src.stats()['frames']))
        src.stop()
        print('退出：%s' % src.stats())
    return 0


if __name__ == '__main__':
    raise SystemExit(_main())
