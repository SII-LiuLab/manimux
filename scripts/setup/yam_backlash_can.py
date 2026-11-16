"""Read YAM CAN feedback passively or with explicit feedback-only queries.

The standard YAM uses motor IDs 1–3 (DM4340), 4–6 (DM4310), and feedback
IDs 0x11–0x16. Position decoding follows the installed i2rt MIT codec:
16 bits over [-12.5, 12.5] radians. These are raw motor feedback positions;
SDK software offsets and wrap tracking are deliberately not applied.

PassiveCanSource only subscribes to an already configured Linux SocketCAN
interface and sends nothing. ReadQueryCanSource can explicitly send one round
of the manufacturer's 0xCC status query when refresh() is called; it never
polls in the background and caps requests at 10 per second per arm joint.
Neither enables motors, starts a controller, or changes interface settings.
Missing feedback must remain missing, rather than becoming zero.

Each callback's time is the kernel receive timestamp (SO_TIMESTAMPNS,
CLOCK_REALTIME), translated to time.monotonic() using the host clock offset
at receipt. Buffered frames therefore keep their original age. Frames with
missing or future kernel timestamps are discarded. A host wall-clock change
can affect this translation; this is not a device sampling timestamp.
"""

import errno
import socket
import struct
import threading
import time
from collections.abc import Callable

CAN_EFF_FLAG = 0x80000000
CAN_RTR_FLAG = 0x40000000
CAN_ERR_FLAG = 0x20000000
CAN_SFF_MASK = 0x7FF
CAN_ERR_CRTL = 0x04
CAN_ERR_BUSOFF = 0x40
CAN_RAW_ERR_FILTER = getattr(socket, "CAN_RAW_ERR_FILTER", 2)
CAN_FRAME = struct.Struct("=IB3x8s")
# Linux's native timespec timestamp option; Python does not expose it on all builds.
SO_TIMESTAMPNS = getattr(socket, "SO_TIMESTAMPNS", 35)
TIMESPEC = struct.Struct("@ll")


def decode_feedback(can_id: int, data: bytes) -> dict | None:
    """Decode a standard eight-byte arm feedback frame, rejecting other traffic."""
    if not isinstance(can_id, int) or not 0x11 <= can_id <= 0x16 or len(data) != 8:
        return None
    joint = can_id - 0x10
    if data[0] & 0x0F != joint:
        return None
    raw_position = (data[1] << 8) | data[2]
    return {
        "joint": joint,
        "position_rad": raw_position * 25.0 / 65535.0 - 12.5,
        "status": data[0] >> 4,
        "raw_position": raw_position,
        "motor_type": "DM4340" if joint <= 3 else "DM4310",
    }


def _received_monotonic(ancillary: list) -> float:
    """Translate an actual kernel receive time; never stamp queued data as new."""
    for level, kind, value in ancillary:
        if level == socket.SOL_SOCKET and kind == SO_TIMESTAMPNS and len(value) >= TIMESPEC.size:
            seconds, nanoseconds = TIMESPEC.unpack_from(value)
            if seconds < 0 or not 0 <= nanoseconds < 1_000_000_000:
                raise ValueError("Invalid kernel CAN receive timestamp")
            # Sample monotonic first: scheduling delay can only age a frame,
            # never make a queued frame look fresher than its kernel timestamp.
            monotonic = time.monotonic()
            age_ns = time.time_ns() - (seconds * 1_000_000_000 + nanoseconds)
            if age_ns < 0:
                raise ValueError("Kernel CAN timestamp is in the future; check the host clock")
            return monotonic - age_ns / 1_000_000_000
    raise ValueError("CAN frame has no kernel receive timestamp")


class PassiveCanSource:
    """An inert-until-started receive-only subscription; callbacks run in threads.

    on_error receives the aggregate of outstanding arm errors, or None when all
    recoverable errors have cleared. Valid data clears only its own arm's
    timestamp, controller or interface-down warning. ENETDOWN retains the
    socket and retries receiving with a bounded backoff; other receive failures
    remain errors until shutdown.
    """

    timestamp_basis = "kernel CLOCK_REALTIME receive time translated to host monotonic time"

    def __init__(
        self,
        channels: dict[str, str],
        on_sample: Callable[[str, dict, float], None],
        on_error: Callable[[str | None], None],
    ):
        if not channels or any(not arm or not channel for arm, channel in channels.items()):
            raise ValueError("At least one named arm and CAN channel is required")
        if len(set(channels.values())) != len(channels):
            raise ValueError("Each arm must use a different CAN channel")
        self.channels = dict(channels)
        self.on_sample = on_sample
        self.on_error = on_error
        self._stop = threading.Event()
        self._sockets = []
        self._threads = []
        self._started = False
        self._error_lock = threading.Lock()
        self._errors: dict[str, tuple[str, bool]] = {}
        self._reported_error = None

    def start(self) -> None:
        """Open all receive sockets, or close partial setup and raise on failure."""
        if self._started or self._stop.is_set():
            raise RuntimeError("CAN source has already been started or closed")
        # CAN_ERR_FLAG in this mask selects SocketCAN's error receive list,
        # rather than excluding error frames. Subscribe to errors separately.
        mask = CAN_SFF_MASK | CAN_EFF_FLAG | CAN_RTR_FLAG
        filters = b"".join(struct.pack("=II", ident, mask) for ident in range(0x11, 0x17))
        opened = []
        try:
            for arm, channel in self.channels.items():
                bus = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
                opened.append((arm, channel, bus))
                bus.setsockopt(socket.SOL_SOCKET, SO_TIMESTAMPNS, 1)
                bus.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, filters)
                bus.setsockopt(
                    socket.SOL_CAN_RAW,
                    CAN_RAW_ERR_FILTER,
                    struct.pack("=I", CAN_ERR_CRTL | CAN_ERR_BUSOFF),
                )
                bus.settimeout(0.2)
                bus.bind((channel,))
        except Exception:
            for _, _, bus in opened:
                bus.close()
            raise
        self._sockets = opened
        self._started = True
        try:
            for arm, channel, bus in opened:
                worker = threading.Thread(
                    target=self._receive,
                    args=(arm, channel, bus),
                    name=f"backlash-receive-{arm}",
                    daemon=True,
                )
                self._threads.append(worker)
                worker.start()
        except Exception:
            self.close()
            raise

    def _update_error(
        self, arm: str, message: str | None, *, fatal: bool = False, replace: bool = False
    ) -> None:
        # Hold the lock through the callback so two receiver threads cannot
        # deliver aggregate updates out of order to the UI's shared state.
        with self._error_lock:
            previous = self._errors.get(arm)
            if message is None:
                if previous is None or previous[1]:
                    return
                del self._errors[arm]
            else:
                # Report one timestamp warning until recovery; a fatal socket
                # failure supersedes it and cannot be cleared by another arm.
                if previous is not None and (previous[1] or (not fatal and not replace)):
                    return
                self._errors[arm] = (message, fatal)
            aggregate = (
                "; ".join(self._errors[name][0] for name in self.channels if name in self._errors)
                or None
            )
            if aggregate != self._reported_error:
                self._reported_error = aggregate
                self.on_error(aggregate)

    def _receive(self, arm: str, channel: str, bus) -> None:
        while not self._stop.is_set():
            try:
                frame, ancillary, flags, _address = bus.recvmsg(
                    CAN_FRAME.size, socket.CMSG_SPACE(TIMESPEC.size)
                )
            except TimeoutError:
                continue
            except OSError as exc:
                if exc.errno == errno.ENETDOWN and not self._stop.is_set():
                    # NETDEV_DOWN leaves SocketCAN's binding and filters intact.
                    # Keep listening on the same socket when the interface returns;
                    # this retries reception only and never sends a query.
                    self._update_error(
                        arm,
                        f"{arm} ({channel}): CAN interface is down; waiting for valid feedback",
                        replace=True,
                    )
                    if self._stop.wait(0.2):
                        return
                    continue
                if not self._stop.is_set():
                    self._update_error(
                        arm, f"{arm} ({channel}): CAN receive failed: {exc}", fatal=True
                    )
                return
            if self._stop.is_set():
                return
            if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC) or len(frame) != CAN_FRAME.size:
                continue
            can_id, length, data = CAN_FRAME.unpack(frame)
            if length != 8:
                continue
            if can_id & CAN_ERR_FLAG:
                controller_states = []
                if can_id & CAN_ERR_BUSOFF:
                    controller_states.append("BUS-OFF")
                if can_id & CAN_ERR_CRTL:
                    if data[1] & 0x20:
                        controller_states.append("TX ERROR-PASSIVE")
                    if data[1] & 0x10:
                        controller_states.append("RX ERROR-PASSIVE")
                if controller_states:
                    self._update_error(
                        arm,
                        f"{arm} ({channel}): CAN controller {' / '.join(controller_states)} "
                        f"(TX errors {data[6]}, RX errors {data[7]}); waiting for valid feedback",
                        replace=True,
                    )
                continue
            decoded = decode_feedback(can_id, data)
            if decoded is None:
                continue
            try:
                received = _received_monotonic(ancillary)
            except ValueError as exc:
                self._update_error(arm, f"{arm} ({channel}): {exc}; frame discarded")
                continue
            self._update_error(arm, None)
            self.on_sample(arm, decoded, received)

    def close(self) -> None:
        """Close this source's sockets only; no motor shutdown or other commands."""
        self._stop.set()
        for _, _, bus in self._sockets:
            bus.close()
        self._sockets.clear()
        for worker in self._threads:
            if worker is not threading.current_thread() and worker.ident is not None:
                worker.join(timeout=0.5)
        self._threads.clear()


class ReadQueryCanSource(PassiveCanSource):
    """Explicitly request ordinary feedback, without enabling or controlling motors.

    DAMIAO's SDK refresh_motor_status sends standard CAN ID 0x7FF with
    [ID low, ID high, 0xCC, 0, 0, 0, 0, 0]. Source:
    https://github.com/dmBots/motor-sdk/blob/main/Python%E4%BE%8B%E7%A8%8B/u2can/DM_CAN.py

    Only arm IDs 1–6 can be queried. There are no parameter writes, mode
    switches, enables, disables, error resets, zeroing or control frames.
    Unsupported firmware remains without data; there is no control fallback.
    start() only starts passive listeners. Each refresh() explicitly requests
    one round, serialized and capped at 10 Hz. Replies retain normal kernel
    timestamps; requesting data does not make missing replies into samples.
    """

    def __init__(self, channels, on_sample, on_error):
        super().__init__(channels, on_sample, on_error)
        self._refresh_lock = threading.Lock()
        self._last_refresh = float("-inf")

    def refresh(self) -> bool:
        """Request all arm joints once; return False if throttled within 100 ms.

        True means only that a query round was sent, not that feedback arrived.
        Startup, shutdown and transport failures raise without any fallback.
        """
        with self._refresh_lock:
            if not self._started or self._stop.is_set():
                raise RuntimeError("CAN source is not running")
            if time.monotonic() - self._last_refresh < 0.1:
                return False
            with self._error_lock:
                if any(fatal for _message, fatal in self._errors.values()):
                    raise RuntimeError("CAN source has a failed receiver or query transport")
            for arm, channel, bus in list(self._sockets):
                try:
                    for joint in range(1, 7):
                        if self._stop.is_set():
                            raise RuntimeError("CAN source closed during refresh")
                        frame = CAN_FRAME.pack(0x7FF, 8, bytes((joint, 0, 0xCC, 0, 0, 0, 0, 0)))
                        if bus.send(frame) != CAN_FRAME.size:
                            raise OSError("Incomplete CAN status query send")
                except OSError as exc:
                    if not self._stop.is_set():
                        if exc.errno == errno.ENETDOWN:
                            # A manual refresh during the outage must not prevent
                            # another manual refresh after the interface returns.
                            # Throttle partial rounds too; never retry this send.
                            self._last_refresh = time.monotonic()
                            self._update_error(
                                arm,
                                f"{arm} ({channel}): CAN interface is down; "
                                "waiting for valid feedback",
                                replace=True,
                            )
                        else:
                            self._update_error(
                                arm,
                                f"{arm} ({channel}): CAN status query failed: {exc}",
                                fatal=True,
                            )
                    raise RuntimeError(f"CAN status query failed on {channel}") from exc
            # Pace from round completion, preserving the per-joint upper rate
            # bound even if an earlier send was delayed.
            self._last_refresh = time.monotonic()
            return True
