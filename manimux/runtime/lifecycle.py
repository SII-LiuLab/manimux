"""Report blocking shutdown phases without mistaking them for idle readiness."""

import logging
import threading
import time
import uuid

from manimux.embodiments.robot.release import ReleaseDecision
from manimux.robogui.communication import ControlClient, RoboGUIPublisher, RuntimeEvent

log = logging.getLogger(__name__)


class OperationStatus:
    """A liveness/progress reporter for blocking work; never access hardware."""

    def __init__(self, *, enabled, robot, policy, event, identity,
                 publisher_factory=RoboGUIPublisher, interval_s=0.5):
        self.enabled, self.robot, self.policy = enabled, robot, policy
        self.event, self.identity = event, dict(identity)
        self._factory, self._interval = publisher_factory, interval_s
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._phase, self._since = "Preparing runtime", time.monotonic()

    def update(self, phase):
        with self._lock:
            self._phase, self._since = phase, time.monotonic()
        if self.enabled and self._thread is None:
            try:
                reporter = threading.Thread(target=self._run, daemon=True,
                                            name="robogui-operation-status")
                reporter.start()
                self._thread = reporter
            except Exception:
                log.exception("Cannot start operation reporter; runtime ownership is unchanged")

    def _run(self):
        try:
            publisher = self._factory()
            try:
                while not self._stop.is_set():
                    with self._lock:
                        metadata = {**self.identity, "phase": self._phase,
                                    "elapsed_s": time.monotonic() - self._since}
                    publisher.publish(RuntimeEvent(self.event, robot=self.robot,
                                                   policy=self.policy, metadata=metadata))
                    self._stop.wait(self._interval)
            finally:
                publisher.close()
        except Exception:
            log.exception("Operation progress reporting failed")

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


class ShutdownStatus:
    """Own a separate publisher thread only during cleanup; never touch hardware.

    This heartbeat establishes reporter liveness, not motor health or progress.
    Each ZMQ socket is created, used and closed on its owning thread.
    """

    def __init__(self, *, enabled, robot, policy, run_dir, episode_id,
                 publisher_factory=RoboGUIPublisher, interval_s=0.5,
                 home=None, control_factory=ControlClient):
        self.enabled = enabled
        self.robot, self.policy = robot, policy
        self.identity = {"run_dir": run_dir, "episode_id": episode_id}
        self._factory, self._interval = publisher_factory, interval_s
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._changed = threading.Event()
        self._thread = None
        self._phase, self._detail, self._since = "", "", 0.0
        self._home = home
        self._control_factory = control_factory
        self._recovery_token = ""

    def update(self, phase, detail=""):
        with self._lock:
            self._phase, self._detail, self._since = phase, detail, time.monotonic()
            if phase != "release_blocked":
                self._recovery_token = ""
        if not self.enabled:
            return
        if self._thread is None:
            try:
                reporter = threading.Thread(target=self._run, name="shutdown-status", daemon=True)
                reporter.start()
                self._thread = reporter
            except Exception:
                # Reporting is optional; failure to start a GUI heartbeat must
                # never prevent the hardware owner from stopping or homing.
                log.exception("Cannot start shutdown reporter; hardware cleanup continues")
                return
        self._changed.set()

    def _run(self):
        try:
            publisher = self._factory()
            try:
                while not self._stop.is_set():
                    self._changed.clear()
                    with self._lock:
                        metadata = {**self.identity, "phase": self._phase,
                                    "detail": self._detail,
                                    "recovery_token": self._recovery_token,
                                    "can_home": self._home is not None,
                                    "elapsed_s": time.monotonic() - self._since}
                    publisher.publish(RuntimeEvent(
                        "runtime_shutdown", robot=self.robot, policy=self.policy,
                        metadata=metadata,
                    ))
                    self._changed.wait(self._interval)
            finally:
                publisher.close()
        except Exception:
            log.exception("Shutdown status publisher failed; hardware cleanup continues")

    def wait_for_release(self, reason):
        """The hardware owner handles GUI requests; the heartbeat never moves a robot."""
        if not self.enabled:
            return None
        client = self._control_factory()
        try:
            while True:
                with self._lock:
                    token = self._recovery_token = uuid.uuid4().hex
                self.update("release_blocked", reason)
                try:
                    while True:
                        state = client.poll_state()
                        request = (state or {}).get("shutdown_recovery")
                        if not isinstance(request, dict):
                            time.sleep(0.05)
                            continue
                        if (request.get("token") != token
                                or any(request.get(k) != v for k, v in self.identity.items())):
                            time.sleep(0.05)
                            continue
                        action = request.get("action")
                        if action == "release" and request.get("supported") is True:
                            self.update("closing", "Operator confirmed support for ALL arms and payloads.")
                            return ReleaseDecision.SUPPORTED_RELEASE
                        if action == "home" and request.get("path_clear") is True and self._home is not None:
                            self.update("homing", "Operator requested Home retry; checking control health.")
                            self._home()
                            self.update("closing", "Home completed; rechecking every arm before release.")
                            return ReleaseDecision.RECHECK
                        time.sleep(0.05)
                except (Exception, KeyboardInterrupt) as exc:
                    # A failed Home, lost UI or interrupt never authorizes closing.
                    reason = f"Recovery did not complete: {type(exc).__name__}: {exc}"
                    log.error(reason)
                    time.sleep(0.1)
        finally:
            client.close()

    def close(self):
        self._stop.set()
        self._changed.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
