"""Operator gate for controllers whose shutdown would remove physical support."""

import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum, auto

log = logging.getLogger(__name__)
_observer = ContextVar("supported_release_observer", default=None)
_recovery = ContextVar("supported_release_recovery", default=None)


class ReleaseDecision(Enum):
    """Recovery outcomes are distinct from a boolean permission to close."""

    RECHECK = auto()
    SUPPORTED_RELEASE = auto()


@contextmanager
def observe_release(callback, *, recover=None):
    """Report a gate to its owner without coupling hardware to a GUI/transport."""
    token = _observer.set(callback)
    recovery_token = _recovery.set(recover)
    try:
        yield
    finally:
        _observer.reset(token)
        _recovery.reset(recovery_token)


def _report(phase, reason):
    callback = _observer.get()
    if callback is not None:
        try:
            callback(phase, reason)
        except Exception:
            log.exception("Could not report actuator release status; gate remains enforced")


def require_supported_release(reason: str) -> ReleaseDecision:
    """Keep the owning process alive until explicit physical-support confirmation.

    SUPPORTED_RELEASE requires explicit physical-support confirmation;
    RECHECK means recovery ran and callers must revalidate before closing.
    Missing GUI and interrupts are not permission to remove actuator support. This is
    deliberately blocking: no new runtime may acquire these controllers meanwhile.
    It cannot protect against SIGKILL, power loss or a failed motor/CAN controller.
    """
    log.critical("ACTUATOR RELEASE BLOCKED: %s", reason)
    _report("release_blocked", reason)
    while True:
        try:
            recover = _recovery.get()
            if recover is not None:
                decision = recover(reason)
                if isinstance(decision, ReleaseDecision):
                    return decision
            log.critical("Control retained. Use RoboGUI hardware recovery; terminal input is disabled.")
            time.sleep(5.0)
        except KeyboardInterrupt:
            log.critical("Interrupt does not authorize actuator release; support arms first.")
        except Exception:
            log.exception("RoboGUI recovery failed; control remains owned and release is blocked")
            try:
                time.sleep(5.0)
            except KeyboardInterrupt:
                log.critical("Interrupt does not authorize actuator release.")
