"""Serializable operator diagnostics; classification never changes recovery policy."""

from contextlib import contextmanager


@contextmanager
def failure_context(source: str, stage: str, target: str = ""):
    """Attach origin at the owning call without changing the exception type."""
    try:
        yield
    except Exception as error:
        if not getattr(error, "manimux_failure", None):
            error.manimux_failure = {"source": source, "stage": stage, "target": target}
        raise


def describe_failure(error, *, context=None):
    """Use explicit origin first; recognize legacy service messages conservatively."""
    detail = str(error)
    origin = dict(context or {})
    if isinstance(error, BaseException):
        seen = set()
        pending = [error]
        messages = []
        while pending:
            item = pending.pop(0)
            if id(item) in seen:
                continue
            seen.add(id(item))
            messages.append(f"{type(item).__name__}: {item}")
            if not origin and getattr(item, "manimux_failure", None):
                origin = dict(item.manimux_failure)
            pending.extend(getattr(item, "exceptions", ()))
            cause = item.__cause__ or (None if item.__suppress_context__ else item.__context__)
            if cause is not None:
                pending.append(cause)
        detail = "\nCaused by: ".join(messages)
    lower = detail.lower()
    source = origin.get("source", "")
    if not source:
        if any(term in lower for term in ("camera", "sensor reader", "realsense")):
            source = "camera"
        elif any(term in lower for term in ("policy worker", "policy server", "model service", "policy backend identity mismatch")):
            source = "model"
        elif any(term in lower for term in ("socketcan", "can_left", "can_right", "motor", "actuator")):
            source = "robot"
        elif any(term in lower for term in ("inverse kinematics", "quaternion", "action decoder", "step ik", "tcp trajectory", "normalized gripper", "candidate decoding")):
            source = "action"
        else:
            source = "runtime"
    title, reason, action = {
        "model": ("Model service failed", "The model request did not complete.",
                  "Check the model service terminal and the configured endpoint."),
        "camera": ("Camera / sensor failed", "Fresh sensor data could not be obtained.",
                   "Check the named camera, camera service, USB connection and power."),
        "robot": ("Robot / CAN failed", "Robot communication or control failed.",
                  "Check the named interface and controller feedback before preparing again."),
        "action": ("Action decoding / IK failed", "Model actions could not be converted to executable commands.",
                   "Check the action format, coordinate convention and IK error details."),
        "inference": ("Inference timing infeasible", detail,
                      "Check execution steps, chunk horizon and measured inference delay. Reduce model latency or explicitly choose a feasible experiment setting before retrying."),
        "runtime": ("Runtime failed", "The error source has not been identified.",
                    "Inspect the original error below and the runtime terminal."),
    }.get(source, ("Runtime failed", "The operation failed.", "Inspect the original error below."))
    refused = "connection refused" in lower or "connectionrefusederror" in lower
    if source == "model" and "policy backend identity mismatch" in lower:
        title = "Model configuration mismatch"
        reason = detail
        action = "Check the model service's reported settings against the selected config, then restart the matching model service."
    elif source == "model" and refused:
        title = "Model service unavailable"
        reason = "The configured model endpoint refused the connection."
        action = "Wait for model loading to finish and confirm the endpoint is listening; then Prepare again."
    elif source == "camera":
        if refused or "timeout" in lower or "timed out" in lower:
            reason = "The camera service or camera did not respond in time."
        if "frame didn't arrive" in lower or "frame is stale" in lower:
            reason = "The camera did not deliver a fresh frame."
    elif source == "robot" and ("no such device" in lower or "errno 19" in lower):
        title = "Robot interface missing"
        reason = "The configured robot interface does not exist in the operating system."
        action = "Restore the configured CAN interface first. Restarting the model will not restore it."
    elif source == "action" and "candidate decoding" in lower:
        title = "Algorithm / adapter incompatibility"
        reason = detail
        action = "The selected algorithm requires complete candidate action decoding. Check the policy adapter's implementation and capability declaration."
    elif source == "action" and "normalized gripper" in lower:
        title = "Gripper target out of range"
        reason = detail
        action = "Check the observation opening, predicted gripper delta and adapter output handling before Start."
    elif source == "runtime" and any(term in lower for term in (
        "inference_deadline_exceeded", "deadline_exceeded_before_decode",
        "deadline_exceeded_before_start",
    )):
        title = "Inference response rejected by deadline check"
        reason = detail
        action = "Inspect request and response deadlines, model round-trip time and time waiting before decoding in the runtime log."
    return {"source": source, "stage": origin.get("stage", ""),
            "target": origin.get("target", ""), "title": title,
            "reason": reason, "action": action, "detail": detail}
