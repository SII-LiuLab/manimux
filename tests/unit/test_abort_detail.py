import pytest

from manimux.runtime.edge import _exception_detail


def test_a_plain_error_keeps_its_message():
    assert _exception_detail(RuntimeError("right_arm: stale arm feedback")) == (
        "RuntimeError: right_arm: stale arm feedback"
    )


def test_a_group_expands_every_member():
    # robot/base.py raises this shape when a command fails and the stop also fails.
    group = ExceptionGroup(
        "command and stop failed",
        [
            RuntimeError("right_arm: command exceeds tracking error limit"),
            RuntimeError("arm stop incomplete"),
        ],
    )
    assert _exception_detail(group) == (
        "ExceptionGroup: command and stop failed "
        "[RuntimeError: right_arm: command exceeds tracking error limit; "
        "RuntimeError: arm stop incomplete]"
    )


def test_an_explicit_cause_is_followed():
    try:
        try:
            raise ValueError("invalid measured joints")
        except ValueError as cause:
            raise RuntimeError("left_arm: read failed") from cause
    except RuntimeError as exc:
        detail = _exception_detail(exc)
    assert detail == ("RuntimeError: left_arm: read failed <- ValueError: invalid measured joints")


def test_a_suppressed_context_is_not_followed():
    try:
        try:
            raise ValueError("noise")
        except ValueError:
            raise RuntimeError("real failure") from None
    except RuntimeError as exc:
        assert _exception_detail(exc) == "RuntimeError: real failure"


def test_a_long_cause_chain_stops_at_the_depth_limit():
    exc: BaseException = RuntimeError("f")
    for message in ("e", "d", "c", "b", "a"):
        outer = RuntimeError(message)
        outer.__cause__ = exc
        exc = outer
    detail = _exception_detail(exc)
    # Four levels are rendered; the rest of the chain is dropped, not appended.
    assert detail == ("RuntimeError: a <- RuntimeError: b <- RuntimeError: c <- RuntimeError: d")


@pytest.mark.parametrize("detail", ["", "RuntimeError: boom"])
def test_the_recorder_writes_the_detail(tmp_path, detail):
    import json

    from manimux.recording import EpisodeRecorder

    recorder = EpisodeRecorder(tmp_path, "rollout-001", {"left_arm": 8}, {})
    recorder.abort("RuntimeError", detail=detail)
    result = json.loads((tmp_path / "rollout-001.partial" / "result.json").read_text())
    assert result["detail"] == detail
    events = [
        json.loads(line)
        for line in (tmp_path / "rollout-001.partial" / "events.jsonl").read_text().splitlines()
    ]
    aborted = next(e for e in events if e["kind"] == "episode_aborted")
    assert aborted["detail"] == detail
