"""Optional strategy feedback at the validated command submission boundary."""

from dataclasses import dataclass
from typing import Literal

from manimux.types import RobotCommand, copy_group_vector

FeedbackSource = Literal["submitted", "preview", "validated_only"]


@dataclass(frozen=True, slots=True)
class CommandFeedback:
    """A command snapshot, not an acknowledgment of physical motion.

    ``submitted`` means the runtime's controller call returned successfully.
    It does not establish CAN transmission or measured arrival. A missing
    reference identifies holds outside an actively executing task plan.
    """

    command: RobotCommand
    reference_plan_id: str | None
    source: FeedbackSource


def notify_command_feedback(
    strategy: object,
    *,
    command: RobotCommand,
    reference_plan_id: str | None,
    source: FeedbackSource,
) -> None:
    """Notify opt-in strategies only, keeping command mutation out of the hook."""
    callback = getattr(strategy, "on_command_feedback", None)
    if callable(callback):
        callback(CommandFeedback(
            command=RobotCommand(
                copy_group_vector(command.groups), command.monotonic_ns, command.plan_id,
            ),
            reference_plan_id=reference_plan_id,
            source=source,
        ))
