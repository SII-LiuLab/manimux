"""Measured UMI history and coordinate conversion for trained RTC prefixes."""

from dataclasses import dataclass, fields

import numpy as np

from manimux.policy_adapter.umi_dp.history import HistoryStrategy
from manimux.policy_adapter.umi_dp.tianji import UmiDpTianjiAdapter
from manimux.runtime.train_time_rtc import TrainTimeRtcRequest


def align_fixed_prefix(request, timeline, *, offset_ns, dt_ns, group_order) -> bool:
    """Read execution references at model target times; never interpolate a mask.

    Missing coverage clears the entire prefix instead of extrapolating commands
    or sending a mask with holes, which the trained sampler cannot represent.
    """
    if not request.prefix_steps:
        return True
    rows = []
    for index in range(request.prefix_steps):
        target = request.observation_time_ns + offset_ns + index * dt_ns
        reference = timeline.handoff_reference(target, 0)
        if reference is None:
            request.prefix_steps = 0
            request.action_condition = request.condition_weights = None
            return False
        old = np.concatenate([reference.groups[name] for name in group_order], axis=1)
        times = reference.start_time_ns + np.arange(len(old)) * reference.dt_ns
        rows.append([np.interp(target, times, old[:, column]) for column in range(old.shape[1])])
    request.action_condition[:] = 0
    request.action_condition[: request.prefix_steps] = rows
    return True


class TrainTimeRtcHistoryStrategy(HistoryStrategy):
    """Keep measured history unchanged, with a separate hard-prefix alignment."""

    def __init__(self, config):
        super().__init__(config)
        identity = config["policy"].get("expected_backend") or {}
        self.trained_prefix_lengths = identity.get("model", {}).get("trained_prefix_lengths")

    def build_submission(self, **kwargs):
        self.history.observe(kwargs["snapshot"])
        window = self.history.window(kwargs["now_ns"])
        if window is None:
            return None
        submission = self.delegate.build_submission(**{**kwargs, "snapshot": window})
        if submission is not None and not align_fixed_prefix(
            submission.request,
            kwargs["timeline"],
            offset_ns=self.offset_ns,
            dt_ns=self.dt_ns,
            group_order=self.group_order,
        ):
            self.delegate.clear_condition(submission.request.request_seq)
            submission.event_fields.update(
                conditioned=False,
                prefix_steps=0,
                forecast_delay=0,
                condition_reason="incomplete_committed_prefix",
            )
        if submission is not None and self.trained_prefix_lengths is not None:
            submission.event_fields["prefix_in_training_support"] = (
                submission.request.prefix_steps in self.trained_prefix_lengths
            )
        return submission


@dataclass(slots=True)
class TrainTimeUmiRequest(TrainTimeRtcRequest):
    xpolicylab_state: dict | None = None
    xpolicylab_additional_info: dict | None = None


class TrainTimeRtcUmiDpTianjiAdapter(UmiDpTianjiAdapter):
    """Reuse UMI FK, gripper mapping and decoding; retain the typed sampler request."""

    def prepare_request(self, request):
        prepared = super().prepare_request(request)
        return TrainTimeUmiRequest(
            **{
                item.name: getattr(prepared, item.name)
                for item in fields(TrainTimeUmiRequest)
                if item.name != "prefix_steps"
            },
            prefix_steps=getattr(request, "prefix_steps", 0),
        )


def build_strategy(config):
    return TrainTimeRtcHistoryStrategy(config)
