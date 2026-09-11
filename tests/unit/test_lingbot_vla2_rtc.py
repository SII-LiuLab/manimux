from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

RTC_PATH = (
    Path(__file__).resolve().parents[2] / "XPolicyLab/policy/LingBot_VLA2/rtc.py"
)


def _load_rtc_module():
    spec = importlib.util.spec_from_file_location("lingbot_vla2_rtc_test", RTC_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_guided_velocity_pulls_clean_estimate_toward_condition() -> None:
    rtc = _load_rtc_module()
    sample = torch.zeros(1, 2, 3)
    target = torch.ones_like(sample)
    weights = torch.ones_like(sample)
    velocity = rtc.guided_velocity(
        lambda value: torch.zeros_like(value),
        sample,
        torch.tensor(0.5),
        target,
        weights,
        beta=1.0,
    )
    torch.testing.assert_close(velocity, -torch.ones_like(sample))
    next_sample = sample + torch.tensor(-0.1) * velocity
    assert torch.all(next_sample > sample)


def test_zero_rtc_weights_preserve_native_velocity() -> None:
    rtc = _load_rtc_module()
    sample = torch.randn(1, 2, 3)
    native = rtc.guided_velocity(
        lambda value: 2 * value,
        sample,
        torch.tensor(0.5),
        torch.ones_like(sample),
        torch.zeros_like(sample),
        beta=5.0,
    )
    torch.testing.assert_close(native, 2 * sample)


def test_rtc_rejects_weights_outside_soft_mask_range() -> None:
    rtc = _load_rtc_module()
    sample = torch.zeros(1, 2, 3)
    with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
        rtc.guided_velocity(
            lambda value: value,
            sample,
            torch.tensor(0.5),
            sample,
            torch.full_like(sample, 1.1),
            beta=5.0,
        )


def test_encode_raw_condition_preserves_yam_arm_order() -> None:
    rtc = _load_rtc_module()
    condition = np.arange(28, dtype=np.float32).reshape(2, 14)
    encoded = rtc.encode_raw_condition(
        condition,
        {"arm_dim": [6, 6], "ee_dim": [1, 1]},
    )
    np.testing.assert_array_equal(
        encoded["action.arm.position"],
        np.concatenate([condition[:, :6], condition[:, 7:13]], axis=-1),
    )
    np.testing.assert_array_equal(
        encoded["action.effector.position"], condition[:, [6, 13]]
    )


def test_sampler_applies_guidance_inside_each_flow_step() -> None:
    rtc = _load_rtc_module()

    class FakeQwen:
        @staticmethod
        def forward(**_):
            return None, "cache", None

    class FakeFlow:
        config = SimpleNamespace(
            n_action_steps=2,
            max_action_dim=3,
            num_steps=2,
            use_cache=True,
        )
        qwenvl_with_expert = FakeQwen()

        @staticmethod
        def embed_prefix(*_, **__):
            return (
                torch.zeros(1, 1, 1),
                torch.ones(1, 1, dtype=torch.bool),
                torch.zeros(1, dtype=torch.bool),
                torch.zeros(1, 1, dtype=torch.long),
                None,
                None,
            )

        @staticmethod
        def predict_velocity(*args, **_):
            sample = args[3]
            return torch.zeros_like(sample)

    zeros = torch.zeros(1, 2, 3)
    output = rtc.sample_actions_rtc(
        FakeFlow(),
        torch.zeros(1, 1, 1),
        torch.ones(1, 1, dtype=torch.bool),
        torch.zeros(1, 1, dtype=torch.long),
        torch.ones(1, 1, dtype=torch.bool),
        torch.zeros(1, 1),
        action_condition=torch.ones_like(zeros),
        condition_weights=torch.ones_like(zeros),
        beta=1.0,
        noise=zeros,
        _make_masks=lambda *_: torch.ones(1, 1, 1, dtype=torch.bool),
    )
    assert torch.all(output > 0)


def test_packed_observation_anchor_and_relative_condition():
    rtc = _load_rtc_module()
    bridge = rtc.LingBotRtcBridge(None, {"arm_dim": [6, 6], "ee_dim": [1, 1]},
                                 return_relative=True)
    packed = np.arange(14, dtype=np.float32)
    state = bridge._current_state({"observation.state": packed})
    np.testing.assert_array_equal(state["observation.state.arm.position"],
                                  np.r_[packed[:6], packed[7:13]])
    np.testing.assert_array_equal(state["observation.state.effector.position"], packed[[6, 13]])
    transform = SimpleNamespace(
        action_subtract_state={"action.arm.position": True, "action.effector.position": False},
        action_relative_type={"action.arm.position": "vector"},
        normalizer=SimpleNamespace(normalize=lambda data: data),
        feature_config=SimpleNamespace(joints=["arm.position", "effector.position"],
                                       joints_max_dim={"arm.position": 14, "effector.position": 2}),
        model_config=SimpleNamespace(max_action_dim=20),
    )
    absolute = np.repeat(packed[None, :], 2, axis=0)
    absolute[:, :6] += .25
    absolute[:, 7:13] -= .5
    absolute[:, [6, 13]] = [.2, .8]
    target, weights = rtc.normalize_condition(
        transform, rtc.encode_raw_condition(absolute, bridge.robot_info),
        np.array([1, .5], dtype=np.float32), state,
    )
    torch.testing.assert_close(target[:, :6], torch.full((2, 6), .25))
    torch.testing.assert_close(target[:, 6:12], torch.full((2, 6), -.5))
    torch.testing.assert_close(target[:, 14:16], torch.tensor([[.2, .8], [.2, .8]]))
    assert torch.count_nonzero(weights[:, 12:14]) == 0
    assert torch.count_nonzero(weights[:, 16:]) == 0


def test_relative_rtc_output_does_not_add_anchor_twice():
    rtc = _load_rtc_module()
    calls = []
    native = {"action.arm.position": np.full((2, 12), .25),
              "action.effector.position": np.full((2, 2), .8)}
    transform = SimpleNamespace(
        reverse_pad_and_concat=lambda data: calls.append("unpad") or native,
        normalizer=SimpleNamespace(unnormalize=lambda data: calls.append("unnormalize") or data),
        unapply=lambda data: calls.append("absolute") or data,
    )
    bridge = rtc.LingBotRtcBridge(SimpleNamespace(vla=SimpleNamespace(feature_transform=transform)),
                                 {}, return_relative=True)
    assert bridge._decode_sample({}) is native
    assert calls == ["unpad", "unnormalize"]
    bridge.return_relative = False
    bridge._decode_sample({})
    assert calls[-1] == "absolute"
