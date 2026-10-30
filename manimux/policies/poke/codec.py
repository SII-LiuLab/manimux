"""Single wrist RGB and absolute flange poses in Poke's calibrated model frame."""

import base64
import io
import json

import numpy as np
from PIL import Image


def require_default_sampling(request):
    for name in (
        "action_condition",
        "condition_weights",
        "paint_action_prefix",
        "paint_delay_steps",
        "aac_num_samples",
        "bid_num_samples",
    ):
        if getattr(request, name, None) is not None:
            raise ValueError(f"Poke API does not support {name}")
    if getattr(request, "autohorizon", False):
        raise ValueError("Poke API does not support AutoHorizon")


def pose_rows(value, *, label):
    rows = np.array(value, dtype=float, copy=True)
    if rows.ndim != 2 or rows.shape[1] != 8 or not len(rows):
        raise ValueError(f"{label}: expected nonempty (T, 8), got {rows.shape}")
    invalid = np.argwhere(~np.isfinite(rows))
    if invalid.size:
        raise ValueError(
            f"{label}: non-finite count={len(invalid)}, first_index={invalid[0].tolist()}"
        )
    norms = np.linalg.norm(rows[:, 3:7], axis=1)
    invalid = np.flatnonzero(~np.isclose(norms, 1.0, atol=1e-3, rtol=0))
    if invalid.size:
        raise ValueError(
            f"{label}: quaternion must be normalized; row={invalid[0]}, norm={norms[invalid[0]]}"
        )
    invalid = np.flatnonzero((rows[:, 7] < -0.01) | (rows[:, 7] > 1.01))
    if invalid.size:
        raise ValueError(
            f"{label}: gripper outside [0, 1] tolerance 0.01; "
            f"row={invalid[0]}, value={rows[invalid[0], 7]}"
        )
    rows[:, 3:7] /= norms[:, None]
    return rows


class PokeCodec:
    def __init__(self, config):
        self.camera = config["adapter"]["camera"]
        self.max_size = config["options"].get("image_max_size", [640, 360])
        if (
            not isinstance(self.max_size, (list, tuple))
            or len(self.max_size) != 2
            or any(type(v) is not int or v < 1 for v in self.max_size)
        ):
            raise ValueError("Poke image_max_size must contain two positive integers")
        self.schema = None

    def bind_schema(self, schema):
        state = schema["inputs"]["properties"]["state"]
        output = schema["output"]["properties"]["output"]
        item = output["items"]
        if (
            state.get("minItems"),
            state.get("maxItems"),
            item.get("minItems"),
            item.get("maxItems"),
        ) != (8, 8, 8, 8):
            raise ValueError("Poke schema must declare eight state/action coordinates")
        if schema["image"]["mode"] != "RGB" or "PNG" not in schema["image"]["formats"]:
            raise ValueError("Poke schema must accept RGB PNG")
        self.schema = schema

    def encode(self, request):
        require_default_sampling(request)
        if self.schema is None:
            raise ValueError("Poke schema has not been loaded")
        state = np.asarray(getattr(request, "model_state", None), dtype=float)
        if state.shape != (8,):
            raise ValueError("Poke requires a calibrated eight-coordinate model_state")
        state = pose_rows(state[None], label="Poke state")[0]
        if not 0 <= state[7] <= 1:
            raise ValueError("Poke observed gripper must be in [0, 1]")
        instruction = request.instruction
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Poke instruction must be nonempty")
        limit = self.schema.get("text", {}).get("max_chars")
        if limit is not None and len(instruction) > limit:
            raise ValueError(f"Poke instruction exceeds {limit} characters")
        frame = request.observation.frames[self.camera]
        image = Image.fromarray(frame.data)
        image.thumbnail(tuple(self.max_size), Image.Resampling.BILINEAR)
        width, height = image.size
        spec = self.schema["image"]
        if (
            width * height > spec["max_pixels"]
            or max(width / height, height / width) > spec["max_aspect_ratio"]
        ):
            raise ValueError("Poke image exceeds the model schema limits")
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        observation = {"instruction": instruction, "state": state.tolist()}
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": json.dumps(observation, allow_nan=False)},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,"
                            + base64.b64encode(stream.getvalue()).decode("ascii")
                        },
                    },
                ],
            }
        ], observation

    def decode(self, raw):
        try:
            choice = raw["choices"][0]
            message = choice["message"]
            calls = message["tool_calls"]
            if (
                choice["finish_reason"] != "tool_calls"
                or message["role"] != "assistant"
                or len(calls) != 1
                or calls[0]["type"] != "function"
                or calls[0]["function"]["name"] != "execute_actions"
            ):
                raise ValueError("expected one execute_actions tool call")
            output = json.loads(calls[0]["function"]["arguments"])["output"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid Poke completion: {exc}") from None
        rows = pose_rows(output, label="Poke output")
        metadata = raw.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError("Poke metadata must be an object")
        shape = metadata.get("action_shape")
        if shape is not None and json.loads(shape) != list(rows.shape):
            raise ValueError("Poke metadata.action_shape disagrees with delivered actions")
        spec = self.schema["output"]["properties"]["output"]
        if not spec.get("minItems", 1) <= len(rows) <= spec.get("maxItems", float("inf")):
            raise ValueError(f"Poke stateless output rows={len(rows)} violate model schema")
        clipped = np.flatnonzero((rows[:, 7] < 0) | (rows[:, 7] > 1)).tolist()
        rows[:, 7] = np.clip(rows[:, 7], 0, 1)
        return {
            "format": "native",
            "action_semantics": "absolute_model_flange_xyz_xyzw",
            "actions": rows,
            "sampling_stats": {"poke_api": {"gripper_clipped_rows": clipped}},
        }
