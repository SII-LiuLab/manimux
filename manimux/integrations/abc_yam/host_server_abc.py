"""ABC-DiT Bimanual YAM inference server.

Mirrors ``host_server_yam.py`` (MolmoAct2) so both policies expose the same wire
protocol to ManiMux:

  * 3 cameras in fixed order [top, left, right]
  * raw robot state is shape (14,)  (per-arm 6 joints + 1 gripper, two arms)
  * actions are absolute joint positions in radians; gripper dims are [0, 1]

Wire protocol:

    GET  /act        -> health check, returns {"status": "ok", ...}
    POST /act        -> action inference
        request body  (json_numpy):
            {
              "top_cam":     ndarray(H, W, 3) uint8 RGB,
              "left_cam":    ndarray(H, W, 3) uint8 RGB,
              "right_cam":   ndarray(H, W, 3) uint8 RGB,
              "instruction": str,
              "state":       ndarray(14,) float32,
              "timestamp":   float (optional),
              "num_steps":   int   (optional, default 10 flow-matching steps),
              "action_condition":         ndarray(30, 14) float32 (optional, Pi-RTC),
              "action_condition_weights": ndarray(30,)    float32 (optional, Pi-RTC),
              "rtc_beta":    float (optional, default 5.0),
            }
        response body (json_numpy):
            {"actions": ndarray(N, D) float32, "dt_ms": float}

Run:

    manimux-abc-server --host 0.0.0.0 --port 8300 \
        --checkpoint checkpoints/pretrained/abc/abc_dit_xl_200k_model.pt
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from pathlib import Path
from typing import Any

import json_numpy
import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from .abc_minimal.config import ClipConfig, DiTConfig
from .abc_minimal.dit import CLIPTextEmbedder, DiTPolicy, infer_dit_shape, load_pretrained
from .abc_minimal.preprocess import (
    load_norm_stats,
    normalize,
    parse_norm_stats,
    resize_pad_normalize,
    unnormalize,
)

# Patches the stdlib `json` module so np.ndarray round-trips through JSON.
# Must be called before any json.dumps/loads we rely on.
json_numpy.patch()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("abc.yam.server")

DEFAULT_CHECKPOINT = "checkpoints/pretrained/abc/abc_dit_xl_200k_model.pt"
DEFAULT_PROMPT = "put the bottle into the bin"
DEFAULT_NUM_STEPS = 10
DEFAULT_RTC_BETA = 5.0
STATE_DIM = 14
NUM_CAMERAS = 3
# Per-arm layout is [6 joints, 1 gripper]; these are the two gripper columns.
GRIPPER_DIMS = (6, 13)
# Wire field -> camera key the checkpoint was trained with.
CAMERA_FIELDS = {"top_cam": "top", "left_cam": "left", "right_cam": "right"}


class Policy:
    """Load ABC-DiT once and serve single-observation action chunks.

    The inference body is the non-sim half of ``abc_minimal.eval_policy.SimPolicy``:
    z-score the state, letterbox+ImageNet-normalize each camera, run rectified-flow
    sampling, then un-normalize the actions.
    """

    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda:0",
        prompt: str = DEFAULT_PROMPT,
        diffusion_steps: int = DEFAULT_NUM_STEPS,
        clip_cache_dir: str | None = None,
        norm_stats_path: str | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.diffusion_steps = diffusion_steps
        self.checkpoint = str(Path(checkpoint).expanduser().resolve())
        self.default_prompt = prompt

        # Recover hidden_size/depth/num_heads from the checkpoint so any ABC-DiT
        # size loads without CLI flags (same call the upstream server makes).
        shape = infer_dit_shape(self.checkpoint)
        log.info("Checkpoint DiT shape: %s", shape or "(defaults)")
        self.model_config = DiTConfig(**shape)

        log.info("Building DiTPolicy and loading %s", self.checkpoint)
        self.model = DiTPolicy(self.model_config).to(self.device)
        ckpt = load_pretrained(self.model, self.checkpoint)
        self.model.eval()

        if ckpt.get("norm_stats") is not None:
            self.norm_stats = parse_norm_stats(ckpt["norm_stats"])
            stats_source = "checkpoint"
        elif norm_stats_path:
            stats_path = Path(norm_stats_path).expanduser().resolve()
            self.norm_stats = load_norm_stats(stats_path)
            stats_source = str(stats_path)
        else:
            raise ValueError(
                f"no norm_stats embedded in checkpoint: {self.checkpoint}; "
                "pass --norm-stats-path for model-only checkpoints"
            )
        log.info("Loaded norm_stats from %s (train step %s)", stats_source, ckpt.get("step"))

        clip_config = (
            ClipConfig(cache_dir=clip_cache_dir) if clip_cache_dir else ClipConfig()
        )
        log.info("Loading CLIP text encoder from %s", clip_config.cache_dir)
        self.embedder = CLIPTextEmbedder(clip_config, device=self.device)
        # Warm the prompt memo cache so the first request does not pay for BPE.
        self.embedder.encode([prompt])

        self._gripper_dims = [d for d in GRIPPER_DIMS if d < self.model_config.action_dim]

        # Robot clients poll at ~30 Hz; flow sampling is not concurrency-safe.
        self._lock = threading.Lock()

    @property
    def camera_keys(self) -> tuple[str, ...]:
        return tuple(self.model_config.camera_keys)

    @property
    def chunk_length(self) -> int:
        return int(self.model_config.chunk_length)

    @property
    def action_dim(self) -> int:
        return int(self.model_config.action_dim)

    def predict(
        self,
        *,
        top_cam: Any,
        left_cam: Any,
        right_cam: Any,
        instruction: str,
        state: Any,
        num_steps: int | None = None,
        action_condition: Any = None,
        condition_weights: Any = None,
        rtc_beta: float = DEFAULT_RTC_BETA,
    ) -> np.ndarray:
        images = {"top": top_cam, "left": left_cam, "right": right_cam}
        state_np = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_np.shape != (self.model_config.state_dim,):
            raise ValueError(
                f"state must have shape ({self.model_config.state_dim},), got {state_np.shape}"
            )
        steps = self.diffusion_steps if num_steps is None else int(num_steps)
        prompt = instruction.strip() or self.default_prompt

        with self._lock, torch.no_grad():
            task_vec = self.embedder.encode([prompt]).to(self.device)
            batch = {
                "state": torch.from_numpy(
                    normalize(state_np, self.norm_stats["state"])[None]
                )
                .float()
                .to(self.device),
                "actions": torch.zeros(
                    1,
                    self.model_config.chunk_length,
                    self.model_config.action_dim,
                    device=self.device,
                ),
                "images": {
                    cam: resize_pad_normalize(_to_chw_uint8(images[cam]))
                    .unsqueeze(0)
                    .to(self.device)
                    for cam in self.camera_keys
                },
                "task_vec_clip": task_vec,
            }

            if (action_condition is None) != (condition_weights is None):
                raise ValueError(
                    "action_condition and action_condition_weights must be provided together"
                )
            if action_condition is None:
                actions = self.model.sample_actions(batch, num_steps=steps)
            else:
                condition_t, weights_t = self._rtc_tensors(
                    action_condition, condition_weights, batch["state"].dtype
                )
                actions = self.model.sample_actions_pi_rtc(
                    batch,
                    condition_t,
                    weights_t,
                    beta=float(rtc_beta),
                    num_steps=steps,
                )

        actions_np = actions[0].float().detach().cpu().numpy()
        actions_np = unnormalize(actions_np, self.norm_stats["actions"]).astype(np.float32)
        # Gripper dims are trained normalized to [0, 1]; the reference ABC server
        # clips them before returning, so the same rows reach the robot here.
        actions_np[:, self._gripper_dims] = np.clip(actions_np[:, self._gripper_dims], 0.0, 1.0)
        return actions_np

    def _rtc_tensors(
        self,
        action_condition: Any,
        condition_weights: Any,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        horizon = self.model_config.chunk_length
        dim = self.model_config.action_dim
        condition = np.asarray(action_condition, dtype=np.float32)
        if condition.shape != (horizon, dim):
            raise ValueError(
                f"action_condition must have shape {(horizon, dim)}, got {condition.shape}"
            )
        weights = np.asarray(condition_weights, dtype=np.float32).reshape(-1)
        if weights.shape != (horizon,):
            raise ValueError(
                f"action_condition_weights must have shape {(horizon,)}, got {weights.shape}"
            )
        if not np.isfinite(condition).all():
            raise ValueError("action_condition contains non-finite values")
        if not np.isfinite(weights).all() or np.any((weights < 0) | (weights > 1)):
            raise ValueError("action_condition_weights must be finite values in [0, 1]")
        condition_t = torch.from_numpy(
            normalize(condition, self.norm_stats["actions"])[None]
        ).to(device=self.device, dtype=dtype)
        weights_t = torch.from_numpy(weights[None]).to(device=self.device, dtype=dtype)
        return condition_t, weights_t


def _letterbox(arr: np.ndarray, target_h: int = 224, target_w: int = 224) -> np.ndarray:
    """Aspect-preserving bicubic downscale + centered zero-pad, as HWC uint8.

    ABC's training cache is built by ``export_mcap.py``, which resizes with
    ffmpeg ``scale=224:224:force_original_aspect_ratio=decrease:flags=bicubic``
    followed by a centered ``pad``. ``resize_pad_normalize`` reproduces the
    geometry but interpolates bilinearly, so do the downscale here with bicubic
    to match training; it then sees a 224x224 image and only normalizes.
    """
    from PIL import Image

    h, w = arr.shape[:2]
    if (h, w) == (target_h, target_w):
        return arr
    ratio = max(w / target_w, h / target_h)
    new_h = max(1, int(h / ratio))
    new_w = max(1, int(w / ratio))
    resized = np.asarray(
        Image.fromarray(arr, mode="RGB").resize((new_w, new_h), Image.BICUBIC),
        dtype=np.uint8,
    )
    out = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    out[top : top + new_h, left : left + new_w] = resized
    return out


def _to_chw_uint8(arr: Any) -> np.ndarray:
    """Accept an (H, W, 3) uint8 RGB frame and return the CHW view ABC expects."""
    a = np.asarray(arr)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"image must be HxWx3, got shape {a.shape}")
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(_letterbox(a).transpose(2, 0, 1))


def build_app(policy: Policy) -> FastAPI:
    app = FastAPI(title="ABC-DiT Bimanual YAM server", version="0.1.0")

    @app.get("/act")
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "checkpoint": policy.checkpoint,
                "device": str(policy.device),
                "dtype": "float32",
                "num_cameras": NUM_CAMERAS,
                "camera_order": list(policy.camera_keys),
                "state_dim": policy.model_config.state_dim,
                "action_dim": policy.action_dim,
                "chunk_length": policy.chunk_length,
                "diffusion_steps": policy.diffusion_steps,
                "default_prompt": policy.default_prompt,
            }
        )

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.post("/act")
    async def act(request: Request) -> Response:
        raw = await request.body()
        try:
            payload = json_numpy.loads(raw.decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            return _error_response(400, f"failed to decode json_numpy body: {e}")

        try:
            top_cam = payload["top_cam"]
            left_cam = payload["left_cam"]
            right_cam = payload["right_cam"]
            instruction = str(payload["instruction"])
            state = payload["state"]
        except KeyError as e:
            return _error_response(400, f"missing required field: {e}")

        t0 = time.perf_counter()
        try:
            actions = policy.predict(
                top_cam=top_cam,
                left_cam=left_cam,
                right_cam=right_cam,
                instruction=instruction,
                state=state,
                num_steps=payload.get("num_steps"),
                action_condition=payload.get("action_condition"),
                condition_weights=payload.get("action_condition_weights"),
                rtc_beta=float(payload.get("rtc_beta", DEFAULT_RTC_BETA)),
            )
        except Exception as e:  # noqa: BLE001
            log.exception("inference failed")
            return _error_response(500, f"inference failed: {e}")
        dt_ms = (time.perf_counter() - t0) * 1000.0

        body = json_numpy.dumps({"actions": actions, "dt_ms": dt_ms})
        return Response(content=body, media_type="application/json")

    return app


def _error_response(status: int, message: str) -> Response:
    body = json_numpy.dumps({"error": message})
    return Response(content=body, status_code=status, media_type="application/json")


def warmup(policy: Policy) -> None:
    log.info("Warming up model with dummy frames ...")
    dummy_img = np.zeros((360, 640, 3), dtype=np.uint8)
    dummy_state = np.zeros(policy.model_config.state_dim, dtype=np.float32)
    t0 = time.perf_counter()
    try:
        actions = policy.predict(
            top_cam=dummy_img,
            left_cam=dummy_img,
            right_cam=dummy_img,
            instruction=policy.default_prompt,
            state=dummy_state,
        )
    except Exception:  # noqa: BLE001
        log.exception("warmup inference failed (server will still start)")
        return
    log.info(
        "Warmup OK (%.1f ms, actions %s)",
        (time.perf_counter() - t0) * 1000.0,
        actions.shape,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ABC-DiT Bimanual YAM inference server")
    p.add_argument("--host", default="0.0.0.0", help="bind address (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=8300, help="bind port (default: 8300)")
    p.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=f"ABC-DiT checkpoint path (default: {DEFAULT_CHECKPOINT})",
    )
    p.add_argument("--device", default="cuda:0", help="torch device (default: cuda:0)")
    p.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="fallback task prompt when a request sends an empty instruction",
    )
    p.add_argument(
        "--diffusion-steps",
        type=int,
        default=DEFAULT_NUM_STEPS,
        help=f"rectified-flow sampling steps (default: {DEFAULT_NUM_STEPS})",
    )
    p.add_argument(
        "--clip-cache-dir",
        default=None,
        help="CLIP ViT-B/32 asset directory (default: ~/.cache/clip; downloaded if missing)",
    )
    p.add_argument(
        "--norm-stats-path",
        default=None,
        help="external norm_stats.json for model-only checkpoints",
    )
    p.add_argument("--no-warmup", action="store_true", help="skip warmup pass")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    policy = Policy(
        checkpoint=args.checkpoint,
        device=args.device,
        prompt=args.prompt,
        diffusion_steps=args.diffusion_steps,
        clip_cache_dir=args.clip_cache_dir,
        norm_stats_path=args.norm_stats_path,
    )
    if not args.no_warmup:
        warmup(policy)

    app = build_app(policy)

    import uvicorn

    log.info("Listening on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
