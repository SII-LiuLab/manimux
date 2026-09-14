# UMI DP integration validation — 2026-09-13

All checks below were offline or used temporary localhost model-server ports.
No camera or robot driver was opened, no robot was commanded, and no training
run or dataset/cache modification was performed. The existing CalibWrist Python
3.11 environment was used with missing WebSocket/OpenCV/HDF5/test packages in a
separate `/tmp` package directory; its installed environment was not modified.

## Real checkpoint evidence

The recorded input was a two-frame window from the native pass-ball LeRobot
export. Both implementations received the same RGB images and absolute TCP
poses; random seed 17 controlled the original inference augmentations and DDIM.

| Checkpoint | SHA256 | Result |
|---|---|---|
| 2026-09-08 H16, `epoch=0019-train_loss=0.013.ckpt` | `48dda9ce86239b28cd42d42e7630cfadc0f8bd46a8d89e792949b8832af7569a` | Every observation tensor and all 16×20 native actions exactly match the old PolicyRunner (max error 0). |
| 2026-09-12 H64, `epoch=0019-train_loss=0.012.ckpt` | `640c2ad99f939e5d1a39f32880266a17fd7cefdb960b5e1da9a2af46d14be548` | Every observation tensor and all 64×20 native actions exactly match (max error 0). |
| H16 with PiGDM RTC | H16 SHA above | Same observation, condition, weights and seed: native guided actions exactly match the old enabled RTC sampler (max error 0). |
| 2026-09-04 legacy `latest.ckpt` | `1d7d9def9742b8cc549e5e785e84e61120a8beb7648da60662962fd09b5e4767` | Actual artifact inspection confirms Normalize=false, H16, action dt=.1s and first target offset=1/30s. Legacy forward was not rerun. |

The real H16/H64 models use EMA, 16 DDIM steps, CLIP Normalize=true and .1s
observation spacing. Action dt and first offset are both 1/30s for these newer
artifacts. One cold forward took 513ms for H16 and 283ms for H64; another H16
process took 255ms. The H16 PiGDM forward after an ordinary forward took 90ms.
These single-call timings include initialization effects and are not throughput
or worst-case timing guarantees.

Commands are documented in `XPolicyLab/policy/UMI_DP/README.md`. The optional
`validate --rtc-guidance pigdm` adds the guided numerical comparison. Local
machine-readable results were written to:

- `/tmp/umi-dp-h16-validation.json`
- `/tmp/umi-dp-h64-validation.json`
- `/tmp/umi-dp-rtc-validation.json`
- `/tmp/umi-dp-legacy-identity.json`

## Shared server and adapters

Standard `EVAL_ENV_TYPE=debug bash XPolicyLab/policy/UMI_DP/eval.sh ...` completed
with `[MAIN] eval finished` for H16 plain RGB, H16 encoded RGB, and H64 encoded
RGB. Every run exercised real default inference, PiGDM RTC, batch indices `[3,7]`
and reset followed by another inference. Encoding uses the shared image helper
and decoding occurs in the unmodified shared server. The temporary servers were
terminated by the standard eval script's cleanup trap.

The model-environment launcher successfully created paired deployment configs.
Their artifact identities match exactly; profile merge, delegated strategy
validation and construction of the real Tianji adapter also passed. The checked-in
unbound templates retain the real driver's binding guard.

Logs: `/tmp/umi-dp-h16-server-plain.log`,
`/tmp/umi-dp-h16-server-encoded.log`, `/tmp/umi-dp-h64-server-encoded.log`,
`/tmp/umi-dp-bind-validation.log`. These are temporary local validation artifacts,
not runtime dependencies.

Twelve ManiMux unit tests cover timing, captured-frame identity, state/history
matching, RTC target clocks/tail weights and FK/IK contract rejection. Seven
XPolicyLab tests cover row rotations, relative/absolute roundtrip, RGB encoded
roundtrip, temporal rejection, batch/reset and robot dimensions. All policy shell
scripts passed individual `bash -n`, Python compilation passed, and changed
adapter/launcher/probe files passed Ruff. The real training workspace imports and
the standard H64 `train.sh ... --cfg job` compose successfully.

The final parent-workspace regression also included the configurable motion
limit modes, Tianji driver and camera backend tests: **129 passed** across
`test_executors.py`, `test_config.py`, `test_tianji_driver.py`,
`test_taccap_camera.py` and `test_umi_dp_tianji.py`. Camera tests include capture
advancing immediately after a reader releases its lock, for both TacCap and
RealSense: the server retains the returned image's timestamp. These timestamps
are host receipt times, not hardware exposure times.

## Runtime timing limitation

`python scripts/validation/umi_dp_tianji_decode_probe.py` calls the real vendored
libKine on small, reachable synthetic two-arm paths, without hardware. Three
H16 decodes took 19–23ms; three H64 decodes took 77–79ms. The adapter currently
runs inline, so this exceeds a 250Hz control tick's 4ms budget. The unified loop
and its frequency were not changed; the IK branch threshold was not relaxed.
Coordinating the shared process decoder with measured history/RTC needs separate
work before claiming the whole 250Hz real-motion chain is ready.

The optional differential IK follow-up is documented in
[tianji-diff-ik.md](tianji-diff-ik.md), including real reference-algorithm parity,
failure coverage, shared-profile binding and full-chunk timing. The default
analytic backend remains unchanged; differential IK does not resolve the inline
decoding scheduling limitation.

The provided continuous gripper profile differs from CalibWrist's close latch;
shared joint interpolation differs from per-servo SE(3) interpolation. These are
explicit execution differences, not model-parity failures. Real hardware timing,
task success, full training, installation into a clean model venv, simulator
rollout and the non-default soft-inpaint mode's real-weight forward remain
unverified. See the runbook for these boundaries.
