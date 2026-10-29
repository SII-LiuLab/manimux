# ChunkTrust offline fixtures

Run from an installed ManiMux checkout with the XPolicyLab transport extra and
the separately pinned ChunkTrust package available:

```bash
python scripts/validation/chunktrust_offline_probe.py --output /tmp/chunktrust-probe
```

The output directory must be outside the checkout. The probe saves resolved
configuration, dependency versions, request/response wire bytes, input arrays,
selection and history snapshots, and a JSON report. Network sockets are forbidden.
It constructs no robot driver or learned model. All observations and velocities
in the end-to-end checks are synthetic; passing this probe is not a station
qualification, model reproduction, or physical deployment result.

`synthetic_reference.json` specifies deterministic synthetic arrays and frozen
expected selection diagnostics from the **unchanged published** PI selector at
ChunkTrust commit `a9fd57412b3a19a0728cd726b0558dd89395fb84`. Its first selected
length is 39, which is deliberately not a candidate in `[10,20,30,40,50]`.
The probe runs the actual ManiMux strategy and checks these captured values;
it does not regenerate expected values from the strategy under test.

`recorded_decision_scores.json` retains only the 23 recorded score/candidate/K
triples and provenance from `examples/data/duck_replans.json` in that same
[ChunkTrust release](https://github.com/hf618/ChunkTrust/tree/a9fd57412b3a19a0728cd726b0558dd89395fb84).
The original export is MIT licensed, copyright 2026 Fanding Huang; its full
license is retained in [LICENSE](LICENSE), copied from the release's root
[license](https://github.com/hf618/ChunkTrust/blob/a9fd57412b3a19a0728cd726b0558dd89395fb84/LICENSE).
This score replay checks selection arithmetic only. It contains no recorded
velocity spectrum and is an interrupted episode, not a success-rate sample.

The dispatch checks execute the real `EdgeRuntime._send_command` on an
uninitialized runtime shell whose controller is a disconnected recorder.
Warmup uses the real `PolicyWarmup` and `PreviewPlanner`, an immediate in-memory
policy worker and the actual RESET acknowledgment fence. Pause/Home tests cover
strategy invalidation/reset semantics, not the hardware lifecycle or homing.
