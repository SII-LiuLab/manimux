# ChunkTrust numerical fixtures

These are CPU-only validation inputs and oracle results. They do not contain
model weights or camera observations and do not establish model or robot results.

- `core_inputs.npz`: synthetic float32 velocity, actions and history generated
  by the recipe recorded in `core_golden.json`; loading disables pickle.
- `core_golden.json`: 25 three-decision sequences, including one numerical/state
  consumer case for each of the 19 AHS options. Expected values were generated
  with the original PI05 integration's AST-extracted numerical methods at
  ChunkTrust commit `a9fd57412b3a19a0728cd726b0558dd89395fb84`, independently
  of the ManiMux adapter. The original source path/hash, input hash and NumPy
  version are recorded. Extra cases cover empty history, T=1, trend, intra-only
  and stateless qmix selection.
- `duck_replans.json`: unchanged ChunkTrust `examples/data/duck_replans.json`
  from that commit, SHA-256
  `4b54ae0dce47a354872210c7b062835e04c15634857886bc524cf77499984c65`.
  Its 23 recorded decisions validate score-to-K arithmetic only. They cannot
  reconstruct raw velocity spectra or imply physical task success.

From the ManiMux checkout, run:

```bash
python scripts/validation/chunktrust_core_probe.py --output /path/to/core-report.json
```

Optionally clone the public ChunkTrust repository separately and add
`--reference /path/to/ChunkTrust` to verify the golden sequences against its
original source again. The probe reads the pinned Git objects, not the reference
checkout's current branch. It extracts numerical methods without importing the
model stack. No sibling repository or network access is needed for the default
probe.

Float diagnostics use relative tolerance `2e-6` and absolute tolerance `2e-7`;
candidate choices, posterior keys and RNG state are checked exactly except for
floating posterior values. Transactions additionally verify speculative-state
isolation, discard, duplicate/stale rejection, complete nondefault configuration
across reset/clone/JSON replay and bad-input rejection. Restoring a runtime RNG
snapshot requires the recorded NumPy version. Source/license attribution is in
`licenses/THIRD_PARTY_NOTICES.md`.
