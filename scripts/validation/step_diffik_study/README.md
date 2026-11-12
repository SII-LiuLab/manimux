# Step DiffIK offline study

This directory is independent of the chunk study and production configuration.
It never creates a robot/controller or a policy client. `RobotModel` loads the
vendor's offline FK library and the recorded tool assembly; it does not connect
to hardware. Candidate states and task success are not simulated.

Evidence: `data/analysis/step_diffik_study/20261019/`.

- `pack_plate_conclusion.txt`, `pack_bottle_conclusion.txt`: short conclusions.
- `pack_plate_report.html`, `pack_bottle_report.html`: baseline, candidates,
  separate configuration/status/session strata, layer comparisons and plots.
- `candidate_parameters_review.yaml`: review only, **not a deployment YAML**.
- `cohort.json`, `episodes/*/identity.json`: inclusion/exclusion, frozen config,
  source hashes, input hashes and time-trim/handoff audit.
- `episodes/*/baseline_parity.json`: free-running and historical-seed parity.
- `episodes/*/candidates/*/trace.npz`: candidate-owned previous commands, newly
  solved references, limiter inputs, saturation flags, reasons and local timing.
- `baseline_forensics.json`: first-failure windows, joint bounds and guard rates.
- `verification.json`, `tcp_verification.json`: hold/reset and TCP Jacobian tests.
- `source_snapshot/`, `script_snapshot/`: reproducibility snapshots. Native FK
  and TCP asset historical hashes were not recorded; see
  `asset_provenance_limit.json` for that distinction.

## Reproduce

From the repository root, using the existing analysis environment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/validation/step_diffik_study/reproduce.py \
  --output data/analysis/step_diffik_study/reproduction-001
```

The destination must not exist. This recomputes the baseline, 21 parameter
single factors, flange-objective combinations, the TCP-objective single factor,
TCP combinations and frozen holdout candidates. It uses two offline workers.
The complete study contains 298 candidate/episode runs plus baseline and tests.
It preserves the original campaign and every production file. Plot-only
dependencies are isolated in the original evidence directory's `plot-deps/`.
NumPy/SciPy/OSQP runtime versions are recorded in `environment.json`.

To rerun only the independent verification against existing evidence:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  scripts/validation/step_diffik_study/verify.py
```

Individual phases are available through `run.py`:
`baseline`, `single`, `rank-single`, `joint`, `tcp`, `tcp-joint`, `holdout`,
`refresh-metrics`. Joint spaces and frozen holdout choices are JSON inputs.
`STEP_DIFFIK_OUTPUT` selects an output directory and `STEP_DIFFIK_SOURCE_ROOT`
selects archived runtime sources. Candidate caching avoids repeating completed
trials; use a fresh output directory for independent reproduction.

## Interpretation

The committed TCP trajectory is sampled at the original tick timestamps after
the recorded latency trim and handoff skip. The skip is always the historical
value, 1; it changes source rows without delaying the handoff. Initial gaps hold
the prior command and clear execution history. Every active candidate tick runs
the real pose limiter, differential QP and direct executor. The next seed is the
candidate's previous executor command, not the historical reference or state.
QP failure atomically holds both arms and resets primal/dual and executor history.
The nominal IK/executor period is 10 ms, independent of measured wall-clock jitter.

The mathematical replay disables nondeterministic local deadline rejection.
Historical budget failures would be injected with their reset/hold semantics;
there are none in this cohort. Independent synthetic tests verify actual budget
rejection behavior. Offline elapsed times cannot validate robot deadlines.

Every baseline reference and command is bit-identical to the recorded arrays.
This is command-generation parity, not a simulation of controller tracking,
contacts or policy feedback. Recorded state is used only for historical
attribution. Same-tick state was read before sending the corresponding command.
The future target stream stays historical even though a real changed command
would change observations and future policy outputs.

Metrics use Euclidean TCP displacement and geodesic rotation angle. Solver
diagnostic residuals retain their separate frame/Euler semantics: flange in the
baseline, TCP in the study-only objective. Differences of vector norms across
layers are not additive. A large target-to-limiter gap can reflect accumulated
IK/executor lag because the limiter reanchors to FK(previous command).

Velocity/acceleration/jerk use actual nonuniform tick intervals; no smoothing is
applied to measurements. `handoff_100ms` includes the first active tick and plan
changes, `stable` is its complement including holds, `stable_success` excludes
IK holds, and `ik_hold` is separate. `matched_pre_hold_metrics.json` diagnoses
the common pre-failure prefix without using that diagnostic to reselect candidates.

Selection weights episodes and arms equally, does not pool complete and aborted
records into a success claim, and never uses candidate holdout results to tune.
Plate's latest session is entirely held out, but configuration and session are
confounded. Bottle has only one session and no independent holdout. No candidate
changes physical joint bounds, IK margins, J6/J7 checks or historical guard rates.
`tcp_objective.py` is a study-only code change, not an existing deployable switch.
