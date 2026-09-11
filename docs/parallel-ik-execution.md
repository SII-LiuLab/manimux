# Full IK with independent decoding processes

`configs/sapolicy/yam/infra/manimux-direct-async.yaml` preserves SAPolicy's
full-pose IK and direct joint execution. It enables `policy.action_decoding:
process`; the default remains `inline` for existing configurations.

Pair it with `configs/sapolicy/yam/server/teleop50-raw.yaml`: both now use a
50-step horizon at 30 Hz (about 1.67 seconds). The server returns all 50 model
steps and the adapter decodes all 50. Rolling replanning still removes expired
prefixes and replaces the remaining trajectory when a new plan arrives; it does
not guarantee open-loop execution of every row. Direct execution still has no
chunk-boundary blending, so increasing the horizon does not remove command jumps.

The adapter converts EEF targets into complete joint-position trajectories in
two spawned processes, one per arm. Each owns its adapter, MuJoCo data and IK
solver. Within an arm, targets retain their original order, previous-successful
solution seeds, iteration budget, tolerances, clipping and failed-step fallback.
The measured state at decode submission seeds both arms. Warmup initializes the
solvers before robot connection, without connecting hardware.

The control loop continues sampling and sending the existing trajectory while
decoding runs. It publishes a new plan only after both arms finish and their
contracts agree. If the previous trajectory expires, the existing runtime hold
path applies. No partial-arm plan is published. The inference-plus-decoding
pipeline permits only one in-flight request, so slow IK cannot build an unbounded
backlog. A dead or deadline-exceeding decoder faults the runtime and uses its
ordinary robot shutdown path.

This adopts the execution separation used by UMI's
[independent interpolation controller](https://github.com/real-stanford/universal_manipulation_interface/blob/main/umi/real_world/franka_interpolation_controller.py)
and checks actual receipt time before accepting actions, as illustrated by its
[environment scheduling](https://github.com/real-stanford/universal_manipulation_interface/blob/main/umi/real_world/bimanual_umi_env.py).
It is not UMI's robot controller or a replacement with differential IK.

## Time and pause semantics

Decoding completion and commit timestamps use the current clock. Plans beyond
their original final source timestamp are rejected, including the final-step
edge case where integer trimming alone would resurrect an expired last point.
Partially available plans use the existing joint timeline's trimming and
interpolation semantics; this does not introduce Cartesian interpolation or new
joint smoothing. It also does not implement calibrated camera/actuator latency
compensation. Faster preparation changes which source rows remain available,
so identical closed-loop hardware motion is not guaranteed by identical IK output.

Process decoding stops requesting inference while paused, clears its active
trajectory and invalidates outstanding responses. Resume needs a fresh observation;
home similarly invalidates results from before the home operation. Pausing does
not synchronously wait for or interrupt an individual numerical solve. Finish
closes the robot and decoder processes; their cleanup has bounded waits.

## Adapter contract

This mode currently requires the default `manimux` runtime strategy. An adapter
must declare `supports_context_only_decode = True`: its decode result must be
fully determined by raw action plus `ActionContext`, without observation-side
mutable caches. Optional `decode_partitions` and `decode_action_partition`
partition independent groups; optional `warmup_decode` initializes a child.
Other adapters retain inline decoding unless they explicitly support this contract.

SAPolicy observation preprocessing retains its own separate FK model. Decoding
never holds that model's lock. The parent sends only actions and a copied state
snapshot; robot drivers, images and robot control sockets are not passed to the
decoder. Startup has no model RPC or motor commands.

## Diagnostics and verification

Recorded plans include `raw_model_eef` in the original model frame and per-arm
`ik` diagnostics: seed, each solve's convergence and time, and failed-step count.
`decode_ms` is the maximum partition compute time; `decode_stage_ms` includes
communication and polling delay. `observation_to_commit_ms` includes the full
observation/model/decode/commit path. Neither compute metric is a control frequency
or a task-success measure.

```bash
envs/yam/.venv/bin/python scripts/validation/benchmark_parallel_ik.py \
  /path/to/saved/direct/diagnostics --output /tmp/parallel-ik
```

This consumes `offline-ik-profile.npz`, `offline-ik-profile.json` and `signals.npz`
from the existing recorded-observation diagnostic. It performs no robot I/O or
model-server calls. Every output joint and failed-solve status is compared against
serial full IK. IPC can outweigh the parallel speedup for already-cheap targets;
the most reliable benefit is removing full IK from the control thread.

Integration tests also inject 250 ms of CPU-bound decoding while a mock runtime
continues issuing commands, check expiry and pause invalidation, exercise failed
IK equivalence and reject incomplete/invalid dual-arm results. These tests do not
establish hard realtime scheduling or successful hardware grasping.
