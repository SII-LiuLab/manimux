# Config lists

A list selects reusable runtime configurations, a formal trial plan and one recording
root. It does not override model or inference parameters. Concrete variants remain
in `../experiments/`; any variant also works directly with `run --config` for trials.

```yaml
id: my-study
recording:
  root: /data/robot-experiments
experiments:
  - id: coin_rtc
    config: ../experiments/my-policy/coin/rtc32_beta10.yaml
    label: RTC beta 10
    trials:
      layout_ids: ['01', '02']
      repeats: 3
```

The example requires a matching concrete variant and task definitions. Relative
`config` and `recording.root` paths resolve beside this file. Optional `local` selects
a station; command-line `--local` takes precedence. Each entry can declare `history`
paths relative to the recording root; only matching recorded conditions count.

Formal output: `<recording.root>/<id>/<task>/<variant>/session-…/rollout-…`.
Reloading the same list resumes that identity and progress. Use the same
`--config-list` file for camera, generic policy and runtime. When service recipes
differ, select an entry explicitly with `--experiment-id`; switching does not
restart services automatically. `inspect --config-list` is read-only unless `--index`
is supplied, which refreshes the derived inventory.

Published lists: [`example-synthetic`](example-synthetic.yaml) (no hardware; its `local`
selects the empty `local/synthetic.yaml` station), [`example-yam-pi05`](example-yam-pi05.yaml),
[`example-yam-dream1`](example-yam-dream1.yaml) and
[`example-tianji-pi05`](example-tianji-pi05.yaml), and
[`example-flexiv-pi05-models`](example-flexiv-pi05-models.yaml).

Optional `policy_switching: on_demand` allows different XPolicyLab model recipes
with identical robot and camera services. The generic policy launcher owns one
model subprocess and loads the selected recipe on the next client connection.
Use `manimux.servers.policy --config-list <list>` without `--experiment-id` for
this mode. Model identity/capability checks still apply; active connections block
cross-model replacement. Idle RoboGUI selection rebuilds the adapter/runtime at
Prepare. Sampler changes for the same recipe retain loaded weights. Each model's
dependencies must be installed in the launcher's environment; no hardware is
owned by the model catalog.

Only the published `example-*` lists and this README are allowlisted in Git. Other local
lists remain ignored unless deliberately published together with their variants.
The former `collections/` directory is now `lists/`; YAML contents and recorded data
are unchanged. Internal `manimux.collections` and saved `collection` metadata names
remain for historical compatibility.

See the [config map](../README.md) for ownership and the
[user guide](../../../guideline.html#experiment-lists) for complete launch workflows.
