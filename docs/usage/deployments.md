# Choose a policy deployment

Choose the framework, checkpoint, action representation and embodiment together.
A checkpoint is not portable to another robot just because its output width matches.
Use the matching experiment on the runtime and model server, and bind local paths
in your station file where that launcher supports it.

## Integration counts

The README counts distinct policy implementations selected by checked-in deployment
recipes, not checkpoints, YAML files or every model in an upstream framework.

| Scope | Count | Included implementations |
| --- | --- | --- |
| Policies with robot deployment recipes | 8 | Pi05, DP, SAPolicy, GR00T N1.7, LingBot-VLA2, Xiaomi XR-1, UMI DP, OpenWAM |
| Policies with offline recipes | 5 | Isaac 0.5; StarVLA QwenOFT, QwenPI-v3, QwenGR00T, QwenFast |
| Inference modes | 7 | Serial, asynchronous chunking, RTC, ACT temporal ensembling, AAC, PAINT, AutoHorizon |
| Hardware assembly integrations | 4 | YAM, Tianji–TacCap; experimental ARX X5 (2023), standard PiPER |

The **13 policy integrations** are selected through `policy_name` for XPolicyLab
and `framework` for StarVLA under `manimux/configs/policy/`. QwenGR00T and GR00T N1.7
are separate implementations; additional tasks and checkpoint variants are not counted.
Cosmos3 has an additional offline server guide but no checked-in ManiMux policy recipe,
so it is outside this count. A framework's other models can be integrated through the
same client protocol; they are not automatically counted as ManiMux deployments.

The **8 inference modes** use seven registered strategies: Serial and asynchronous
chunking both select `algorithm: manimux`, with different `inference_schedule` settings.
This count excludes executors (Direct, Smooth, MPC), blending parameters and history wrappers.
See [scheduling](../advanced/inference.md) for behavior and model-side sampler requirements.

The **4 hardware integrations** count assemblies with selectable controllers:
YAM, Tianji–TacCap, ARX X5 (2023) and standard PiPER. The latter two are
experimental: they have offline interface checks and station templates, but no
validated physical deployment or checked-in real-robot policy experiment.
X5 device feedback freshness remains an explicit limitation of the official binding.
This count does not imply hardware validation or support for every model/robot pairing.

## RoboGUI previews and hardware scope

| Embodiment | RoboGUI preset | Hardware scope |
| --- | --- | --- |
| YAM | `--robot yam` | Assembly and policy deployment recipes |
| Tianji–TacCap | `--robot tianji` (local assets required) | Assembly and policy deployment recipes |
| Standard PiPER | `--robot piper` | Experimental SDK controller; physical validation pending |
| ALOHA-AgileX | `--robot aloha` | Offline follower-arm assets only; no hardware controller |
| ARX X5 (2023) | No bundled mesh preset | Kinematic model and experimental SDK controller; physical validation pending |

Use `--demo` with PiPER or ALOHA to animate synthetic joint trajectories without
hardware. Both include gripper visuals. [Try the presets](getting-started.md#hardware-free-start).
ALOHA's RoboTwin geometry is separate from the physical X5 and PiPER models;
its offline preset does not add another hardware integration. LIBERO policy contracts
also do not add a hardware driver. See the [SDK adapter guide](can-arms.md) for
feedback, calibration and lifecycle limitations.

## Model and robot recipes

| Deployment | Guide |
| --- | --- |
| Pi05 / OpenPI on YAM | [Joint](../deployment/pi05-yam.md) · [EEF](../deployment/pi05-yam-eef.md) |
| DP on YAM | [Absolute EEF](../deployment/dp-yam.md) |
| SAPolicy on YAM | [Joint/EEF and camera mapping](../deployment/sapolicy-yam.md) |
| GR00T N1.7 on YAM | [Deployment and RTC](../deployment/gr00t-yam.md) |
| LingBot-VLA2 on YAM | [Action semantics and deployment](../deployment/lingbot-vla2-yam.md) |
| Xiaomi XR-1 | [YAM](../deployment/xiaomi-xr1-yam.md) · [Tianji–TacCap](../deployment/xiaomi-xr1-tianji-taccap.md) |
| UMI DP on Tianji–TacCap | [Artifact binding and services](../deployment/umi-dp-tianji-taccap.md) |
| OpenWAM on YAM | [Checkpoint contract](../deployment/openwam-yam.md) |
| StarVLA | [Offline deployment](../deployment/starvla-offline.md) |
| Cosmos3 / Isaac 0.5 | [Cosmos3](../deployment/cosmos3-offline.md) · [Isaac 0.5](../deployment/isaac05-offline.md) |

Model dependencies run in their own environments. The hardware runtime does not
need torch or JAX merely to talk to a policy server. Follow the selected framework's
installation instructions and [environment guidance](environments.md).

## What support means

XPolicyLab and StarVLA are peer policy frameworks. Model implementations stay with
the owning framework; ManiMux clients handle transport and capability/identity
checks. [XPolicyLab bridge](../deployment/xpolicylab.md).

YAM and Tianji–TacCap have assembly integrations with different SDK and asset
requirements. Direct, Smooth and MPC are executors, independent of inference
strategies. The [integration protocols](../development/README.md) describe these boundaries.

A recipe, successful offline forward, ready server and successful physical trial
are different evidence levels. Each detailed guide records its scope; historical
measurements describe their recorded revision rather than certifying this checkout.
An unbound template or unavailable checkpoint is not a working deployment.
