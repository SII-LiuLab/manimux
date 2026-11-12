# Inference Method Reproduction Records

This directory stores algorithm-reproduction audit records, with a different role from model/embodiment runbooks:

- **Method documents** explain the official method, our integration, parameters, commands, and evidence.
- **Model runbooks** cover model environments, checkpoints, input/output contracts, and default service startup.

Before hardware acceptance, each method needs its own document recording:

1. Paper, official repository, pinned commit, and source files inspected.
2. Official model, action space, horizon, control frequency, and evaluation environment.
3. Equations, default parameters, and actual execution order in code.
4. ManiMux/XPolicy responsibilities and every adaptation.
5. Exact matches, equivalent adaptations, explicit differences, and unverified items.
6. Configurations, weights, stats, data sources, and their bindings.
7. Separate commands and evidence for offline unit tests, actual forward inference, simulation, and hardware.
8. Known risks, failure criteria, rollback entry points, and reviewer checklist.

## Index

| Status | Method | Method document and commands | Current gate |
|---|---|---|---|
| ✅ | Default ManiMux | [`../architecture.md`](../../../manimux-development/references/architecture.md) | Current dual-YAM model integrations exercised |
| ✅ | RTC (inference-time Pi-guided) | [`../xpolicylab-runbook.md#rtc-contract`](../deployment/xpolicylab.md#rtc-contract) | Pi05/YAM hardware complete; measured delay-aware execution |
| ✅ | Adaptive Action Chunking | [`aac.md`](aac.md) · [`aac-pi05.md`](aac-pi05.md) | Pi05 GPU/YAM hardware complete; functional but visibly laggy |
| ✅ | ACT Temporal Ensembling | [`../act-temporal-ensemble.md`](../inference.md) | Pi05/YAM hardware complete; operator observed smooth execution |
| ✅ | PAINT | [`paint-pi05.md`](paint-pi05.md) | Pi05 GPU/YAM hardware complete; operator observed very good continuity |
| ✅ | AutoHorizon | [`autohorizon-pi05.md`](autohorizon-pi05.md) | Pi05/YAM hardware exercised; synchronous inference holds were visibly stop-and-go |

| Offline | BID backward-only | [`bid-backward.md`](bid-backward.md) | Formula, transport and runtime checks; real checkpoint and robot validation pending |

New methods cannot be marked complete only in a README table; every status must link back to evidence here.
