# Viewer body configuration

Tianji uses `tianji/viewer.yaml` and the generic `RobotView`.
`base.py`, `yam.py` and the old discovery exports remain only for the deferred YAM
migration. They are not used by the new Tianji Viewer, runtime or policy boundary.
Do not add new per-robot Python display adapters here.
