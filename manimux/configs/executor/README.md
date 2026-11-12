# Executor presets

`executor.type` selects `direct`, `smooth` or `mpc`. Other fields contain the
existing smoothing, motion-limit and command-limit settings. Shared collection
and deployment control profiles retain their merge rules.

Executor selection is independent of `inference.algorithm`. Changing RTC to AAC
does not also change the filter or arm/gripper limits. Joint angles use radians,
timing uses seconds, and gripper values retain the embodiment's convention.
