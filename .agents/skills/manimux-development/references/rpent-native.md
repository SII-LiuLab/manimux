# RPent native bridge migration

The separate native environment launcher has been retired. Use the
[unified RPent integration](rpent.md): one recipe, Toolkit and ManiMux runtime
for motion tools and the configured VLA. The public commands are in
[the user guide](../../../../guideline.html#agent-integrations).

The remaining native Env classes are protocol regression fixtures, not a second
supported deployment mode. Do not launch RPent's native hardware backend alongside
ManiMux. Existing station calibrations can be reused by the new motion profile;
verify group base frames, tool conventions and checkpoint semantics explicitly.
