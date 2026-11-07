# Flexiv dual-arm mounting stand

`dual_arm_stand_20260523.stl` is the unmodified operator-supplied CAD export,
received on 2026-11-05 with the corresponding dual-arm stand drawing. Its SHA-256
is `7727b67d73934e91702b7b63b502045e828937cde7b328f7ee3982c9c6d78125`.
It is a display asset, not part of arm kinematics or hardware configuration.

## Local-only CAD and license status

As of 2026-11-07, no license, copyright-holder attribution, or verifiable public
source has been established for this exact STL. Its binary header names the
dual-arm column base and the 20260523 design date; it contains no license notice.
The operator-supplied provenance above is not redistribution permission. Do not
apply ManiMux's MIT license or an unrelated Flexiv repository's license to it.
Keep the STL local and Git-ignored until the rights holder supplies permission
covering this asset. Compression or format conversion does not establish permission.

The public Flexiv presets use schematic boxes and do not reference this CAD.
`stand.urdf` is a hand-authored optional wrapper; its mesh is deliberately absent
from a fresh checkout. A private RoboGUI YAML selected with `--config` can reference
this wrapper after the operator installs the STL beside it. Place private presets
in `manimux/configs/robogui/` with names that do not start with `example-`; that
directory already ignores station files. Resolve model and mesh paths relative to
the private YAML. The prepared local copies are `flexiv-local-stand.yaml` and
`flexiv-omnigripper-local-stand.yaml`; these copies are not shipped.

## Local CAD placement

The STL uses meters, with +Z up and bounds approximately
`[0, 0, 0.0000193864]` to `[0.86, 0.86, 1.395]`. Do not apply a millimeter scale.
`stand.urdf` shifts XY by `[-0.43, -0.43]` to center the base plate.

The two front mounting faces point toward -Y. Their source centers are
`[0.18, 0.315, 1.305]` and `[0.68, 0.315, 1.305]` m. In the centered stand frame
they are `[-0.25, -0.115, 1.305]` and `[0.25, -0.115, 1.305]`, giving the drawing's
500 mm center spacing. The mounting plates have an outer diameter of 180 mm;
the drawing's 150 mm dimension describes the supporting tube.

The private CAD presets place the stand at `[0, 0.8, 0]`, so the arm display origins are
`[+/-0.25, 0.685, 1.305]` m. Each preset retains its existing arm identities,
orientations and initial joint values. These display transforms do not calibrate
physical controller frames.

The table center is `[0, -0.4, 0.68]` m with dimensions `[1.6, 0.9, 0.04]` m.
Its edge nearest the robot is at Y=0.05, giving the operator-measured 0.75 m
horizontal distance from the column center at Y=0.8 to that edge.
