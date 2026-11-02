# ARX X5 and PiPER SDK installation

Install the SDK needed by your controller in an isolated runtime environment.
The adapters use the official **pyAgxArm** package for standard PiPER and the
compiled **arx_x5_python** binding for X5 (2023). SDK source and native libraries
remain external dependencies; ManiMux packages the controller adapters and
public robot models.

For CAN bindings, calibration and lifecycle behavior, continue with the
[SDK adapter guide](can-arms.md). For visualization alone, the
[ALOHA](aloha.md) and [PiPER](piper.md) assets need only core ManiMux dependencies.

## Runtime environment

From the repository root, create a local environment and install ManiMux:

```bash
uv venv envs/aloha/.venv --python 3.12
uv pip install --python envs/aloha/.venv/bin/python -e .
```

`envs/aloha/` is an ignored workspace convention, not a device or assembly
selection. A different environment path is fine; use its interpreter consistently.
Install policy transport or camera extras only when the selected deployment needs
them. Model inference environments stay separate from the robot runtime.

## Standard PiPER

Install [agilexrobotics/pyAgxArm](https://github.com/agilexrobotics/pyAgxArm)
at revision `841a625f5f4920e776f20b934eb13048b747e6d0` (LGPL-3.0-only):

```bash
uv pip install --python envs/aloha/.venv/bin/python \
  'pyAgxArm @ git+https://github.com/agilexrobotics/pyAgxArm.git@841a625f5f4920e776f20b934eb13048b747e6d0'
```

The controller selects `ArmModel.PIPER` and one explicit firmware value:
`default`, `v183`, `v188` or `v189`. Match the firmware installed on the arm;
the selector does not change the offline URDF's geometry. PiPER H, L and X
are not covered by this adapter.

Pure configuration construction, class inspection and MDH FK do not need CAN.
The installation probe performs only these operations. Communication drivers
are constructed later by the controller's `connect()` operation.

## ARX X5 (2023)

Use [ARXroboticsX/X5](https://github.com/ARXroboticsX/X5) at revision
`9a255e38156e3f5a263f341df209ab028d49dad9` (BSD-3-Clause). The adapter uses
model type 0, which denotes X5 (2023); type 2 denotes a different model.
The documented `set_catch` range is 0..5 in vendor units. Encoder feedback
and command endpoints require separate station calibration.

Keep the vendor source under the ignored SDK workspace:

```bash
git clone https://github.com/ARXroboticsX/X5.git envs/aloha/sdk/ARX_X5
git -C envs/aloha/sdk/ARX_X5 checkout 9a255e38156e3f5a263f341df209ab028d49dad9
```

The pinned native libraries require Linux, a compiler, CMake, pybind11,
`libkdl_parser.so`, `liborocos-kdl.so.1.5` and their runtime dependencies.
The commands below cover Linux x86-64 with these prerequisites already present;
consult upstream for other architectures. Compile the extensions for the exact
Python interpreter that will import them.

```bash
uv pip install --python envs/aloha/.venv/bin/python 'pybind11==3.1.0'
cmake -S envs/aloha/sdk/ARX_X5/py/arx_x5_python/bimanual \
  -B envs/aloha/sdk/ARX_X5/py/arx_x5_python/bimanual/build \
  -Dpybind11_DIR="$(envs/aloha/.venv/bin/python -m pybind11 --cmakedir)" \
  -DPYTHON_EXECUTABLE="$PWD/envs/aloha/.venv/bin/python" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build envs/aloha/sdk/ARX_X5/py/arx_x5_python/bimanual/build -j2
cmake --install envs/aloha/sdk/ARX_X5/py/arx_x5_python/bimanual/build
export PYTHONPATH="$PWD/envs/aloha/sdk/ARX_X5/py/arx_x5_python/bimanual/api/arx_x5_python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$PWD/envs/aloha/sdk/ARX_X5/py/arx_x5_python/bimanual/api:$PWD/envs/aloha/sdk/ARX_X5/py/arx_x5_python/bimanual/api/arx_x5_src${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Upstream CMake installs into the source tree's `bimanual/api/`. The Python path
above contains both `arx_x5_python` and `kinematic_solver` extensions. Use `ldd`
on those extensions to find missing libraries. If dependencies are extracted
to a local prefix, prepend that prefix's library directory to `LD_LIBRARY_PATH`
as well. Retain these exports in your local environment activation configuration.

Installation verification must not instantiate `InterfacesPy` or `SingleArm`:
they acquire hardware resources, and `SingleArm` starts active control.
The adapter owns the native session in a separate process because the binding
has no explicit shutdown operation. Do not run upstream CAN or machine setup
scripts as part of an offline installation check.

## Offline installation check

From the repository root, select the SDKs installed in this environment:

```bash
# PiPER only; ARX native paths are not required.
envs/aloha/.venv/bin/python scripts/setup/check_can_arm_sdks.py --sdk piper

# Both official SDKs, after configuring ARX library paths.
envs/aloha/.venv/bin/python scripts/setup/check_can_arm_sdks.py --sdk both
```

Use `--sdk arx` for X5 alone. Missing imports, required methods or FK mismatches
fail the check; the probe does not choose another SDK. The check loads all three
public assemblies, inspects the selected official binding APIs, and verifies
PiPER MDH FK or X5 FK against the packaged model. It never creates a device
session, enables motors or sends commands.

The X5 solver reports translation relative to the zero-pose endpoint rather
than the URDF's `base_link` origin. The probe subtracts that endpoint's URDF
translation before comparing FK; rotations keep the URDF convention. The
ManiMux kinematics use the URDF arm-base frame directly.

The probe prints a JSON report. Native ARX libraries may also print startup
messages, so use `--output /path/to/report.json` for a separate JSON file.
Keep generated reports outside commits. A passing check establishes imports,
API availability and offline geometry, not CAN readiness, physical calibration,
device feedback freshness, stopping behavior or task success.
