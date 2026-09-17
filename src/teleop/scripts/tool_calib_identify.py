#!/usr/bin/env python3
"""Offline tool-parameter identification -- step 3/3 of the tool-parameter
calibration pipeline described in docs/scripts.md's gravity-compensation
section. Steps 1+2 (no-load / load PVT data collection, both homed to
all-zero joints first) are scripts/tool_calib_collect.py; this script only
diffs the two datasets that script already produced -- no robot connection.

Wraps SDK_PYTHON.fx_kine.Marvin_Kine.identify_tool_dyn(robot_type, ipath)
(DEMO_PYTHON/showcase_identy_tool_dynamic_CCS_B.py's run_offline()), which
expects <ipath>/{LoadData.csv, NoLoadData.csv,
CfgFile/LoadIdenCfg_Marvin_CCS.txt} -- scripts/tool_calib_collect.py's
--out-dir already lays data out this way.

Prints the raw [m, mcp_x, mcp_y, mcp_z, ixx, ixy, ixz, iyy, iyz, izz]
result and a ready-to-paste configs/tool/<name>.yaml block
(drivers.tool_config.ToolArmConfig's shape). Deliberately does NOT write
the yaml file itself: that file is hand-curated with a prose `source:`
note per entry (see configs/tool/omnigripper.yaml) and this project treats
tool mass/COM as too safety-critical to auto-merge into a comment-bearing
file via a plain YAML dump, which would silently strip every comment.
Paste the printed block in by hand and fill in `source:` with the actual
collection date/notes.

Usage:
    python3 scripts/tool_calib_identify.py --dir data/tool_calib/omnigripper/A
    python3 scripts/tool_calib_identify.py --dir data/tool_calib/omnigripper/A --arm A --tool omnigripper
"""
import argparse
import os
import sys

_SDK = os.environ.get('MARVIN_SDK', '/home/jw/Downloads/TJ_FX_ROBOT_CONTRL_SDK')
if _SDK not in sys.path:
    sys.path.insert(0, _SDK)
from SDK_PYTHON.fx_kine import Marvin_Kine   # noqa: E402

# robot_type=1 (CCS) is this project's only machine type -- config.KINE_CFG
# points at ccs_m6_40.MvKDCfg. robot_type=2 (SRS) is a different machine,
# out of scope here (see docs/scripts.md).
ROBOT_TYPE = 1

_REQUIRED = ('LoadData.csv', 'NoLoadData.csv',
            os.path.join('CfgFile', 'LoadIdenCfg_Marvin_CCS.txt'))

_FIELD_NAMES = ('mass_kg', 'com_x_mm', 'com_y_mm', 'com_z_mm',
               'ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz')


def identify(ipath):
    """:return: [m, mcp_x, mcp_y, mcp_z, ixx, ixy, ixz, iyy, iyz, izz]
    :raises ValueError: missing input file, or identify_tool_dyn itself
        returned an error string (already human-readable, see
        SDK_PYTHON/fx_kine.py's identify_tool_dyn docstring for the codes)
    """
    for name in _REQUIRED:
        p = os.path.join(ipath, name)
        if not os.path.exists(p):
            raise ValueError('缺少 %s —— 先跑 scripts/tool_calib_collect.py' % p)
    result = Marvin_Kine().identify_tool_dyn(robot_type=ROBOT_TYPE, ipath=ipath)
    if isinstance(result, str):
        raise ValueError('辨识失败：%s' % result)
    return result


def format_yaml_block(arm, values, tool, ipath):
    mass, com, inertia = values[0], values[1:4], values[4:10]
    return (
        '  %s:\n'
        '    mass_kg: %s\n'
        '    com_mm: [%s]\n'
        '    inertia: [%s]\n'
        '    kine_offset: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]\n'
        '    source: >-\n'
        '      Marvin_Kine.identify_tool_dyn(robot_type=1, ipath=...) against\n'
        '      %s\n'
        '      (scripts/tool_calib_collect.py --arm %s --tool %s) -- FILL IN\n'
        '      collection date / notes before committing.\n'
        % (arm, mass, ', '.join(str(v) for v in com),
           ', '.join(str(v) for v in inertia), ipath, arm, tool))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True,
                    help='scripts/tool_calib_collect.py 的 --out-dir 输出目录 '
                         '(含 LoadData.csv/NoLoadData.csv/CfgFile/)')
    ap.add_argument('--arm', choices=['A', 'B'], default='A',
                    help='粘贴块里用的 yaml key，仅影响打印格式，不影响辨识本身')
    ap.add_argument('--tool', default='<tool>',
                    help='粘贴块里 source 注释用的工具名，仅影响打印格式')
    args = ap.parse_args()

    ipath = os.path.abspath(args.dir)
    print('辨识数据目录：%s' % ipath)
    try:
        values = identify(ipath)
    except ValueError as e:
        raise SystemExit('❌ %s' % e)

    print('\n辨识结果：')
    for n, v in zip(_FIELD_NAMES, values):
        print('  %-10s %s' % (n, v))

    print('\n粘贴到 configs/tool/%s.yaml 的 arms.%s（记得手填 source 里的日期/备注）：\n'
         % (args.tool, args.arm))
    print(format_yaml_block(args.arm, values, args.tool, ipath))


if __name__ == '__main__':
    main()
