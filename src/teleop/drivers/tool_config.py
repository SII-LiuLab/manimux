#!/usr/bin/env python3
"""Typed, validated YAML config for Marvin_Robot.set_tool() -- the mounted
end-effector's mass/center-of-mass/inertia, fed to the controller's
torque-mode gravity feedforward so it accounts for tools beyond the arm's
own Mass/MCP/I model (robot.ini). See docs/scripts.md's
gravity-compensation section for why this exists.

Pattern mirrors algos.solver_config: one pydantic model + one yaml per
tool under configs/tool/<name>.yaml, e.g. configs/tool/omnigripper.yaml
for drivers/gripper.py's OmniGripper (DM4310).

Per-arm, not one set of numbers shared across A/B: a gripper's mount
isn't guaranteed symmetric between the two arms (mirrored center-of-mass
offset is plausible but unconfirmed), so each arm needs its own measured
or identified entry -- see configs/tool/omnigripper.yaml, which currently
has both arms identified (B from vendor reference data, A from a live
collection pass on this project's own robot -- see that file's per-entry
`source:` fields).
"""
import os

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ToolArmConfig(BaseModel):
    """Marvin_Robot.set_tool()'s kineParams/dynamicParams for one mounted
    tool, on one specific arm. mass_kg/com_mm/source have no default -- a
    guessed value feeds the wrong gravity torque into a live arm, worse
    than not registering the tool at all, so an unfilled arm entry fails
    to load instead of silently defaulting to 0.
    """

    model_config = ConfigDict(extra='forbid')

    mass_kg: float = Field(..., ge=0, description='tool mass, kg')
    com_mm: list[float] = Field(
        ..., min_length=3, max_length=3,
        description='[x,y,z] center of mass offset from the flange, mm')
    inertia: list[float] = Field(
        [0.0] * 6, min_length=6, max_length=6,
        description='[ixx,ixy,ixz,iyy,iyz,izz] about the center of mass. '
                    'Optional -- the vendor demo (showcase_set_save_tool.py) '
                    'leaves this zeroed too.')
    kine_offset: list[float] = Field(
        [0.0] * 6, min_length=6, max_length=6,
        description='[x,y,z,a,b,c] tool frame offset from the flange, '
                    'mm/deg -- the fingertip midpoint for a gripper. Only '
                    'affects Cartesian/TCP targeting, not gravity '
                    'feedforward. Also read by algos.tool_frame.ToolFrame so '
                    'UMI replay/filtering retargets at the fingertip; '
                    'all-zero means NOT MEASURED there, not "no offset".')
    kine_offset_source: str | None = Field(
        None,
        description='where kine_offset came from. Required as soon as it is '
                    'nonzero -- same rule as `source`, and for the same '
                    'reason: a tool frame nobody can trace is a tool frame '
                    'nobody can check. scripts/tip_calib.py prints a line '
                    'meant to be pasted here.')
    kine_offset_verification: str | None = Field(
        None,
        description='how kine_offset was obtained, so consumers can report '
                    'what is still unchecked instead of hard-coding a '
                    'sentence that outlives the value. "physical" = measured '
                    'on the real gripper (tip_calib for the translation plus '
                    'a fixture that constrains orientation); "cad" = derived '
                    'from the assembly geometry and cross-checked numerically '
                    'but never confirmed on the mount. Absent means unknown '
                    'provenance, which deployment reports as unverified. '
                    'Read by deploy.tianji.teleop_provider.load_tool_transforms.')
    source: str = Field(
        ..., min_length=1,
        description='where these numbers came from (measurement method/'
                    'date, or identification dataset/date) -- required so '
                    'provenance is never a guess months later.')

    @field_validator('kine_offset_verification')
    @classmethod
    def _known_verification(cls, value):
        if value is None:
            return value
        allowed = ('physical', 'cad')
        if str(value).strip().lower() not in allowed:
            raise ValueError(
                'kine_offset_verification 只能是 %s 之一（或整条省略表示来源'
                '未知）；收到 %r。' % ('/'.join(allowed), value))
        return str(value).strip().lower()

    @model_validator(mode='after')
    def _kine_offset_needs_provenance(self):
        if any(abs(v) > 0 for v in self.kine_offset) \
                and not (self.kine_offset_source or '').strip():
            raise ValueError(
                'kine_offset 非零但没有 kine_offset_source —— 工具坐标系'
                '偏移必须能追溯来源（scripts/tip_calib.py 会打印可直接粘贴'
                '的一行）。')
        return self

    def dynamic_params(self):
        """Marvin_Robot.set_tool()'s dynamicParams argument."""
        return [self.mass_kg] + list(self.com_mm) + list(self.inertia)

    def kine_params(self):
        """Marvin_Robot.set_tool()'s kineParams argument."""
        return list(self.kine_offset)


class ToolConfig(BaseModel):
    """One mounted tool's per-arm entries -- configs/tool/<name>.yaml."""

    model_config = ConfigDict(extra='forbid')

    name: str = Field(..., description='label, e.g. "omnigripper"')
    arms: dict[str, ToolArmConfig] = Field(
        default_factory=dict,
        description='arm id ("A"/"B") -> that arm\'s ToolArmConfig. Arms '
                    'with no entry here are not yet measured/identified.')


def load_tool_config(name, arm, repo_root):
    """Load+validate configs/tool/<name>.yaml and pick out `arm`'s entry.

    :return: (ToolArmConfig, resolved cfg_path)
    :raises ValueError: file missing, fails validation, or has no entry
        for `arm` -- message is meant to be printed directly, not parsed.
    """
    cfg_path = os.path.join(repo_root, 'configs', 'tool', '%s.yaml' % name)
    try:
        with open(cfg_path) as fh:
            raw = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        raise ValueError('工具参数配置文件不存在：%s' % cfg_path)
    try:
        cfg = ToolConfig.model_validate(raw)
    except Exception as e:      # pydantic.ValidationError et al.
        raise ValueError(
            '工具参数配置校验失败（%s）：\n%s\n'
            '提示：mass_kg/com_mm/source 需要先测量（或用 SDK 自带的 '
            'showcase_identy_tool_dynamic_*.py 辨识）再填入 yaml，猜测值比'
            '不填更危险 —— 会把错误的重力矩喂给通电的机械臂。'
            % (cfg_path, e))
    if arm not in cfg.arms:
        raise ValueError(
            '工具 %s（%s）没有臂 %s 的参数条目——还没测量/辨识过，不能瞎填。'
            '已有条目：%s'
            % (name, cfg_path, arm, sorted(cfg.arms) or '（无）'))
    return cfg.arms[arm], cfg_path
