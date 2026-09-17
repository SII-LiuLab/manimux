#!/usr/bin/env python3
"""Typed, validated YAML config for pluggable IK/control algorithms.

Pattern for adding a new algorithm (impedance control, WBC, ...):
    1. Define a pydantic model here, one per algorithm, with field
       constraints (ge=/gt=/description=) doing the actual parameter
       cleaning -- range checks, type coercion, unknown-key rejection.
    2. Add a matching configs/solver/<name>.yaml with tuned defaults.
    3. Load it with load_solver_config(YourConfig, path).
No other plumbing changes -- run_teleop.py/core/arm_channel.py only need
to know "which model class for which --solver value."

The existing analytic solver (ik_solver.ArmIK) does NOT use this -- its
tunables are hardware-validated production values that already live in
config.py; this module is only for solvers still being validated
(diff_ik.py today, impedance/WBC later).
"""
import os
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class DiffIKConfig(BaseModel):
    """algos.diff_ik.DiffIKSolver's tunables. See docs/algos.md#diff_ikpy
    for what each one does and why the defaults are what they are."""

    # extra='forbid': an unknown key (e.g. a typo'd 'mu_nullpace') raises
    # at load time instead of being silently ignored -- this is the
    # "parameter cleaning" a plain yaml.safe_load() doesn't give you.
    model_config = ConfigDict(extra='forbid')

    w_pos: float = Field(1.0, gt=0,
                         description='task weight on position error (mm)')
    w_rot: float = Field(1.0, gt=0,
                         description='task weight on orientation error (deg)')
    lam: float = Field(1e-3, ge=0,
                       description='minimum-norm regularization')
    limit_margin_deg: Optional[float] = Field(
        None, gt=0, description='None -> falls back to config.LIMIT_MARGIN_DEG')
    check_j67: bool = Field(
        True, description='J6/J7 interference: hard QP constraint + '
                          'post-hoc check. See docs/algos.md#diff_ikpy '
                          'for why this is a constraint, not a cost term.')
    j67_margin_deg: Optional[float] = Field(
        None, gt=0, description='None -> falls back to limit_margin_deg')
    mu_nullspace: float = Field(
        0.0, ge=0, description='null-space joint-limit-avoidance cost '
                               'weight, 0 = off (default off; leverage is '
                               'configuration-dependent, see docs/algos.md)')
    nullspace_activation_deg: float = Field(25.0, gt=0)


class ImpedanceConfig(BaseModel):
    """drivers.arm_driver.ArmDriver's ARM_STATE=3 (torque) tunables --
    impedance type + joint/Cartesian K/D. See docs/algos.md#impedanceconfig.

    No upper bound on K fields: the vendor SDK doc states a translation-
    stiffness range of 0~1200 for Cartesian impedance, but the GUI-
    validated default (cart_k[0:3]=3000) already exceeds it and is known
    to run correctly -- the doc's range is not trustworthy, so it isn't
    encoded as a validation ceiling here.
    """

    model_config = ConfigDict(extra='forbid')

    type: int = Field(
        1, ge=1, le=2,
        description='1=joint impedance, 2=cartesian impedance. Vendor '
                    '3=force control is not implemented by ArmDriver '
                    '(different call sequence) -- rejected here rather '
                    'than accepted and silently mis-driven.')
    joint_k: list[float] = Field(
        ..., min_length=7, max_length=7,
        description='per-joint stiffness, N*m/deg, J1..J7, each >=0')
    joint_d: list[float] = Field(
        ..., min_length=7, max_length=7,
        description='per-joint damping, J1..J7, each >=0')
    cart_k: list[float] = Field(
        ..., min_length=7, max_length=7,
        description='[transX,transY,transZ,rotX,rotY,rotZ,nullspace] '
                    'stiffness, each >=0')
    cart_d: list[float] = Field(
        ..., min_length=7, max_length=7,
        description='[transX,transY,transZ,rotX,rotY,rotZ,nullspace] '
                    'damping, each >=0')
    rot_type: int = Field(
        2, ge=1, le=3,
        description='Cartesian impedance only (type=2): passed as fcType '
                    "to the vendor SDK's Marvin_Robot.set_EefCart_"
                    'control_params (OnSetEefRot_A/B) -- a call our '
                    'original code never made at all, which is the likely '
                    'root cause of "other joints doing something '
                    'unexplained" under end-effector force (see '
                    'docs/algos.md#impedanceconfig). 1=user-defined '
                    'direction (see cart_ctrl_para), 2=system '
                    'auto-calculated, 3=used together with '
                    'set_force_control_params (not wired here). Default 2 '
                    'is the only value taken from a real, runnable vendor '
                    'demo (DEMO_C++/showcase_eef_cart_impedance.cpp) '
                    'rather than inferred from a docstring -- the SDK\'s '
                    'own docstrings for this parameter disagree with each '
                    'other (Concise_Marvin_Robot.set_imp_cart_state calls '
                    'it "rot_type" with a 0/1/2 numbering that includes an '
                    '"undefined" option; Marvin_Robot.set_EefCart_control_'
                    'params -- what ArmDriver actually calls -- calls it '
                    '"fcType" with a 1/2/3 numbering that doesn\'t. Not '
                    'trustworthy without independent verification. Unused '
                    'when type=1 (joint impedance).')
    cart_ctrl_para: list[float] = Field(
        [0.0] * 7, min_length=7, max_length=7,
        description='Cartesian impedance only (type=2): meaning depends '
                    'on rot_type -- see that field. [0]*7 (the default) '
                    'is what the vendor demo uses alongside rot_type=2.')

    @field_validator('joint_k', 'joint_d', 'cart_k', 'cart_d')
    @classmethod
    def _all_non_negative(cls, v):
        if any(x < 0 for x in v):
            raise ValueError('K/D entries must be >= 0')
        return v


def load_solver_config(model_cls, path):
    """Read+validate a YAML file against a pydantic model. Raises
    pydantic.ValidationError with a specific field/reason on a type
    mismatch, an out-of-range value, or an unknown key -- not a silent
    pass-through of whatever the file happened to contain."""
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    return model_cls.model_validate(raw)


# --impedance-type -> ImpedanceConfig.type, and the default
# configs/solver/impedance_<name>.yaml filename suffix. Shared by every
# entry point that offers --impedance-type (run_teleop.py, goto_joints.py,
# set_state.py) so the CLI-value/yaml-type mapping can't drift between them.
IMPEDANCE_TYPE_VALUE = {'joint': 1, 'cartesian': 2}


def load_impedance_config(impedance_type, path, repo_root):
    """Load+validate an ImpedanceConfig for --impedance-type, defaulting
    `path` to configs/solver/impedance_<impedance_type>.yaml under
    `repo_root` when not given. Refuses a mismatch between the CLI
    selection and the yaml's own `type` field (e.g. --impedance-type
    cartesian pointed via --impedance-config at a joint-impedance file) --
    see docs/core.md's `--control-mode impedance` section for why that's a
    hard refusal, not a warning.

    :return: (ImpedanceConfig, resolved cfg_path)
    :raises ValueError: file missing, fails validation, or type mismatch --
        message is meant to be printed directly, not parsed.
    """
    cfg_path = path or os.path.join(repo_root, 'configs', 'solver',
                                    'impedance_%s.yaml' % impedance_type)
    try:
        cfg = load_solver_config(ImpedanceConfig, cfg_path)
    except FileNotFoundError:
        raise ValueError('阻抗配置文件不存在：%s' % cfg_path)
    except Exception as e:   # pydantic.ValidationError et al.
        raise ValueError('阻抗配置校验失败（%s）：\n%s' % (cfg_path, e))
    expect_type = IMPEDANCE_TYPE_VALUE[impedance_type]
    if cfg.type != expect_type:
        raise ValueError(
            '--impedance-type %s 要求 yaml 里 type=%d，实际读到 %d（%s）—— '
            '文件和命令行选的不是同一种阻抗，拒绝启动'
            % (impedance_type, expect_type, cfg.type, cfg_path))
    return cfg, cfg_path


def load_diff_ik_config(path, repo_root):
    """Load+validate a DiffIKConfig, defaulting `path` to
    configs/solver/diff.yaml under `repo_root` when not given. Same shape
    as load_impedance_config() so every entry point offering --solver diff
    (run_teleop.py, replay.py, scripts/umi_replay.py) resolves the default
    path identically.

    :return: (DiffIKConfig, resolved cfg_path)
    :raises ValueError: file missing or fails validation -- message is
        meant to be printed directly, not parsed.
    """
    cfg_path = path or os.path.join(repo_root, 'configs', 'solver', 'diff.yaml')
    try:
        return load_solver_config(DiffIKConfig, cfg_path), cfg_path
    except FileNotFoundError:
        raise ValueError('求解器配置文件不存在：%s' % cfg_path)
    except Exception as e:   # pydantic.ValidationError et al.
        raise ValueError('求解器配置校验失败（%s）：\n%s' % (cfg_path, e))
