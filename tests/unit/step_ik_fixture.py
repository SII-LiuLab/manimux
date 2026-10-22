from dataclasses import replace
from types import SimpleNamespace
from copy import deepcopy
import time

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from manimux.kinematics.poses import matrix_pose, pose_matrix
from manimux.policy_adapter.pose_output import POSE_SPACE, pose_chunk, interpolate_pose
from manimux.runtime.ik import StepIKResolver, validate_ik_config
from manimux.runtime.execution import ExecutionPipeline
from manimux.runtime.executors.direct import DirectExecutor
from manimux.runtime.timeline import ActionTimeline
from manimux.types import ActionContext, ActionHorizon, RobotState

GROUPS = ('left_arm', 'right_arm')

class Arm:
    coordinates = [SimpleNamespace(name=f'q{i}') for i in range(6)] + [SimpleNamespace(name='gripper')]
    def __init__(self):
        self.calls = []; self.fail = False; self.delay = 0
    def fk(self, q):
        m = np.eye(4); m[:3, 3] = q[:3]; m[:3, :3] = Rotation.from_rotvec(q[3:6]).as_matrix(); return m
    def ik(self, target, seed, *, fixed_coordinates):
        self.calls.append((target.copy(), seed.copy()))
        if self.delay: time.sleep(self.delay)
        q = np.r_[target[:3, 3], Rotation.from_matrix(target[:3, :3]).as_rotvec(), fixed_coordinates['gripper']]
        return SimpleNamespace(converged=not self.fail, joints=q, reason='test_failure')


def inputs():
    kin = SimpleNamespace(models={g: Arm() for g in GROUPS})
    seed = {g: np.array([.1, .2, .3, 0, 0, 0, .5]) for g in GROUPS}
    rows = {g: np.tile(np.r_[matrix_pose(kin.models[g].fk(q)), q[-1]], (5, 1)) for g, q in seed.items()}
    for values in rows.values(): values[:, 0] += np.arange(5) * .01
    return kin, seed, rows
