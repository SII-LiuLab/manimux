"""XPolicyLab action-list boundary for the shared AAC selector."""

from manimux.policies import aac as common
from manimux.policies.aac import (  # noqa: F401
    AacPreviousAction,
    AacSelection,
    EeActionStats,
    ee_entropy,
    ee_motion_magnitude,
    ee_pose_increment,
    entropy_elbow,
    load_ee_action_stats,
    motion_floor,
    select_candidate,
)
from manimux.policies.xpolicylab.codec import decode_action_steps


def build_ee_candidates(candidate_chunks, *, layouts, current_groups, kinematics):
    decoded = [decode_action_steps(c, layouts=layouts) for c in candidate_chunks]
    features = common.build_ee_features(
        decoded, layouts=layouts, current_groups=current_groups, kinematics=kinematics
    )
    return features, list(candidate_chunks)


def select_ee_chunk(candidate_chunks, *, layouts, **options):
    decoded = [decode_action_steps(c, layouts=layouts) for c in candidate_chunks]
    _, selection, previous = common.select_ee_chunk(decoded, layouts=layouts, **options)
    selected = list(candidate_chunks[selection.chunk_id][: selection.chunk_size])
    return selected, selection, previous
