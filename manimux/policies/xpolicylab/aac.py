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


def select_native_chunk(candidate_chunks, *, layouts, current_groups, kinematics, **options):
    """Score absolute EE motion without interpreting relative rows as joints."""
    from collections.abc import Mapping

    import numpy as np
    from scipy.spatial.transform import Rotation

    from manimux.kinematics.poses import pose_matrix

    features = []
    for candidate in candidate_chunks:
        semantics = candidate.get("action_semantics") if isinstance(candidate, Mapping) else None
        actions = candidate["actions"] if isinstance(candidate, Mapping) else candidate
        if semantics == "anchor_relative_eef_axis_angle_60":
            rows = np.asarray(actions, dtype=float)
            if rows.ndim != 2 or rows.shape[1] != 60 or not np.isfinite(rows).all():
                raise ValueError("AAC XR1 candidates must be finite [H,60] arrays")
            poses, grips = {}, {}
            for index, layout in enumerate(layouts):
                anchor = kinematics.models[layout.group].fk(current_groups[layout.group])
                offset = 0 if index == 0 else 8
                targets = np.repeat(np.eye(4)[None], len(rows), axis=0)
                targets[:, :3, 3] = anchor[:3, 3] + rows[:, offset : offset + 3] @ anchor[:3, :3].T
                targets[:, :3, :3] = (
                    anchor[:3, :3]
                    @ Rotation.from_rotvec(rows[:, offset + 3 : offset + 6]).as_matrix()
                )
                poses[layout.group] = targets
                grips[layout.group] = current_groups[layout.group][-1] + rows[:, offset + 6]
        elif (
            len(actions)
            and isinstance(actions[0], Mapping)
            and f"{layouts[0].prefix}_ee_pose" in actions[0]
        ):
            if semantics not in {None, "absolute_per_arm_base_xyz_wxyz"}:
                raise ValueError(f"Unsupported AAC pose semantics: {semantics}")
            poses = {
                layout.group: np.stack(
                    [pose_matrix(a[f"{layout.prefix}_ee_pose"]) for a in actions]
                )
                for layout in layouts
            }
            grips = {
                layout.group: np.array(
                    [float(np.asarray(a[layout.gripper_key]).item()) for a in actions]
                )
                for layout in layouts
            }
        else:
            groups = decode_action_steps(actions, layouts=layouts)
            if semantics == "anchor_relative_arm_absolute_gripper":
                for layout in layouts:
                    groups[layout.group][:, : layout.arm_dofs] += current_groups[layout.group][
                        : layout.arm_dofs
                    ]
            elif semantics is not None:
                raise ValueError(f"Unsupported AAC joint semantics: {semantics}")
            poses = {
                layout.group: np.stack(
                    [kinematics.models[layout.group].fk(a) for a in groups[layout.group]]
                )
                for layout in layouts
            }
            grips = {layout.group: groups[layout.group][:, -1] for layout in layouts}
        arms = []
        for layout in layouts:
            previous = kinematics.models[layout.group].fk(current_groups[layout.group])
            rows = []
            for pose, grip in zip(poses[layout.group], grips[layout.group], strict=True):
                rows.append(np.r_[common.ee_pose_increment(previous, pose), grip])
                previous = pose
            arms.append(rows)
        features.append(np.stack(arms, axis=1))
    features = np.asarray(features)
    if features.ndim != 4 or features.shape[2:] != (2, 7) or not np.isfinite(features).all():
        raise ValueError("AAC requires finite, equal-horizon dual-arm candidates")
    selection, previous = common.select_ee_features(features, **options)
    chosen = candidate_chunks[selection.chunk_id]
    result = dict(chosen) if isinstance(chosen, Mapping) else {"actions": chosen}
    result["aac"] = {**selection.metadata(), "execution_steps": selection.chunk_size}
    return result, previous


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
