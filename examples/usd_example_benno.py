#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

# Third Party
import numpy as np
import open3d as o3d
import os

# CuRobo
from curobo.benno_test.shelf_pointcloud import (
    shelf_mesh_from_pointcloud,
    shelf_pointcloud,
)
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cylinder, WorldConfig
from curobo.rollout.cost.pose_cost import PoseCostMetric
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.util.logger import log_error, setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

_CURRENT_JOINT_POS = None
_CURRENT_JOINT_NAMES = None


def set_current_joint_state(joint_positions, joint_names=None):
    """Set current joint angles (radians) to use as the planning start state."""
    global _CURRENT_JOINT_POS, _CURRENT_JOINT_NAMES
    pos = np.asarray(joint_positions, dtype=float).reshape(1, -1)
    _CURRENT_JOINT_POS = pos
    _CURRENT_JOINT_NAMES = joint_names


def _get_start_state(motion_gen: MotionGen) -> JointState:
    if _CURRENT_JOINT_POS is not None:
        pos_t = motion_gen.tensor_args.to_device(_CURRENT_JOINT_POS)
        return JointState.from_position(pos_t, joint_names=_CURRENT_JOINT_NAMES)
    retract_cfg = motion_gen.get_retract_config()
    return JointState.from_position(retract_cfg.view(1, -1).clone())


def build_shelf_world(
    center_xyz=(2.0, 0.0, 0.3),
    width=0.6,
    depth=0.4,
    height=0.6,
    board_thickness=0.02,
    point_step=0.01,
    mesh_pitch=0.02,
):
    shelf_pc = shelf_pointcloud(
        center_xyz=center_xyz,
        width=width,
        depth=depth,
        height=height,
        board_thickness=board_thickness,
        step=point_step,
    )
    shelf_mesh = shelf_mesh_from_pointcloud(shelf_pc, pitch=mesh_pitch, name="shelf")
    shelf_mesh.color = [0.7, 0.7, 0.7, 1.0]
    world_cfg = WorldConfig(mesh=[shelf_mesh])
    return world_cfg, shelf_mesh, shelf_pc


def save_shelf_world_to_usd(save_path="shelf_world.usd", shelf_center=(2.0, 0.0, 0.3)):
    world_cfg, _, _ = build_shelf_world(center_xyz=shelf_center)
    usd_helper = UsdHelper()
    usd_helper.create_stage(save_path)
    usd_helper.add_world_to_stage(world_cfg)
    usd_helper.write_stage_to_file(save_path)
    print("Wrote:", save_path)


def plan_ur5e_into_shelf(
    world_cfg: WorldConfig,
    goal_pose: Pose,
    interpolation_dt=0.01,
    robot_file="ur5e_robotiq_2f_140.yml",
):
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_cfg,
        collision_checker_type=CollisionCheckerType.MESH,
        interpolation_dt=interpolation_dt,
    )
    motion_gen = MotionGen(motion_gen_config)
    motion_gen.warmup()

    start_state = _get_start_state(motion_gen)

    result = motion_gen.plan_single(
        start_state,
        goal_pose,
        MotionGenPlanConfig(max_attempts=1),
    )
    if not result.success:
        log_error(f"Failed to plan: {result.status}")
        return start_state, None
    return start_state, result.get_interpolated_plan()


def _linear_pose_metric(motion_gen, allow_linear_axis: int, project_to_goal_frame=False):
    # hold orientation and two position axes, allow motion along one axis
    hold = motion_gen.tensor_args.to_device([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    hold[3 + allow_linear_axis] = 0.0
    return PoseCostMetric(
        hold_partial_pose=True,
        hold_vec_weight=hold,
        project_to_goal_frame=project_to_goal_frame,
    )


def _ensure_batched_joint_state(js: JointState) -> JointState:
    pos = js.position
    if pos.ndim == 1:
        pos = pos.view(1, -1)
    elif pos.ndim == 3:
        pos = pos[:, 0, :]
    return JointState.from_position(pos, joint_names=js.joint_names)


def _path_constraint_for_axis(axis: int, constrain_orientation: bool = False):
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0 (X), 1 (Y), or 2 (Z)")
    if constrain_orientation:
        constraint = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    else:
        constraint = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    constraint[3 + axis] = 0.0
    return constraint


def _pose_goalset_from_list(pose_list, tensor_args):
    pos = np.asarray(pose_list[:3], dtype=float).reshape(1, 1, 3)
    quat = np.asarray(pose_list[3:], dtype=float).reshape(1, 1, 4)
    pos_t = tensor_args.to_device(pos)
    quat_t = tensor_args.to_device(quat)
    return Pose(position=pos_t, quaternion=quat_t)


def _pose_goalset_from_poses(pose_lists, tensor_args):
    poses = np.asarray(pose_lists, dtype=float)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(
            "Goalset segment (mode 6) expects a sequence of poses with shape [N, 7]"
        )
    pos_t = tensor_args.to_device(poses[:, :3].reshape(1, -1, 3))
    quat_t = tensor_args.to_device(poses[:, 3:].reshape(1, -1, 4))
    return Pose(position=pos_t, quaternion=quat_t)


def _pose_batch_from_poses(pose_lists, tensor_args):
    poses = np.asarray(pose_lists, dtype=float)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError("Batch segment (mode 7) expects a sequence of poses with shape [N, 7]")
    pos_t = tensor_args.to_device(poses[:, :3].reshape(-1, 1, 3))
    quat_t = tensor_args.to_device(poses[:, 3:].reshape(-1, 1, 4))
    return Pose(position=pos_t, quaternion=quat_t)


def _is_pose7(candidate):
    try:
        pose = np.asarray(candidate, dtype=float)
    except Exception:
        return False
    return pose.ndim == 1 and pose.shape[0] == 7


def _normalize_pose_lists_for_modes(pose_lists, segment_modes):
    """
    Normalize common shorthand input shapes.
    For mode 6/7, allow passing pose_lists=[pose1, pose2, ...] and wrap to one segment.
    """
    if pose_lists is None or segment_modes is None:
        return pose_lists
    if len(segment_modes) == 1 and segment_modes[0] in (6, 7) and len(pose_lists) > 0:
        if _is_pose7(pose_lists[0]):
            return [pose_lists]
    return pose_lists


def _is_joint6(candidate):
    try:
        arr = np.asarray(candidate, dtype=float)
    except Exception:
        return False
    return arr.ndim == 1 and arr.shape[0] == 6


def _is_joint6_list(candidates):
    if not isinstance(candidates, (list, tuple)) or len(candidates) == 0:
        return False
    return all(_is_joint6(c) for c in candidates)


def _parse_mode7_entry(mode7_entry):
    """
    Parse mode 7 input.
    Accepted:
    1) [goal_pose_0, goal_pose_1, ...]                          (start replicated)
    2) [start_joint_or_list, [goal_pose_0, goal_pose_1, ...]]   (explicit starts)
       start_joint_or_list can be [6] or [[6], [6], ...].
    """
    explicit_starts = None
    goals = mode7_entry
    if (
        isinstance(mode7_entry, (list, tuple))
        and len(mode7_entry) == 2
        and (
            _is_joint6(mode7_entry[0])
            or _is_joint6_list(mode7_entry[0])
            or (isinstance(mode7_entry[0], np.ndarray) and np.asarray(mode7_entry[0]).shape[-1] == 6)
        )
    ):
        explicit_starts = np.asarray(mode7_entry[0], dtype=float)
        goals = mode7_entry[1]
    goals_np = np.asarray(goals, dtype=float)
    if goals_np.ndim != 2 or goals_np.shape[1] != 7:
        raise ValueError(
            "mode 7 expects goals as a sequence of poses with shape [N, 7], "
            "or input [[start_joints], [goal_poses]] where start_joints are [6] or [N,6]"
        )
    return explicit_starts, goals_np


def _expand_pose_entry_for_viz(entry):
    if _is_pose7(entry):
        return [list(entry)]
    if isinstance(entry, (list, tuple)) and len(entry) > 0 and all(_is_pose7(p) for p in entry):
        return [list(p) for p in entry]
    if (
        isinstance(entry, (list, tuple))
        and len(entry) == 2
        and isinstance(entry[1], (list, tuple))
        and len(entry[1]) > 0
        and all(_is_pose7(p) for p in entry[1])
    ):
        return [list(p) for p in entry[1]]
    return []


def _prepare_visualization_poses(pose_lists):
    if not pose_lists:
        return None, []
    extras = []
    for entry in pose_lists[:-1]:
        extras.extend(_expand_pose_entry_for_viz(entry))
    last_candidates = _expand_pose_entry_for_viz(pose_lists[-1])
    if not last_candidates:
        all_candidates = []
        for entry in pose_lists:
            all_candidates.extend(_expand_pose_entry_for_viz(entry))
        if not all_candidates:
            return None, extras
        return all_candidates[-1], all_candidates[:-1]
    goal_pose = last_candidates[-1]
    extras.extend(last_candidates[:-1])
    return goal_pose, extras


def _nlerp_quat(q0, q1, t):
    q = (1.0 - t) * q0 + t * q1
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return q0
    return q / norm


def _quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def rotate_pose_local(pose_list, axis_xyz, angle_deg):
    """Rotate pose orientation about a local axis (post-multiply)."""
    axis = np.asarray(axis_xyz, dtype=float)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8:
        raise ValueError("axis_xyz must be non-zero")
    axis = axis / axis_norm
    angle_rad = np.deg2rad(angle_deg)
    half = 0.5 * angle_rad
    q_rot = np.array([np.cos(half), *(np.sin(half) * axis)], dtype=float)
    q_pose = np.asarray(pose_list[3:], dtype=float)
    q_new = _quat_multiply(q_pose, q_rot)
    return [pose_list[0], pose_list[1], pose_list[2], q_new[0], q_new[1], q_new[2], q_new[3]]


def _sample_cartesian_waypoints(start_pose_list, goal_pose_list, step=0.01):
    p0 = np.asarray(start_pose_list[:3], dtype=float)
    p1 = np.asarray(goal_pose_list[:3], dtype=float)
    q0 = np.asarray(start_pose_list[3:], dtype=float)
    q1 = np.asarray(goal_pose_list[3:], dtype=float)
    dist = np.linalg.norm(p1 - p0)
    if dist < 1e-8:
        return [list(goal_pose_list)]
    n = int(np.ceil(dist / step))
    waypoints = []
    for i in range(1, n + 1):
        t = float(i) / float(n)
        p = p0 + t * (p1 - p0)
        q = _nlerp_quat(q0, q1, t)
        waypoints.append([p[0], p[1], p[2], q[0], q[1], q[2], q[3]])
    return waypoints


def _load_robot_cfg_with_attached_object(
    robot_file: str,
    parent_link_name: str = "grasp_frame",
    link_name: str = "attached_object",
    n_spheres: int = 20,
):
    cfg = load_yaml(join_path(get_robot_configs_path(), robot_file))
    if "robot_cfg" in cfg:
        cfg = cfg["robot_cfg"]
    kin = cfg["kinematics"]
    if kin.get("extra_links") is None:
        kin["extra_links"] = {}
    else:
        kin.setdefault("extra_links", {})
    kin["extra_links"][link_name] = {
        "parent_link_name": parent_link_name,
        "link_name": link_name,
        "fixed_transform": [0, 0, 0, 1, 0, 0, 0],
        "joint_type": "FIXED",
        "joint_name": "attach_joint",
    }
    if kin.get("extra_collision_spheres") is None:
        kin["extra_collision_spheres"] = {}
    else:
        kin.setdefault("extra_collision_spheres", {})
    kin["extra_collision_spheres"][link_name] = n_spheres
    kin.setdefault("collision_link_names", [])
    if link_name not in kin["collision_link_names"]:
        kin["collision_link_names"].append(link_name)
    kin.setdefault("self_collision_buffer", {})
    if link_name not in kin["self_collision_buffer"]:
        kin["self_collision_buffer"][link_name] = 0.0
    # add attached_object to self-collision ignore lists for gripper links if present
    gripper_links = [
        "tool0",
        "robotiq_arg2f_base_link",
        "left_outer_knuckle",
        "left_inner_knuckle",
        "left_outer_finger",
        "left_inner_finger",
        "left_inner_finger_pad",
        "right_outer_knuckle",
        "right_inner_knuckle",
        "right_outer_finger",
        "right_inner_finger",
        "right_inner_finger_pad",
    ]
    kin.setdefault("self_collision_ignore", {})
    for ln in gripper_links:
        if ln in kin["self_collision_ignore"]:
            if link_name not in kin["self_collision_ignore"][ln]:
                kin["self_collision_ignore"][ln].append(link_name)
    return cfg


def _attach_cylinder_to_robot(
    motion_gen: MotionGen,
    joint_state: JointState,
    radius: float,
    height: float,
    offset_pose_list,
    link_name: str = "attached_object",
):
    ee_pose = motion_gen.compute_kinematics(joint_state).ee_pose
    offset_pose = Pose.from_list(list(offset_pose_list), tensor_args=motion_gen.tensor_args)
    obj_pose = ee_pose.multiply(offset_pose)
    cyl = Cylinder(
        name="attached_cylinder",
        radius=radius,
        height=height,
        pose=obj_pose.tolist(),
        color=[0.2, 0.8, 0.2, 1.0],
    )
    motion_gen.attach_external_objects_to_robot(
        joint_state=joint_state,
        external_objects=[cyl],
        link_name=link_name,
    )

def _stack_trajectories(prev_traj, next_traj, snap=True, snap_tol=1e-4):
    if prev_traj is None:
        return next_traj
    if next_traj is None:
        return prev_traj
    if next_traj.position.shape[-2] == 0:
        return prev_traj

    last_prev = prev_traj.position[-1]
    first_next = next_traj.position[0]
    delta = (first_next - last_prev).abs().max().item()
    if delta > snap_tol:
        print(f"[warn] segment boundary jump (max |dq|={delta:.6f}), snapping first point")
        if snap:
            next_traj = next_traj.clone()
            next_traj.position[0] = last_prev
            if next_traj.velocity is not None:
                next_traj.velocity[0] = 0.0
            if next_traj.acceleration is not None:
                next_traj.acceleration[0] = 0.0
            if next_traj.jerk is not None:
                next_traj.jerk[0] = 0.0
    else:
        # drop duplicate first point to keep smoothness
        if next_traj.position.shape[-2] > 1:
            next_traj = next_traj.trim_trajectory(1, None)

    return prev_traj.stack(next_traj)


def _cartesian_hold_metric(
    motion_gen: MotionGen,
    start_pose: Pose,
    goal_pose: Pose,
    hold_orientation: bool = False,
    axis_tol: float = 1e-4,
):
    hold = motion_gen.tensor_args.to_device([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    if hold_orientation:
        hold[:3] = 1.0
    delta = (goal_pose.position - start_pose.position).abs()
    for ax in range(3):
        if delta[..., ax].max().item() <= axis_tol:
            hold[3 + ax] = 1.0
    if float(hold.sum()) == 0.0:
        return None
    return PoseCostMetric(
        hold_partial_pose=True,
        hold_vec_weight=hold,
        project_to_goal_frame=False,
    )


def plan_linear_cartesian_segments(
    motion_gen: MotionGen,
    pose_lists,
    linear_axes,
    project_to_goal_frame=False,
):
    if len(pose_lists) < 2:
        raise ValueError("pose_lists must contain at least a start and one goal pose")
    if len(linear_axes) != len(pose_lists) - 1:
        raise ValueError("linear_axes must be one shorter than pose_lists")

    # IK to get a start joint state at the first pose
    start_pose = Pose.from_list(list(pose_lists[0]))
    ik = motion_gen.solve_ik(start_pose, return_seeds=1, use_nn_seed=False)
    if not ik.success.item():
        log_error("IK failed for start pose")
        return None
    curr_state = motion_gen.get_active_js(_ensure_batched_joint_state(ik.js_solution))

    full_traj = None
    for goal_pose_list, axis in zip(pose_lists[1:], linear_axes):
        metric = _linear_pose_metric(
            motion_gen, allow_linear_axis=axis, project_to_goal_frame=project_to_goal_frame
        )
        plan_cfg = MotionGenPlanConfig(max_attempts=3, pose_cost_metric=metric)
        result = motion_gen.plan_single(curr_state, Pose.from_list(list(goal_pose_list)), plan_cfg)
        if not result.success:
            log_error(f"Linear segment failed: {result.status}")
            return None
        seg_traj = result.get_interpolated_plan()
        if full_traj is None:
            full_traj = seg_traj
        else:
            full_traj = full_traj.stack(seg_traj)
        last_pos = seg_traj.position[-1].view(1, -1)
        curr_state = JointState.from_position(last_pos, joint_names=seg_traj.joint_names)

    return full_traj


def plan_linear_segments_via_grasp(
    motion_gen: MotionGen,
    pose_lists,
    linear_axes,
    project_to_goal_frame=False,
    constrain_orientation=False,
):
    if len(pose_lists) < 2:
        raise ValueError("pose_lists must contain at least a start and one goal pose")
    if len(linear_axes) != len(pose_lists) - 1:
        raise ValueError("linear_axes must be one shorter than pose_lists")

    start_pose = Pose.from_list(list(pose_lists[0]))
    ik = motion_gen.solve_ik(start_pose, return_seeds=1, use_nn_seed=False)
    if not ik.success.item():
        log_error("IK failed for start pose")
        return None
    curr_state = motion_gen.get_active_js(_ensure_batched_joint_state(ik.js_solution))

    full_traj = None
    for goal_pose_list, axis in zip(pose_lists[1:], linear_axes):
        goal_pose = Pose.from_list(list(goal_pose_list), tensor_args=motion_gen.tensor_args)
        goal_pose_goalset = _pose_goalset_from_list(goal_pose_list, motion_gen.tensor_args)
        # Compute offset so the "approach" pose equals the current start pose
        if project_to_goal_frame:
            approach_offset = goal_pose.compute_local_pose(start_pose)
        else:
            approach_offset = start_pose.multiply(goal_pose.inverse())
        plan_cfg = MotionGenPlanConfig(False, True, max_attempts=3, enable_finetune_trajopt=False)
        result = motion_gen.plan_grasp(
            curr_state,
            goal_pose_goalset,
            plan_cfg,
            grasp_approach_offset=approach_offset,
            grasp_approach_path_constraint=_path_constraint_for_axis(
                axis, constrain_orientation=constrain_orientation
            ),
            plan_approach_to_grasp=True,
            plan_grasp_to_retract=False,
            grasp_approach_constraint_in_goal_frame=project_to_goal_frame,
        )
        if not result.success.item():
            log_error(f"Linear segment failed: {result.status}")
            return None
        seg_traj = result.grasp_interpolated_trajectory
        if full_traj is None:
            full_traj = seg_traj
        else:
            full_traj = full_traj.stack(seg_traj)
        last_pos = seg_traj.position[-1].view(1, -1)
        curr_state = JointState.from_position(last_pos, joint_names=seg_traj.joint_names)
        start_pose = goal_pose

    return full_traj


def plan_mixed_segments(
    motion_gen: MotionGen,
    pose_lists,
    segment_modes,
    linear_axes,
    project_to_goal_frame=False,
    constrain_orientation=False,
    cartesian_step=0.01,
    cartesian_hold_axes=True,
    start_state: JointState | None = None,
    grasp_retract_offset=None,
    grasp_retract_offsets=None,
    grasp_retract_constraint_in_goal_frame=True,
    grasp_approach_offset=None,
    grasp_approach_constraint_in_goal_frame=True,
    snap_stack=True,
    snap_tol=1e-4,
    grasp_prepose_motion=False,
    debug_jumps=False,
    attach_cylinder=False,
    attach_after_index=None,
    detach_after_index=None,
    cylinder_radius=0.025,
    cylinder_height=0.10,
    cylinder_offset_pose=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    attach_link_name="attached_object",
):
    pose_lists = _normalize_pose_lists_for_modes(pose_lists, segment_modes)

    if len(pose_lists) < 1:
        raise ValueError("pose_lists must contain at least one target pose")
    if len(segment_modes) != len(pose_lists):
        raise ValueError("segment_modes must be same length as pose_lists")
    if len(linear_axes) != sum(1 for m in segment_modes if m == 1):
        raise ValueError("linear_axes must match number of linear segments")

    full_traj = None
    segment_end_indices = []
    curr_state = None
    linear_axes_list = list(linear_axes)
    linear_axis_idx = 0
    attach_indices = []
    detach_indices = []
    if attach_after_index is not None:
        attach_indices = (
            list(attach_after_index)
            if isinstance(attach_after_index, (list, tuple))
            else [attach_after_index]
        )
    if detach_after_index is not None:
        detach_indices = (
            list(detach_after_index)
            if isinstance(detach_after_index, (list, tuple))
            else [detach_after_index]
        )

    for i, pose_list in enumerate(pose_lists):
        mode = segment_modes[i]

        if i == 0 and curr_state is None:
            curr_state = start_state if start_state is not None else _get_start_state(motion_gen)

        if mode == 0:
            goal_pose = Pose.from_list(list(pose_list), tensor_args=motion_gen.tensor_args)
            plan_cfg = MotionGenPlanConfig(max_attempts=3)
            result = motion_gen.plan_single(curr_state, goal_pose, plan_cfg)
            if not result.success:
                log_error(f"Normal segment failed: {result.status}")
                return None
            seg_traj = result.get_interpolated_plan()
        elif mode == 1:
            goal_pose = Pose.from_list(list(pose_list), tensor_args=motion_gen.tensor_args)
            axis = linear_axes_list[linear_axis_idx]
            linear_axis_idx += 1
            if i == 0:
                start_pose = motion_gen.rollout_fn.compute_kinematics(curr_state).ee_pose
            else:
                start_pose = motion_gen.rollout_fn.compute_kinematics(curr_state).ee_pose

            # Adjust goal pose to exactly match start pose on held axes (base frame constraint)
            goal_pose_adj = goal_pose.clone()
            for ax in (0, 1, 2):
                if ax != axis:
                    goal_pose_adj.position[..., ax] = start_pose.position[..., ax]

            if project_to_goal_frame:
                approach_offset = goal_pose_adj.compute_local_pose(start_pose)
            else:
                approach_offset = start_pose.multiply(goal_pose_adj.inverse())

            plan_cfg = MotionGenPlanConfig(False, True, max_attempts=3, enable_finetune_trajopt=False)
            goal_pose_goalset = _pose_goalset_from_list(goal_pose_adj.tolist(), motion_gen.tensor_args)
            result = motion_gen.plan_grasp(
                curr_state,
                goal_pose_goalset,
                plan_cfg,
                grasp_approach_offset=approach_offset,
                grasp_approach_path_constraint=_path_constraint_for_axis(
                    axis, constrain_orientation=constrain_orientation
                ),
                plan_approach_to_grasp=True,
                plan_grasp_to_retract=False,
                grasp_approach_constraint_in_goal_frame=project_to_goal_frame,
            )
            if not result.success.item():
                log_error(f"Linear segment failed: {result.status}")
                return None
            seg_traj = result.grasp_interpolated_trajectory
        elif mode == 2:
            goal_pose = Pose.from_list(list(pose_list), tensor_args=motion_gen.tensor_args)
            if i == 0:
                log_error("Cartesian segment cannot be the first segment")
                return None
            start_pose = motion_gen.rollout_fn.compute_kinematics(curr_state).ee_pose
            metric = None
            if cartesian_hold_axes:
                metric = _cartesian_hold_metric(
                    motion_gen,
                    start_pose,
                    goal_pose,
                    hold_orientation=constrain_orientation,
                    axis_tol=1e-4,
                )
            plan_cfg = MotionGenPlanConfig(
                max_attempts=3,
                enable_finetune_trajopt=True,
                pose_cost_metric=metric,
            )
            result = motion_gen.plan_single(curr_state, goal_pose, plan_cfg)
            if not result.success:
                log_error(f"Cartesian segment failed: {result.status}")
                return None
            seg_traj = result.get_interpolated_plan()
        elif mode == 3 or mode == 4:
            goal_pose = Pose.from_list(list(pose_list), tensor_args=motion_gen.tensor_args)
            plan_cfg = MotionGenPlanConfig(False, True, max_attempts=3)
            goal_pose_goalset = _pose_goalset_from_list(pose_list, motion_gen.tensor_args)
            if grasp_approach_offset is not None:
                approach_offset = Pose.from_list(
                    list(grasp_approach_offset), tensor_args=motion_gen.tensor_args
                )
            else:
                approach_offset = Pose.from_list(
                    [0, 0, -0.15, 1, 0, 0, 0], tensor_args=motion_gen.tensor_args
                )
            # per-grasp retract override
            if grasp_retract_offsets is not None and grasp_retract_offsets[i] is not None:
                effective_retract_offset = grasp_retract_offsets[i]
            else:
                effective_retract_offset = grasp_retract_offset
            if mode == 4 and effective_retract_offset is None:
                effective_retract_offset = [0.0, 0.0, -0.10, 1.0, 0.0, 0.0, 0.0]
            # Optionally insert an explicit normal-motion pre-grasp segment
            if grasp_prepose_motion:
                if grasp_approach_constraint_in_goal_frame:
                    approach_pose = goal_pose.clone().multiply(approach_offset)
                else:
                    approach_pose = approach_offset.clone().multiply(goal_pose.clone())
                pre_cfg = MotionGenPlanConfig(max_attempts=3)
                pre_result = motion_gen.plan_single(curr_state, approach_pose, pre_cfg)
                if not pre_result.success:
                    log_error(f"Pre-grasp segment failed: {pre_result.status}")
                    return None
                pre_traj = pre_result.get_interpolated_plan()
                if full_traj is None:
                    full_traj = pre_traj
                else:
                    full_traj = _stack_trajectories(
                        full_traj, pre_traj, snap=snap_stack, snap_tol=snap_tol
                    )
                last_pos = pre_traj.position[-1].view(1, -1)
                curr_state = JointState.from_position(last_pos, joint_names=pre_traj.joint_names)
            result = motion_gen.plan_grasp(
                curr_state,
                goal_pose_goalset,
                plan_cfg,
                grasp_approach_offset=approach_offset,
                grasp_approach_constraint_in_goal_frame=grasp_approach_constraint_in_goal_frame,
                plan_grasp_to_retract=effective_retract_offset is not None,
                retract_offset=(
                    Pose.from_list(list(effective_retract_offset), tensor_args=motion_gen.tensor_args)
                    if effective_retract_offset is not None
                    else None
                ),
                retract_constraint_in_goal_frame=grasp_retract_constraint_in_goal_frame,
            )
            if not result.success.item():
                log_error(f"Grasp segment failed: {result.status}")
                return None
            if (
                effective_retract_offset is not None
                and result.retract_interpolated_trajectory is not None
            ):
                seg_traj = result.grasp_interpolated_trajectory.stack(
                    result.retract_interpolated_trajectory
                )
            else:
                seg_traj = result.grasp_interpolated_trajectory
        elif mode == 5:
            joint_goal = np.asarray(pose_list, dtype=float).reshape(1, -1)
            if joint_goal.shape[-1] != 6:
                raise ValueError("Joint goal must be 6 elements for mode 5")
            goal_state = JointState.from_position(
                motion_gen.tensor_args.to_device(joint_goal),
                joint_names=motion_gen.joint_names,
            )
            plan_cfg = MotionGenPlanConfig(max_attempts=3)
            result = motion_gen.plan_single_js(curr_state, goal_state, plan_cfg)
            if not result.success:
                log_error(f"Joint segment failed: {result.status}")
                return None
            seg_traj = result.get_interpolated_plan()
        elif mode == 6:
            plan_cfg = MotionGenPlanConfig(max_attempts=3)
            goal_pose_goalset = _pose_goalset_from_poses(pose_list, motion_gen.tensor_args)
            result = motion_gen.plan_goalset(curr_state, goal_pose_goalset, plan_cfg)
            success = result.success.item() if hasattr(result.success, "item") else bool(result.success)
            if not success:
                log_error(f"Goalset segment failed: {result.status}")
                return None
            seg_traj = result.get_interpolated_plan()
        elif mode == 7:
            plan_cfg = MotionGenPlanConfig(max_attempts=3)
            explicit_starts, goals_np = _parse_mode7_entry(pose_list)
            goal_pose_batch = _pose_batch_from_poses(goals_np, motion_gen.tensor_args)
            goal_count = int(goal_pose_batch.batch)

            start_pos = curr_state.position
            if start_pos.ndim == 1:
                start_pos = start_pos.view(1, -1)

            if explicit_starts is not None:
                start_np = np.asarray(explicit_starts, dtype=float)
                if start_np.ndim == 1:
                    if start_np.shape[0] != 6:
                        raise ValueError("mode 7 explicit start joints must have 6 values")
                    start_np = start_np.reshape(1, 6)
                elif start_np.ndim == 2 and start_np.shape[1] == 6:
                    pass
                else:
                    raise ValueError("mode 7 explicit start joints must be [6] or [N,6]")
                if start_np.shape[0] == 1:
                    start_pos = motion_gen.tensor_args.to_device(start_np).repeat(goal_count, 1)
                elif start_np.shape[0] == goal_count:
                    start_pos = motion_gen.tensor_args.to_device(start_np)
                else:
                    raise ValueError(
                        f"mode 7 start joint count ({start_np.shape[0]}) must be 1 or match goals ({goal_count})"
                    )
            elif start_pos.shape[0] == 1:
                start_pos = start_pos.repeat(goal_count, 1)
            elif start_pos.shape[0] != goal_count:
                raise ValueError(
                    f"mode 7 start state batch ({start_pos.shape[0]}) must match goal batch ({goal_count})"
                )
            batch_start = JointState.from_position(start_pos, joint_names=curr_state.joint_names)
            print(
                f"[mode7] plan_batch: start_batch={tuple(batch_start.position.shape)}, "
                f"goal_batch={tuple(goal_pose_batch.position.shape)}"
            )
            result = motion_gen.plan_batch(batch_start, goal_pose_batch, plan_cfg)
            success_mask = result.success.view(-1)
            if int(success_mask.sum().item()) == 0:
                log_error(f"Batch segment failed: {result.status}")
                return None
            all_paths = result.get_paths()
            batch_paths = [
                all_paths[i] for i in range(len(all_paths)) if bool(success_mask[i].item())
            ]
            if len(batch_paths) == 0:
                log_error("Batch segment produced no successful paths")
                return None

            head_paths = []
            head_seg_indices = []
            for path in batch_paths:
                if full_traj is None:
                    combined = path
                else:
                    combined = _stack_trajectories(
                        full_traj, path, snap=snap_stack, snap_tol=snap_tol
                    )
                head_paths.append(combined)
                head_seg_indices.append(
                    list(segment_end_indices) + [combined.position.shape[-2] - 1]
                )

            if i == len(pose_lists) - 1:
                return head_paths, head_seg_indices

            # Non-recursive chaining for later mode-7 segments: keep batch index pairing.
            curr_paths = head_paths
            curr_seg_indices = head_seg_indices
            for j in range(i + 1, len(pose_lists)):
                if segment_modes[j] != 7:
                    raise ValueError(
                        "After mode 7, only mode 7 is supported in non-recursive batching"
                    )

                next_plan_cfg = MotionGenPlanConfig(max_attempts=3)
                next_explicit_starts, next_goals_np = _parse_mode7_entry(pose_lists[j])
                next_goal_pose_batch = _pose_batch_from_poses(next_goals_np, motion_gen.tensor_args)
                next_goal_count = int(next_goal_pose_batch.batch)
                curr_count = len(curr_paths)

                if curr_count == 0:
                    log_error("No batch paths available for chained mode 7 planning")
                    return None

                if next_explicit_starts is not None:
                    start_np = np.asarray(next_explicit_starts, dtype=float)
                    if start_np.ndim == 1:
                        if start_np.shape[0] != 6:
                            raise ValueError("mode 7 explicit start joints must have 6 values")
                        start_np = start_np.reshape(1, 6)
                    elif start_np.ndim == 2 and start_np.shape[1] == 6:
                        pass
                    else:
                        raise ValueError("mode 7 explicit start joints must be [6] or [N,6]")

                    if start_np.shape[0] == 1:
                        if curr_count != 1:
                            raise ValueError(
                                "mode 7 explicit start [6] requires a single current batch path"
                            )
                        next_start_pos = motion_gen.tensor_args.to_device(start_np).repeat(
                            next_goal_count, 1
                        )
                        source_map = [0] * next_goal_count
                    elif start_np.shape[0] == next_goal_count:
                        next_start_pos = motion_gen.tensor_args.to_device(start_np)
                        if curr_count == next_goal_count:
                            source_map = list(range(next_goal_count))
                        elif curr_count == 1:
                            source_map = [0] * next_goal_count
                        else:
                            raise ValueError(
                                "mode 7 explicit start [N,6] requires current batch size N or 1"
                            )
                    else:
                        raise ValueError(
                            f"mode 7 start joint count ({start_np.shape[0]}) must be 1 or match goals ({next_goal_count})"
                        )
                else:
                    if curr_count == 1:
                        next_start_pos = curr_paths[0].position[-1].view(1, -1).repeat(
                            next_goal_count, 1
                        )
                        source_map = [0] * next_goal_count
                    elif curr_count == next_goal_count:
                        start_np = np.stack(
                            [
                                p.position[-1].detach().cpu().numpy()
                                for p in curr_paths
                            ],
                            axis=0,
                        )
                        next_start_pos = motion_gen.tensor_args.to_device(start_np)
                        source_map = list(range(next_goal_count))
                    else:
                        raise ValueError(
                            f"mode 7 chained batch size mismatch: current {curr_count}, next goals {next_goal_count}. "
                            "Use equal sizes for index-wise pairing, or 1 current path to branch once."
                        )

                next_batch_start = JointState.from_position(
                    next_start_pos, joint_names=curr_paths[0].joint_names
                )
                print(
                    f"[mode7] chained plan_batch: start_batch={tuple(next_batch_start.position.shape)}, "
                    f"goal_batch={tuple(next_goal_pose_batch.position.shape)}"
                )
                next_result = motion_gen.plan_batch(
                    next_batch_start, next_goal_pose_batch, next_plan_cfg
                )
                next_success_mask = next_result.success.view(-1)
                if int(next_success_mask.sum().item()) == 0:
                    log_error(f"Batch segment failed: {next_result.status}")
                    return None

                next_all_paths = next_result.get_paths()
                next_paths = []
                next_seg_indices = []
                for k in range(min(len(next_all_paths), len(source_map))):
                    if not bool(next_success_mask[k].item()):
                        continue
                    base_idx = source_map[k]
                    if base_idx >= len(curr_paths):
                        continue
                    combined = _stack_trajectories(
                        curr_paths[base_idx], next_all_paths[k], snap=snap_stack, snap_tol=snap_tol
                    )
                    next_paths.append(combined)
                    next_seg_indices.append(
                        list(curr_seg_indices[base_idx]) + [combined.position.shape[-2] - 1]
                    )

                if len(next_paths) == 0:
                    log_error("Batch segment produced no successful paired paths")
                    return None
                curr_paths = next_paths
                curr_seg_indices = next_seg_indices

            return curr_paths, curr_seg_indices
        else:
            raise ValueError(
                "segment_modes must be 0 (normal), 1 (linear), 2 (cartesian), 3 (grasp), 4 (grasp+retreat), 5 (joint), 6 (goalset), or 7 (batch)"
            )

        if full_traj is None:
            full_traj = seg_traj
        else:
            full_traj = _stack_trajectories(
                full_traj, seg_traj, snap=snap_stack, snap_tol=snap_tol
            )

        last_pos = seg_traj.position[-1].view(1, -1)
        curr_state = JointState.from_position(last_pos, joint_names=seg_traj.joint_names)
        segment_end_indices.append(full_traj.position.shape[-2] - 1)

        # attach/detach after reaching a pose index
        if attach_cylinder and i in attach_indices:
            _attach_cylinder_to_robot(
                motion_gen,
                curr_state,
                radius=cylinder_radius,
                height=cylinder_height,
                offset_pose_list=cylinder_offset_pose,
                link_name=attach_link_name,
            )
        if attach_cylinder and i in detach_indices:
            motion_gen.detach_object_from_robot(link_name=attach_link_name)

    if debug_jumps and full_traj is not None:
        dq = (full_traj.position[1:] - full_traj.position[:-1]).abs().max(dim=-1).values
        max_dq = dq.max().item()
        idx = int(dq.argmax().item())
        print(f"[debug] max joint step = {max_dq:.6f} at index {idx}")

    return full_traj, segment_end_indices


def _quat_wxyz_to_rotmat(qw, qx, qy, qz):
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def _pose_list_to_matrix(pose_list):
    if len(pose_list) != 7:
        raise ValueError("Expected pose list [x,y,z,qw,qx,qy,qz]")
    x, y, z, qw, qx, qy, qz = pose_list
    rot = _quat_wxyz_to_rotmat(qw, qx, qy, qz)
    mat = np.eye(4, dtype=float)
    mat[:3, :3] = rot
    mat[:3, 3] = [x, y, z]
    return mat


def visualize_pointcloud_with_goal(
    pointcloud,
    goal_pose_list,
    frame_size=0.1,
    extra_pose_lists=None,
):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud.astype(float))
    pcd.paint_uniform_color([0.7, 0.7, 0.7])

    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)

    geoms = [pcd, world_frame]

    def add_pose_frame(pose_list, color):
        pose_mat = _pose_list_to_matrix(pose_list)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
        frame.transform(pose_mat)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=frame_size * 0.2)
        sphere.paint_uniform_color(color)
        sphere.translate(pose_mat[:3, 3])
        geoms.extend([frame, sphere])

    # goal in red
    add_pose_frame(goal_pose_list, [1.0, 0.2, 0.2])
    if extra_pose_lists:
        colors = [
            [0.2, 1.0, 0.2],
            [0.2, 0.6, 1.0],
            [1.0, 0.6, 0.2],
            [0.8, 0.2, 1.0],
        ]
        for i, pose_list in enumerate(extra_pose_lists):
            add_pose_frame(pose_list, colors[i % len(colors)])

    o3d.visualization.draw_geometries(geoms)


def save_ur5e_shelf_motion_to_usd(
    save_path="shelf_ur5e_motion.usd",
    shelf_center=(2.0, 0.0, 0.3),
    goal_pose_list=None,
    visualize_goal=True,
    robot_file="ur5e_robotiq_2f_140.yml",
):
    world_cfg, _, shelf_pc = build_shelf_world(center_xyz=shelf_center)
    if goal_pose_list is None:
        goal_pose_list = (
            0.4,
            0.0,
            0.4,
            1.0,
            0.0,
            0.0,
            0.0,
        )
    if visualize_goal:
        visualize_pointcloud_with_goal(shelf_pc, list(goal_pose_list))
    goal_pose = Pose.from_list(list(goal_pose_list))
    start_state, traj = plan_ur5e_into_shelf(world_cfg, goal_pose, robot_file=robot_file)
    if traj is None:
        print("No trajectory to save.")
        return

    UsdHelper.write_trajectory_animation_with_robot_usd(
        robot_file,
        world_cfg,
        start_state,
        traj,
        dt=0.01,
        save_path=save_path,
        base_frame="/world",
        flatten_usd=True,
    )
    print("Wrote:", save_path)


def build_world_from_pointcloud_npy(
    npy_path,
    mesh_pitch=0.02,
    name="pc_world",
    z_offset=0.0,
):
    pc = np.load(npy_path)
    if pc.ndim != 2 or pc.shape[1] != 3:
        raise ValueError(f"Expected pointcloud shape [N,3], got {pc.shape}")
    if z_offset != 0.0:
        pc = pc.copy()
        pc[:, 2] += z_offset
    mesh = shelf_mesh_from_pointcloud(pc, pitch=mesh_pitch, name=name)
    mesh.color = [0.6, 0.6, 0.9, 1.0]
    return WorldConfig(mesh=[mesh]), pc


def save_ur5e_motion_with_pointcloud_world_to_usd(
    npy_path,
    save_path="pc_world_ur5e_motion.usd",
    #goal_pose_list=([-0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]),
    goal_pose_list=([0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]),
    visualize_goal=True,
    robot_file="ur5e_robotiq_2f_140.yml",
    z_offset=0.0,
):
    world_cfg, pc = build_world_from_pointcloud_npy(npy_path, z_offset=z_offset)
    if visualize_goal:
        visualize_pointcloud_with_goal(pc, list(goal_pose_list))
    goal_pose = Pose.from_list(list(goal_pose_list))
    start_state, traj = plan_ur5e_into_shelf(world_cfg, goal_pose, robot_file=robot_file)
    if traj is None:
        print("No trajectory to save.")
        return

    UsdHelper.write_trajectory_animation_with_robot_usd(
        robot_file,
        world_cfg,
        start_state,
        traj,
        dt=0.01,
        save_path=save_path,
        base_frame="/world",
        flatten_usd=True,
    )
    print("Wrote:", save_path)


def save_ur5e_linear_waypoints_to_usd(
    world_cfg: WorldConfig,
    save_path="linear_waypoints.usd",
    pose_lists=(
        (0.0, 0.5, 0.47, 0.5, -0.5, 0.5, 0.5),
        (0.0, 0.6, 0.47, 0.5, -0.5, 0.5, 0.5),
        (0.0, 0.6, 0.45, 0.5, -0.5, 0.5, 0.5),
    ),
    # 0->X, 1->Y, 2->Z (world axes)
    linear_axes=(1, 2),
    robot_file="ur5e_robotiq_2f_140.yml",
    pointcloud=None,
    visualize_frames=True,
    use_grasp=True,
    constrain_orientation=False,
):
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_cfg,
        collision_checker_type=CollisionCheckerType.MESH,
        interpolation_dt=0.01,
        use_cuda_graph=False,
    )
    motion_gen = MotionGen(motion_gen_config)
    motion_gen.warmup()

    if visualize_frames and pointcloud is not None and pose_lists:
        goal_pose_list, extra_pose_lists = _prepare_visualization_poses(pose_lists)
        if goal_pose_list is not None:
            visualize_pointcloud_with_goal(
                pointcloud,
                goal_pose_list,
                extra_pose_lists=extra_pose_lists if extra_pose_lists else None,
            )

    if use_grasp:
        traj = plan_linear_segments_via_grasp(
            motion_gen,
            pose_lists=pose_lists,
            linear_axes=linear_axes,
            project_to_goal_frame=False,
            constrain_orientation=constrain_orientation,
        )
    else:
        traj = plan_linear_cartesian_segments(
            motion_gen,
            pose_lists=pose_lists,
            linear_axes=linear_axes,
            project_to_goal_frame=False,
        )
    if traj is None:
        print("No trajectory to save.")
        return
    if isinstance(traj, list):
        base, ext = os.path.splitext(save_path)
        if ext == "":
            ext = ".usd"
        write_count = min(3, len(traj))
        written = []
        for i in range(write_count):
            path_i = traj[i]
            start_state_i = JointState.from_position(
                path_i.position[0].view(1, -1), joint_names=path_i.joint_names
            )
            q_traj_i = path_i.clone()
            q_traj_i.position = q_traj_i.position.contiguous()
            if q_traj_i.velocity is not None:
                q_traj_i.velocity = q_traj_i.velocity.contiguous()
            if q_traj_i.acceleration is not None:
                q_traj_i.acceleration = q_traj_i.acceleration.contiguous()
            if q_traj_i.jerk is not None:
                q_traj_i.jerk = q_traj_i.jerk.contiguous()
            out_path = f"{base}_batch_{i}{ext}"
            UsdHelper.write_trajectory_animation_with_robot_usd(
                robot_file,
                world_cfg,
                start_state_i,
                q_traj_i,
                dt=motion_gen_config.interpolation_dt,
                save_path=out_path,
                base_frame="/world",
                flatten_usd=True,
            )
            written.append(out_path)
        print("Wrote batch files:", ", ".join(written))
        return

    start_state = JointState.from_position(
        traj.position[0].view(1, -1), joint_names=traj.joint_names
    )
    q_traj = traj.clone()
    q_traj.position = q_traj.position.contiguous()
    if q_traj.velocity is not None:
        q_traj.velocity = q_traj.velocity.contiguous()
    if q_traj.acceleration is not None:
        q_traj.acceleration = q_traj.acceleration.contiguous()
    if q_traj.jerk is not None:
        q_traj.jerk = q_traj.jerk.contiguous()
    UsdHelper.write_trajectory_animation_with_robot_usd(
        robot_file,
        world_cfg,
        start_state,
        q_traj,
        dt=motion_gen_config.interpolation_dt,
        save_path=save_path,
        base_frame="/world",
        flatten_usd=True,
    )
    print("Wrote:", save_path)


def save_ur5e_mixed_waypoints_to_usd(
    world_cfg: WorldConfig,
    save_path="mixed_waypoints.usd",
    pose_lists=(
        (0.0, 0.5, 0.47, 0.5, -0.5, 0.5, 0.5),
        (0.0, 0.6, 0.47, 0.5, -0.5, 0.5, 0.5),
        (0.0, 0.6, 0.45, 0.5, -0.5, 0.5, 0.5),
    ),
    segment_modes=(0, 1, 1),
    # linear axis per linear segment: 0->X, 1->Y, 2->Z (world axes)
    linear_axes=(1, 2),
    robot_file="ur5e_robotiq_2f_140.yml",
    pointcloud=None,
    visualize_frames=True,
    constrain_orientation=False,
    cartesian_step=0.01,
    cartesian_hold_axes=True,
    grasp_retract_offset=None,
    grasp_retract_offsets=None,
    grasp_retract_constraint_in_goal_frame=True,
    grasp_approach_offset=None,
    grasp_approach_constraint_in_goal_frame=True,
    snap_stack=True,
    snap_tol=1e-4,
    grasp_prepose_motion=False,
    debug_jumps=False,
    attach_cylinder=False,
    attach_after_index=None,
    detach_after_index=None,
    cylinder_radius=0.025,
    cylinder_height=0.10,
    cylinder_offset_pose=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    attach_link_name="attached_object",
):
    if attach_cylinder:
        robot_cfg = _load_robot_cfg_with_attached_object(
            robot_file, parent_link_name="grasp_frame", link_name=attach_link_name
        )
    else:
        robot_cfg = robot_file
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        collision_checker_type=CollisionCheckerType.MESH,
        interpolation_dt=0.01,
        use_cuda_graph=False,
    )
    motion_gen = MotionGen(motion_gen_config)
    motion_gen.warmup()

    if visualize_frames and pointcloud is not None and pose_lists:
        goal_pose_list, extra_pose_lists = _prepare_visualization_poses(pose_lists)
        if goal_pose_list is not None:
            visualize_pointcloud_with_goal(
                pointcloud,
                goal_pose_list,
                extra_pose_lists=extra_pose_lists if extra_pose_lists else None,
            )

    traj, seg_end_indices = plan_mixed_segments(
        motion_gen,
        pose_lists=pose_lists,
        segment_modes=segment_modes,
        linear_axes=linear_axes,
        project_to_goal_frame=False,
        constrain_orientation=constrain_orientation,
        cartesian_step=cartesian_step,
        cartesian_hold_axes=cartesian_hold_axes,
        grasp_retract_offset=grasp_retract_offset,
        grasp_retract_offsets=grasp_retract_offsets,
        grasp_retract_constraint_in_goal_frame=grasp_retract_constraint_in_goal_frame,
        grasp_approach_offset=grasp_approach_offset,
        grasp_approach_constraint_in_goal_frame=grasp_approach_constraint_in_goal_frame,
        snap_stack=snap_stack,
        snap_tol=snap_tol,
        grasp_prepose_motion=grasp_prepose_motion,
        debug_jumps=debug_jumps,
        attach_cylinder=attach_cylinder,
        attach_after_index=attach_after_index,
        detach_after_index=detach_after_index,
        cylinder_radius=cylinder_radius,
        cylinder_height=cylinder_height,
        cylinder_offset_pose=cylinder_offset_pose,
        attach_link_name=attach_link_name,
    )
    if traj is None:
        print("No trajectory to save.")
        return
    if isinstance(traj, list):
        base, ext = os.path.splitext(save_path)
        if ext == "":
            ext = ".usd"
        write_count = min(3, len(traj))
        written = []
        for i in range(write_count):
            path_i = traj[i]
            start_state_i = JointState.from_position(
                path_i.position[0].view(1, -1), joint_names=path_i.joint_names
            )
            q_traj_i = path_i.clone()
            q_traj_i.position = q_traj_i.position.contiguous()
            if q_traj_i.velocity is not None:
                q_traj_i.velocity = q_traj_i.velocity.contiguous()
            if q_traj_i.acceleration is not None:
                q_traj_i.acceleration = q_traj_i.acceleration.contiguous()
            if q_traj_i.jerk is not None:
                q_traj_i.jerk = q_traj_i.jerk.contiguous()
            out_path = f"{base}_batch_{i}{ext}"
            UsdHelper.write_trajectory_animation_with_robot_usd(
                robot_file,
                world_cfg,
                start_state_i,
                q_traj_i,
                dt=motion_gen_config.interpolation_dt,
                save_path=out_path,
                base_frame="/world",
                flatten_usd=True,
            )
            written.append(out_path)
        print("Wrote batch files:", ", ".join(written))
        return

    start_state = JointState.from_position(
        traj.position[0].view(1, -1), joint_names=traj.joint_names
    )
    q_traj = traj.clone()
    q_traj.position = q_traj.position.contiguous()
    if q_traj.velocity is not None:
        q_traj.velocity = q_traj.velocity.contiguous()
    if q_traj.acceleration is not None:
        q_traj.acceleration = q_traj.acceleration.contiguous()
    if q_traj.jerk is not None:
        q_traj.jerk = q_traj.jerk.contiguous()
    UsdHelper.write_trajectory_animation_with_robot_usd(
        robot_file,
        world_cfg,
        start_state,
        q_traj,
        dt=motion_gen_config.interpolation_dt,
        save_path=save_path,
        base_frame="/world",
        flatten_usd=True,
    )
    if attach_cylinder and attach_after_index is not None:
        usd_helper = UsdHelper()
        usd_helper.load_stage_from_file(save_path)
        usd_helper.interpolation_steps = 1
        usd_helper.dt = motion_gen_config.interpolation_dt
        attach_indices = (
            list(attach_after_index)
            if isinstance(attach_after_index, (list, tuple))
            else [attach_after_index]
        )
        detach_indices = (
            list(detach_after_index)
            if isinstance(detach_after_index, (list, tuple))
            else ([detach_after_index] if detach_after_index is not None else [])
        )
        attach_steps = [seg_end_indices[i] for i in attach_indices]
        detach_steps = [seg_end_indices[i] for i in detach_indices] if detach_indices else []
        link_poses = motion_gen.kinematics.get_link_poses(
            q_traj.position, ["grasp_frame"]
        )
        offset_pose = Pose.from_list(list(cylinder_offset_pose), tensor_args=motion_gen.tensor_args)
        pos_seq = []
        quat_seq = []
        for t in range(q_traj.position.shape[0]):
            visible = False
            for k, a_step in enumerate(attach_steps):
                d_step = detach_steps[k] if k < len(detach_steps) else None
                if t >= a_step and (d_step is None or t <= d_step):
                    visible = True
                    break
            if not visible:
                pos_seq.append([0.0, 0.0, -10.0])
                quat_seq.append([1.0, 0.0, 0.0, 0.0])
            else:
                ee_pose = Pose(
                    link_poses.position[t, 0, :],
                    link_poses.quaternion[t, 0, :],
                    normalize_rotation=False,
                )
                obj_pose = ee_pose.multiply(offset_pose)
                pos_seq.append(obj_pose.position.squeeze().cpu().tolist())
                quat_seq.append(obj_pose.quaternion.squeeze().cpu().tolist())

        pos_t = motion_gen.tensor_args.to_device(np.asarray(pos_seq)).view(-1, 1, 3)
        quat_t = motion_gen.tensor_args.to_device(np.asarray(quat_seq)).view(-1, 1, 4)
        pose_traj = Pose(position=pos_t, quaternion=quat_t, normalize_rotation=False)

        cyl = Cylinder(
            name="attached_cylinder",
            radius=cylinder_radius,
            height=cylinder_height,
            pose=[pos_seq[0][0], pos_seq[0][1], pos_seq[0][2], *quat_seq[0]],
            color=[0.2, 0.8, 0.2, 1.0],
        )
        usd_helper.create_animation(
            WorldConfig(objects=[cyl]),
            pose_traj,
            base_frame="/world",
            robot_frame="attached_object",
            dt=motion_gen_config.interpolation_dt,
        )
        usd_helper.write_stage_to_file(save_path, flatten=True)
    print("Wrote:", save_path)


if __name__ == "__main__":
    setup_curobo_logger("error")
    # Only the shelf geometry:
    # save_shelf_world_to_usd()
    # Always export UR5e + shelf USD:
    #save_ur5e_shelf_motion_to_usd()
    # Optional: load a pointcloud npy and plan with it
    # save_ur5e_motion_with_pointcloud_world_to_usd(
    #     "/home/wingende/Documents/ICRA25/curobo/curobo/src/curobo/benno_test/pcls/bookshelf_tall_index1_filtered.npy",
    #     goal_pose_list=[0.0, 0.6, 0.45, 0.5, -0.5, 0.5, 0.5],
    #     z_offset=-0.4,
    #     #robot_file="ur5e.yml"
    # )
    world_cfg, pc = build_world_from_pointcloud_npy(
        "/home/kreis/ws/packages/curobo/src/curobo/benno_test/pcls/bookshelf_tall_index1_filtered.npy",
        z_offset=-0.4,
    )

    set_current_joint_state([-3.1358493 , -0.70846184 ,-1.60364389,  4.38522927 , 1.5280658  , 3.45011806])  # radians
    # save_ur5e_mixed_waypoints_to_usd(
    #     world_cfg,
    #     pointcloud=pc,
    #     robot_file="ur5e_robotiq_2f_85.yml",
    #     pose_lists = [
    #         (-2.858493 , -0.70846184 ,-1.60364389,  4.38522927 , 1.5280658  , 3.45011806),   # joint goal
    #     ],
    #     segment_modes=(5,),
    #     linear_axes=(),
    #     grasp_prepose_motion=True,
    #     attach_cylinder=True,
    #     attach_after_index=[],
    #     detach_after_index=[],
    #     cylinder_radius=0.05,
    #     cylinder_height=0.15,
    #     cylinder_offset_pose = rotate_pose_local(
    #                             (0, 0, 0.0, 1, 0, 0, 0),
    #                             axis_xyz=(0, 1, 0),
    #                             angle_deg=90,
    #                         )
    # )



    grasp_pose = (0.0, 0.63, 0.72, 0.5, -0.5, 0.5, 0.5)
    grasp_pose = rotate_pose_local(grasp_pose, axis_xyz=(0, 1, 0), angle_deg=25)

    pose = (0.2, 0.5, 0.47, 0.5, -0.5, 0.5, 0.5)
    pose = rotate_pose_local(pose, axis_xyz=(1, 0, 0), angle_deg=15)
    pose = rotate_pose_local(pose, axis_xyz=(0, 1, 0), angle_deg=15)
    pose2 = (0.2, 0.67, 0.47, 0.5, -0.5, 0.5, 0.5)
    pose2 = rotate_pose_local(pose2, axis_xyz=(1, 0, 0), angle_deg=15)
    pose2 = rotate_pose_local(pose2, axis_xyz=(0, 1, 0), angle_deg=15)

    # save_ur5e_mixed_waypoints_to_usd(
    #     world_cfg,
    #     pointcloud=pc,
    #     robot_file="ur5e_robotiq_2f_85.yml",
    #     pose_lists=(
    #         (0.0, 0.5, 0.72, 0.5, -0.5, 0.5, 0.5),  # target 0
    #         grasp_pose,                               # target 1 (grasp)
    #         (0.0, 0.5, 0.47, 0.5, -0.5, 0.5, 0.5),    # target 2
    #         (0.0, 0.63, 0.47, 0.5, -0.5, 0.5, 0.5),   # target 3
    #         (0.0, 0.63, 0.45, 0.5, -0.5, 0.5, 0.5),   # target 4
    #         (0.0, 0.5, 0.47, 0.5, -0.5, 0.5, 0.5),    # target 5
    #         (0.2, 0.5, 0.47, 0.5, -0.5, 0.5, 0.5),  # target 6
    #         pose, # target 7
    #         pose2, # target 8
    #         (0.2, 0.5, 0.47, 0.5, -0.5, 0.5, 0.5),  # target 9
    #     ),
    #     segment_modes=(0, 4, 0, 1, 1, 2, 0, 0, 2, 2),
    #     linear_axes=(1, 2),
    #     grasp_prepose_motion=True,
    #     attach_cylinder=True,
    #     attach_after_index=[1, 8],
    #     detach_after_index=[4, 9],
    #     cylinder_radius=0.05,
    #     cylinder_height=0.15,
    #     cylinder_offset_pose = rotate_pose_local(
    #                             (0, 0, 0.0, 1, 0, 0, 0),
    #                             axis_xyz=(0, 1, 0),
    #                             angle_deg=90,
    #                         )
    # )

    # save_ur5e_mixed_waypoints_to_usd(
    #     world_cfg,
    #     pointcloud=pc,
    #     robot_file="ur5e_robotiq_2f_85.yml",
    #     pose_lists=(
    #         (0.0, 0.5, 0.72, 0.5, -0.5, 0.5, 0.5),  # target 0
    #         grasp_pose,                               # target 1 (grasp)
    #         (0.0, 0.5, 0.47, 0.5, -0.5, 0.5, 0.5),    # target 2
    #         (0.0, 0.63, 0.47, 0.5, -0.5, 0.5, 0.5),   # target 3
    #         (0.0, 0.63, 0.45, 0.5, -0.5, 0.5, 0.5),   # target 4
    #         (0.0, 0.5, 0.47, 0.5, -0.5, 0.5, 0.5),    # target 5
    #         (0.2, 0.5, 0.47, 0.5, -0.5, 0.5, 0.5),  # target 6
    #     ),
    #     segment_modes=(0, 4, 0, 1, 1, 2, 0),
    #     linear_axes=(1, 2),
    #     grasp_prepose_motion=True,
    #     attach_cylinder=True,
    #     attach_after_index=[1],
    #     detach_after_index=[4],
    #     cylinder_radius=0.05,
    #     cylinder_height=0.15,
    #     cylinder_offset_pose = rotate_pose_local(
    #                             (0, 0, 0.0, 1, 0, 0, 0),
    #                             axis_xyz=(0, 1, 0),
    #                             angle_deg=90,
    #                         )
    # )

    # Test collisions faster
    # save_ur5e_mixed_waypoints_to_usd(
    #     world_cfg,
    #     pointcloud=pc,
    #     robot_file="ur5e_robotiq_2f_85.yml",
    #     pose_lists=[
    #         (0.0, 0.5, 0.72, -0.5, 0.5, 0.5, 0.5),  # target 0
    #     ],
    #     segment_modes=(0,),
    #     linear_axes=(),
    #     grasp_prepose_motion=True,
    #     attach_cylinder=True,
    #     attach_after_index=[],
    #     detach_after_index=[],
    #     cylinder_radius=0.05,
    #     cylinder_height=0.15,
    #     cylinder_offset_pose = rotate_pose_local(
    #                             (0, 0, 0.0, 1, 0, 0, 0),
    #                             axis_xyz=(0, 1, 0),
    #                             angle_deg=90,
    #                         )
    # )
    
    save_ur5e_mixed_waypoints_to_usd(
        world_cfg,
        pointcloud=pc,
        robot_file="ur5e_robotiq_2f_85.yml",
        pose_lists=[
            [0.0, 0.5, 0.72, -0.5, 0.5, 0.5, 0.5],  # mode 0
            # [
            #     [0.1, 0.4, 0.72, -0.5, 0.5, 0.5, 0.5],
            #     [0.2, 0.5, 0.72, -0.5, 0.5, 0.5, 0.5],
            #     [0.0, 0.7, 0.72, -0.5, 0.5, 0.5, 0.5],
            # ],
            # [
            #     [0.0, 0.5, 0.73, -0.5, 0.5, 0.5, 0.5],
            #     [0.0, 0.6, 0.72, -0.5, 0.5, 0.5, 0.5],
            #     [0.0, 0.4, 0.74, -0.5, 0.5, 0.5, 0.5],
            # ],
        ],
        segment_modes=(0, ),
        linear_axes=(),
        grasp_prepose_motion=True,
        attach_cylinder=True,
        attach_after_index=[],
        detach_after_index=[],
        cylinder_radius=0.05,
        cylinder_height=0.15,
        cylinder_offset_pose=rotate_pose_local(
            (0, 0, 0.0, 1, 0, 0, 0),
            axis_xyz=(0, 1, 0),
            angle_deg=90,
        ),
    )
