#!/usr/bin/env python3
# ROS 2 node that listens to a pointcloud and triggers mixed waypoint planning on /planner/go.

import importlib.util
import threading
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from generate_motion_msgs.msg import GenerateMotion

from curobo.geom.types import Mesh, WorldConfig


def _load_usd_example_module():
    root = Path(__file__).resolve().parents[3]
    mod_path = root / "examples" / "usd_example_benno.py"
    spec = importlib.util.spec_from_file_location("usd_example_benno", mod_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_usd_example = _load_usd_example_module()


class MixedPlannerNode(Node):
    def __init__(self):
        super().__init__("curobo_mixed_planner")
        self.declare_parameter("pointcloud_topic", "/pointcloud")
        self.declare_parameter("motion_topic", "/generate_motion")
        self.declare_parameter("robot_file", "ur5e_robotiq_2f_85.yml")
        self.declare_parameter("mesh_pitch", 0.02)
        self.declare_parameter("save_path", "mixed_waypoints.usd")

        pc_topic = self.get_parameter("pointcloud_topic").get_parameter_value().string_value
        motion_topic = self.get_parameter("motion_topic").get_parameter_value().string_value

        self._latest_pc = None
        self._busy = False
        self._lock = threading.Lock()

        self.create_subscription(PointCloud2, pc_topic, self._pc_cb, 1)
        self.create_subscription(GenerateMotion, motion_topic, self._motion_cb, 1)
        self.get_logger().info(f"Listening to pointcloud: {pc_topic}")
        self.get_logger().info(f"Listening to motion: {motion_topic}")

    def _pc_cb(self, msg: PointCloud2):
        raw = list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        points = np.asarray(raw)
        if points.size == 0:
            self.get_logger().warn("Received empty pointcloud")
            return
        if points.dtype.fields is not None:
            # Structured array (x,y,z) -> Nx3 float32
            points = np.column_stack((points["x"], points["y"], points["z"]))
        points = np.asarray(points, dtype=np.float32)
        with self._lock:
            self._latest_pc = points

    def _motion_cb(self, msg: GenerateMotion):
        if self._busy:
            self.get_logger().warn("Planner is busy, ignoring trigger")
            return
        with self._lock:
            if self._latest_pc is None:
                self.get_logger().warn("No pointcloud received yet")
                return
            pc = self._latest_pc.copy()
        self._busy = True
        threading.Thread(target=self._run_plan, args=(pc, msg), daemon=True).start()

    def _run_plan(self, pc: np.ndarray, msg: GenerateMotion):
        try:
            mesh_pitch = (
                self.get_parameter("mesh_pitch").get_parameter_value().double_value
            )
            robot_file = msg.robot_file or self.get_parameter("robot_file").get_parameter_value().string_value
            save_path = self.get_parameter("save_path").get_parameter_value().string_value

            mesh = Mesh.from_pointcloud(pc, pitch=mesh_pitch, name="pc_world")
            world_cfg = WorldConfig(mesh=[mesh])

            default_pose_lists = (
                (0.0, 0.5, 0.72, 0.5, -0.5, 0.5, 0.5),  # target 0
                (0.0, 0.63, 0.72, 0.5, -0.5, 0.5, 0.5),  # target 1 (grasp)
                (0.0, 0.5, 0.47, 0.5, -0.5, 0.5, 0.5),  # target 2
                (0.0, 0.63, 0.47, 0.5, -0.5, 0.5, 0.5),  # target 3
                (0.0, 0.63, 0.45, 0.5, -0.5, 0.5, 0.5),  # target 4
                (0.0, 0.5, 0.47, 0.5, -0.5, 0.5, 0.5),  # target 5
                (0.2, 0.5, 0.47, 0.5, -0.5, 0.5, 0.5),  # target 6
            )
            pose_lists = default_pose_lists
            if msg.pose_lists:
                if len(msg.pose_lists) % 7 != 0:
                    self.get_logger().error(
                        f"pose_lists length {len(msg.pose_lists)} is not a multiple of 7"
                    )
                    return
                pose_lists = tuple(
                    tuple(msg.pose_lists[i : i + 7]) for i in range(0, len(msg.pose_lists), 7)
                )
                if msg.segment_modes and len(msg.segment_modes) != len(pose_lists):
                    self.get_logger().error(
                        "segment_modes length must match pose_lists length "
                        f"({len(msg.segment_modes)} != {len(pose_lists)})"
                    )
                    return

            self.get_logger().info("Planning mixed waypoints...")
            _usd_example.save_ur5e_mixed_waypoints_to_usd(
                world_cfg,
                save_path=save_path,
                pointcloud=pc,
                robot_file=robot_file,
                pose_lists=pose_lists,
                segment_modes=tuple(msg.segment_modes) if msg.segment_modes else (0, 4, 0, 1, 1, 2, 0),
                linear_axes=tuple(msg.linear_axes) if msg.linear_axes else (1, 2),
                grasp_prepose_motion=msg.grasp_prepose_motion,
                attach_cylinder=msg.attach_cylinder,
                attach_after_index=list(msg.attach_after_index),
                detach_after_index=list(msg.detach_after_index),
                cylinder_radius=msg.cylinder_radius if msg.cylinder_radius > 0.0 else 0.05,
                cylinder_height=msg.cylinder_height if msg.cylinder_height > 0.0 else 0.15,
                cylinder_offset_pose=(
                    tuple(msg.cylinder_pose)
                    if len(msg.cylinder_pose) == 7
                    else _usd_example.rotate_pose_local(
                        (0, 0, 0.0, 1, 0, 0, 0),
                        axis_xyz=(0, 1, 0),
                        angle_deg=90,
                    )
                ),
                visualize_frames=False,
            )

            self.get_logger().info(f"Done. Wrote USD: {save_path}")
        except Exception as exc:
            self.get_logger().error(f"Planning failed: {exc}")
        finally:
            self._busy = False


def main():
    rclpy.init()
    node = MixedPlannerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
