#!/usr/bin/env python3
"""Publish GenerateMotion messages periodically for testing."""

import argparse
import sys

import rclpy
from rclpy.node import Node

from generate_motion_msgs.msg import GenerateMotion


class GenerateMotionPublisher(Node):
    def __init__(self, topic: str, period_s: float):
        super().__init__("generate_motion_publisher")
        self._pub = self.create_publisher(GenerateMotion, topic, 10)
        self._period_s = period_s
        self._counter = 0
        self._timer = self.create_timer(self._period_s, self._on_timer)
        self.get_logger().info(f"Publishing GenerateMotion on {topic} every {period_s}s")

    def _on_timer(self):
        msg = GenerateMotion()
        msg.robot_file = "ur5e_robotiq_2f_85.yml"
        msg.pose_lists = [
            0.0, 0.5, 0.72, 0.5, -0.5, 0.5, 0.5,
            0.0, 0.63, 0.72, 0.5, -0.5, 0.5, 0.5,
            0.0, 0.5, 0.47, 0.5, -0.5, 0.5, 0.5,
            0.0, 0.63, 0.47, 0.5, -0.5, 0.5, 0.5,
            0.0, 0.63, 0.45, 0.5, -0.5, 0.5, 0.5,
            0.0, 0.5, 0.47, 0.5, -0.5, 0.5, 0.5,
            0.2, 0.5, 0.47, 0.5, -0.5, 0.5, 0.5,
        ]
        msg.segment_modes = [0, 4, 0, 1, 1, 2, 0]
        msg.linear_axes = [1, 2]
        msg.attach_cylinder = True
        msg.attach_after_index = [1]
        msg.detach_after_index = [4]
        msg.cylinder_radius = 0.05
        msg.cylinder_height = 0.10
        msg.cylinder_pose = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        msg.grasp_prepose_motion = True
        self._pub.publish(msg)
        self.get_logger().info("Published GenerateMotion message")
        self._counter += 1


def _parse_args(argv):
    parser = argparse.ArgumentParser(description="Publish GenerateMotion messages.")
    parser.add_argument("--topic", default="/generate_motion", help="Topic name.")
    parser.add_argument("--period", type=float, default=10.0, help="Publish period in seconds.")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv or sys.argv[1:])
    rclpy.init()
    node = GenerateMotionPublisher(topic=args.topic, period_s=max(args.period, 0.1))
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
