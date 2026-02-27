#!/usr/bin/env python3
"""Publish an Nx3 NumPy pointcloud to a ROS 2 PointCloud2 topic."""

import argparse
import sys
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header


class NpyPointCloudPublisher(Node):
    def __init__(self, npy_path: Path, topic: str, frame_id: str, repeat: int, rate_hz: float):
        super().__init__("npy_pointcloud_publisher")
        self._npy_path = npy_path
        self._topic = topic
        self._frame_id = frame_id
        self._repeat = repeat
        self._sent = 0

        self._pub = self.create_publisher(PointCloud2, self._topic, 1)
        self._points = self._load_points(self._npy_path)

        period = 1.0 / max(rate_hz, 0.1)
        self._timer = self.create_timer(period, self._publish_once)

        self.get_logger().info(
            f"Publishing {self._points.shape[0]} points from {self._npy_path} to {self._topic}"
        )

    def _load_points(self, path: Path) -> np.ndarray:
        points = np.load(str(path))
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(f"Expected Nx3 array, got shape {points.shape}")
        points = np.asarray(points[:, :3], dtype=np.float32)
        points = points.copy()
        points[:, 2] -= 0.4
        return points

    def _publish_once(self):
        header = Header()
        header.frame_id = self._frame_id
        header.stamp = self.get_clock().now().to_msg()
        msg = point_cloud2.create_cloud_xyz32(header, self._points)
        self._pub.publish(msg)

        self._sent += 1
        if self._sent >= self._repeat:
            self.get_logger().info("Done publishing, shutting down")
            rclpy.shutdown()


def _parse_args(argv):
    parser = argparse.ArgumentParser(description="Publish a .npy pointcloud to ROS 2.")
    parser.add_argument(
        "--npy",
        default="/home/ws/curobo/src/curobo/benno_test/pcls/bookshelf_tall_index1_filtered.npy",
        help="Path to Nx3 .npy file.",
    )
    parser.add_argument("--topic", default="/pointcloud", help="PointCloud2 topic name.")
    parser.add_argument("--frame", default="map", help="TF frame_id to use.")
    parser.add_argument("--repeat", type=int, default=3, help="Number of times to publish.")
    parser.add_argument("--rate", type=float, default=2.0, help="Publish rate (Hz).")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv or sys.argv[1:])
    rclpy.init()
    node = NpyPointCloudPublisher(
        npy_path=Path(args.npy),
        topic=args.topic,
        frame_id=args.frame,
        repeat=max(args.repeat, 1),
        rate_hz=args.rate,
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
