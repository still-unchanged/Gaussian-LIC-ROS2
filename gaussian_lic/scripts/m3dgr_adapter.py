#!/usr/bin/python3
"""Bridge common M3DGR message encodings to Gaussian-LIC's fixed topics.

Gaussian-LIC consumes PoseStamped and metric 32FC1 depth.  M3DGR bags often
store odometry as nav_msgs/Odometry and depth as 16UC1 (millimetres), so this
node normalizes those streams while RGB and PointCloud2 use launch remaps.
"""

import rclpy
from rclpy.node import Node
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry


class M3DGRAdapter(Node):
    def __init__(self):
        super().__init__("m3dgr_adapter")
        self.pose_topic = self.declare_parameter("pose_topic", "/m3dgr/odom").value
        self.pose_type = self.declare_parameter("pose_type", "odometry").value.lower()
        self.depth_topic = self.declare_parameter(
            "depth_topic", "/m3dgr/camera/depth/image_raw").value
        self.depth_scale = float(self.declare_parameter("depth_scale", 1000.0).value)
        if self.depth_scale <= 0.0:
            raise ValueError("depth_scale must be positive")

        self.bridge = CvBridge()
        self.pose_pub = self.create_publisher(PoseStamped, "/pose_for_gs", 100)
        self.depth_pub = self.create_publisher(Image, "/depth_for_gs", 100)
        if self.pose_type in ("odometry", "odom"):
            self.pose_sub = self.create_subscription(
                Odometry, self.pose_topic, self.odom_callback, 100)
        elif self.pose_type in ("pose_stamped", "pose"):
            self.pose_sub = self.create_subscription(
                PoseStamped, self.pose_topic, self.pose_callback, 100)
        else:
            raise ValueError("pose_type must be 'odometry' or 'pose_stamped'")
        self.depth_sub = self.create_subscription(
            Image, self.depth_topic, self.depth_callback, 100)

    def odom_callback(self, msg):
        out = PoseStamped()
        out.header = msg.header
        out.pose = msg.pose.pose
        self.pose_pub.publish(out)

    def pose_callback(self, msg):
        self.pose_pub.publish(msg)

    def depth_callback(self, msg):
        try:
            raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as exc:
            self.get_logger().warning(
                f"M3DGR depth conversion failed: {exc}", throttle_duration_sec=5.0)
            return
        if raw.ndim != 2:
            self.get_logger().warning(
                f"Expected single-channel depth, got {raw.shape}", throttle_duration_sec=5.0)
            return
        if raw.dtype == np.uint16 or raw.dtype == np.uint32:
            depth_m = raw.astype(np.float32) / self.depth_scale
        else:
            depth_m = raw.astype(np.float32)
        depth_m[~np.isfinite(depth_m)] = 0.0
        depth_m[depth_m < 0.0] = 0.0
        out = self.bridge.cv2_to_imgmsg(depth_m, encoding="32FC1")
        out.header = msg.header
        self.depth_pub.publish(out)


def main():
    rclpy.init()
    node = None
    try:
        node = M3DGRAdapter()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
