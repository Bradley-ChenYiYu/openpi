import random

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image


class RandomTestPublisher(Node):
    def __init__(self) -> None:
        super().__init__("pi_bridge_random_test_publisher")

        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("image_width", 224)
        self.declare_parameter("image_height", 224)
        self.declare_parameter("frame_id", "camera_link")
        self.declare_parameter("linear_x_min", -0.5)
        self.declare_parameter("linear_x_max", 0.5)
        self.declare_parameter("angular_z_min", -1.0)
        self.declare_parameter("angular_z_max", 1.0)
        self.declare_parameter("seed", -1)

        image_topic = str(self.get_parameter("image_topic").value)
        odom_topic = str(self.get_parameter("odom_topic").value)
        publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)

        self._image_width = int(self.get_parameter("image_width").value)
        self._image_height = int(self.get_parameter("image_height").value)
        self._frame_id = str(self.get_parameter("frame_id").value)

        self._linear_x_min = float(self.get_parameter("linear_x_min").value)
        self._linear_x_max = float(self.get_parameter("linear_x_max").value)
        self._angular_z_min = float(self.get_parameter("angular_z_min").value)
        self._angular_z_max = float(self.get_parameter("angular_z_max").value)

        seed = int(self.get_parameter("seed").value)
        if seed >= 0:
            random.seed(seed)
            np.random.seed(seed)

        self._image_pub = self.create_publisher(Image, image_topic, 10)
        self._odom_pub = self.create_publisher(Odometry, odom_topic, 10)

        period_sec = 1.0 / max(publish_rate_hz, 1e-3)
        self._timer = self.create_timer(period_sec, self._on_timer)

        self.get_logger().info(
            f"Publishing random test data to image_topic={image_topic} "
            f"odom_topic={odom_topic} at {publish_rate_hz:.2f} Hz"
        )

    def _on_timer(self) -> None:
        stamp = self.get_clock().now().to_msg()

        image = np.random.randint(
            0,
            256,
            size=(self._image_height, self._image_width, 3),
            dtype=np.uint8,
        )

        image_msg = Image()
        image_msg.header.stamp = stamp
        image_msg.header.frame_id = self._frame_id
        image_msg.height = self._image_height
        image_msg.width = self._image_width
        image_msg.encoding = "rgb8"
        image_msg.is_bigendian = 0
        image_msg.step = self._image_width * 3
        image_msg.data = image.tobytes()
        self._image_pub.publish(image_msg)

        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_link"
        odom_msg.twist.twist.linear.x = random.uniform(self._linear_x_min, self._linear_x_max)
        odom_msg.twist.twist.angular.z = random.uniform(self._angular_z_min, self._angular_z_max)
        self._odom_pub.publish(odom_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RandomTestPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
