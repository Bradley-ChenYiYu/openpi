import rclpy
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import String
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy
from .base_bridge import PiWebsocketBridgeBase

class TracerFrontRightBridge(PiWebsocketBridgeBase):
    def setup_robot_config(self):
        self.declare_parameter("front_image_topic", "/camera/front/color/image_raw")
        self.declare_parameter("right_image_topic", "/camera/right/color/image_raw")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("inferred_cmd_topic", "/pi_bridge/inferred_cmd_vel")

    def get_required_observations(self):
        return {"observation/front_image": "front_image_topic", "observation/right_image": "right_image_topic"}

    def setup_subscriptions(self):
        self.create_subscription(Odometry, self.get_parameter("odom_topic").value, self._on_odom, self._queue_size)
        for key, param in self.get_required_observations().items():
            self.create_subscription(Image, self.get_parameter(param).value, lambda m, k=key: self._on_image(m, k), self._queue_size)

    def setup_publishers(self):
        self._inferred_cmd_pub = self.create_publisher(Twist, self.get_parameter("inferred_cmd_topic").value, QoSProfile(history=HistoryPolicy.KEEP_ALL, depth=1, reliability=ReliabilityPolicy.RELIABLE))
        self._ack_pub = self.create_publisher(String, "/pi_bridge/control_ack", self._queue_size)

    def prepare_observation_state(self, odom):
        return np.asarray([odom.linear_x, odom.angular_z], dtype=np.float32)

    def apply_action(self, action):
        msg = Twist(); msg.linear.x, msg.angular.z = action
        self._inferred_cmd_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = TracerFrontRightBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
