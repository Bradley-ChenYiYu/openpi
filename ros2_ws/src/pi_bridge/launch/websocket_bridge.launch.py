from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument("websocket_host", default_value="127.0.0.1"),
        DeclareLaunchArgument("websocket_port", default_value="8000"),
        DeclareLaunchArgument("api_key", default_value=""),
        DeclareLaunchArgument("robot_config", default_value="tracer"),
        DeclareLaunchArgument("image_topic", default_value="/camera/camera/color/image_raw"),
        DeclareLaunchArgument("odom_topic", default_value="/odom"),
        DeclareLaunchArgument("image_topic_type", default_value="image"),
        DeclareLaunchArgument("inferred_cmd_topic", default_value="/pi_bridge/inferred_cmd_vel"),
        DeclareLaunchArgument("ack_topic", default_value="/pi_bridge/control_ack"),
        DeclareLaunchArgument("raw_response_topic", default_value="/pi_bridge/raw_response"),
        DeclareLaunchArgument("diagnostics_topic", default_value="/pi_bridge/diagnostics"),
        DeclareLaunchArgument("log_level", default_value="info"),
        DeclareLaunchArgument("send_rate_hz", default_value="10.0"),
        DeclareLaunchArgument("sync_tolerance_sec", default_value="0.25"),
        DeclareLaunchArgument("prompt", default_value="do something"),
    ]

    node = Node(
        package="pi_bridge",
        executable="websocket_bridge",
        name="pi_websocket_bridge",
        output="screen",
        arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
        parameters=[
            {
                "websocket_host": LaunchConfiguration("websocket_host"),
                "websocket_port": LaunchConfiguration("websocket_port"),
                "api_key": LaunchConfiguration("api_key"),
                "robot_config": LaunchConfiguration("robot_config"),
                "image_topic": LaunchConfiguration("image_topic"),
                "odom_topic": LaunchConfiguration("odom_topic"),
                "image_topic_type": LaunchConfiguration("image_topic_type"),
                "inferred_cmd_topic": LaunchConfiguration("inferred_cmd_topic"),
                "ack_topic": LaunchConfiguration("ack_topic"),
                "raw_response_topic": LaunchConfiguration("raw_response_topic"),
                "diagnostics_topic": LaunchConfiguration("diagnostics_topic"),
                "send_rate_hz": LaunchConfiguration("send_rate_hz"),
                "sync_tolerance_sec": LaunchConfiguration("sync_tolerance_sec"),
                "prompt": LaunchConfiguration("prompt"),
            }
        ],
    )

    return LaunchDescription(args + [node])
