#!/usr/bin/env python3
"""
Voice Command Node - Receives voice commands from a WebSocket server and updates the prompt parameter.

This node connects to a WebSocket server that provides voice commands (e.g., from whisper.cpp)
and updates the 'prompt' parameter of the pi_websocket_bridge node via ROS2 parameter service.
"""

import json
import re
import sys
import threading
from typing import Dict, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, ReliabilityPolicy, QoSProfile
from rcl_interfaces.srv import SetParameters
from std_msgs.msg import String

try:
    import websocket
except ImportError:
    print("Error: websocket-client not installed")
    print("Install with: pip install websocket-client")
    sys.exit(1)


# Command mapping for voice to task transformation
GO_TO_COMMANDS: Dict[str, str] = {
    "brown cups": "Go to the table with brown cups.",
    "red cups": "Go to the table with big red cups.",
    "sandwich": "Go to the table with a sandwich.",
    "white jug": "Go to the table with a white jug.",
}

# "Go back" command mappings (requires previous location state)
GO_BACK_COMMANDS: Dict[str, str] = {
    "brown cups": "From the table with brown cups, go to the table with a black cloth.",
    "red cups": "From the table with big red cups, go to the table with a black cloth.",
    "sandwich": "From the table with a sandwich, go to the table with a black cloth.",
    "white jug": "From the table with a white jug, go to the table with a black cloth.",
}


class CommandStateMachine:
    """State machine for transforming voice commands into task descriptions."""
    
    def __init__(self) -> None:
        self._current_location: Optional[str] = None
    
    def transform_command(self, command: str) -> Optional[str]:
        """
        Transform a voice command into a standardized task description.
        
        Args:
            command: The raw voice command string
            
        Returns:
            Transformed task description, or None if command cannot be parsed
        """
        command_lower = command.lower().strip()
        
        # Check for "go back" pattern
        if "go back" in command_lower or "go back to" in command_lower:
            return self._handle_go_back(command_lower)
        
        # Check for "go to" pattern
        if "go to" in command_lower:
            return self._handle_go_to(command_lower)
        
        # Direct location match
        return self._handle_direct_location(command_lower)
    
    def _handle_go_to(self, command: str) -> Optional[str]:
        """Handle 'go to <location>' commands."""
        for location, task in GO_TO_COMMANDS.items():
            if location in command:
                self._current_location = location
                return task
        return None
    
    def _handle_go_back(self, command: str) -> Optional[str]:
        """Handle 'go back to <location>' or 'go back' commands."""
        for location, task in GO_BACK_COMMANDS.items():
            if location in command:
                return task
        
        # If no location specified, use current location
        if self._current_location and self._current_location in GO_BACK_COMMANDS:
            return GO_BACK_COMMANDS[self._current_location]
        
        return None
    
    def _handle_direct_location(self, command: str) -> Optional[str]:
        """Handle direct location mentions (assumed to be 'go to')."""
        for location, task in GO_TO_COMMANDS.items():
            if location in command:
                self._current_location = location
                return task
        return None
    
    def reset(self) -> None:
        """Reset the state machine."""
        self._current_location = None


class VoiceCommandNode(Node):
    """Node that listens to WebSocket for voice commands and updates ROS2 parameters."""

    def __init__(self) -> None:
        super().__init__("voice_command_node")

        # Declare parameters
        self.declare_parameter("websocket_host", "0.0.0.0")
        self.declare_parameter("websocket_port", 9001)
        self.declare_parameter("target_node", "pi_websocket_bridge")
        self.declare_parameter("reconnect_interval_sec", 5.0)
        self.declare_parameter("max_reconnect_interval_sec", 30.0)
        self.declare_parameter("command_probability_threshold", 0.2)
        self.declare_parameter("enable_auto_update", True)
        self.declare_parameter("enable_request_action_flow", True)

        # Read parameters
        self._websocket_host = str(self.get_parameter("websocket_host").value)
        self._websocket_port = int(self.get_parameter("websocket_port").value)
        self._target_node = str(self.get_parameter("target_node").value)
        self._reconnect_interval_sec = float(self.get_parameter("reconnect_interval_sec").value)
        self._max_reconnect_interval_sec = float(self.get_parameter("max_reconnect_interval_sec").value)
        self._probability_threshold = float(self.get_parameter("command_probability_threshold").value)
        self._enable_auto_update = bool(self.get_parameter("enable_auto_update").value)
        self._enable_request_action_flow = bool(self.get_parameter("enable_request_action_flow").value)

        # Command state machine
        self._command_state_machine = CommandStateMachine()

        # WebSocket state
        self._ws: websocket.WebSocketApp | None = None
        self._ws_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._ws_thread: threading.Thread | None = None
        self._connected = False

        # Setup parameter client to update pi_websocket_bridge node
        self._param_client = self.create_client(
            SetParameters,
            f"/{self._target_node}/set_parameters"
        )
        
        # Wait for parameter service
        if not self._param_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().warning(
                f"Parameter service for node '{self._target_node}' not available. "
                "Auto-update disabled until service is available."
            )
            self._enable_auto_update = False
        else:
            self.get_logger().info(f"Connected to parameter service for node '{self._target_node}'")

        # Diagnostic publisher
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._status_pub = self.create_publisher(String, "/voice_command/status", qos)

        # Start WebSocket client in a separate thread
        self._start_websocket()

        # Status update timer
        self.create_timer(2.0, self._publish_status)

        self.get_logger().info(f"Voice Command Node started. Connecting to {self._websocket_host}:{self._websocket_port}")

    def _start_websocket(self) -> None:
        """Start WebSocket client in a separate thread."""
        self._stop_event.clear()
        ws_url = f"ws://{self._websocket_host}:{self._websocket_port}"
        
        self._ws = websocket.WebSocketApp(
            ws_url,
            on_open=self._on_ws_open,
            on_message=self._on_ws_message,
            on_error=self._on_ws_error,
            on_close=self._on_ws_close,
        )

        self._ws_thread = threading.Thread(
            target=self._ws.run_forever,
            name="voice-ws-client",
            daemon=True
        )
        self._ws_thread.start()

    def _on_ws_open(self, ws) -> None:
        """Handle WebSocket connection open."""
        self._connected = True
        self.get_logger().info("Connected to WebSocket server")
        self._publish_status()
        
        # Send a ping to test the connection
        ws.send("ping")

    def _on_ws_message(self, ws, message) -> None:
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(message)
            msg_type = data.get("type", "unknown")

            if msg_type == "welcome":
                self.get_logger().info(f"Connected to voice command server. Clients: {data.get('clients', 'N/A')}")
            
            elif msg_type == "command":
                raw_command = data.get("command", "")
                probability = data.get("probability", 0.0)
                mode = data.get("mode", "N/A")
                duration_ms = data.get("duration_ms", 0)
                timestamp = data.get("timestamp", 0)

                self.get_logger().info(
                    f"Voice Command Detected! | Mode: {mode} | Command: '{raw_command}' | "
                    f"Probability: {probability*100:.1f}% | Duration: {duration_ms}ms"
                )

                # Transform command through state machine
                transformed_command = self._command_state_machine.transform_command(raw_command)
                
                if transformed_command:
                    self.get_logger().info(f"✓ Command transformed: '{raw_command}' → '{transformed_command}'")
                else:
                    self.get_logger().warning(f"✗ Could not transform command: '{raw_command}'")
                    transformed_command = raw_command  # Fall back to raw command

                # Update prompt parameter if probability is above threshold
                if not transformed_command:
                    self.get_logger().warning("Empty command received, skipping update")
                elif probability < self._probability_threshold:
                    self.get_logger().warning(
                        f"Command probability {probability*100:.1f}% below threshold {self._probability_threshold*100:.1f}%, skipping update"
                    )
                elif not self._enable_auto_update:
                    self.get_logger().warning("Auto-update disabled, skipping parameter update")
                else:
                    self._update_prompt_parameter(transformed_command)

            elif msg_type == "telemetry":
                pass  # Silently handle telemetry messages
            
            else:
                self.get_logger().debug(f"Unknown message type: {msg_type}")

        except json.JSONDecodeError:
            self.get_logger().warning(f"Received non-JSON message: {message}")

    def _on_ws_error(self, ws, error) -> None:
        """Handle WebSocket errors."""
        self._connected = False
        self.get_logger().error(f"WebSocket error: {error}")

    def _on_ws_close(self, ws, close_status_code, close_msg) -> None:
        """Handle WebSocket connection close."""
        self._connected = False
        self.get_logger().info(f"WebSocket connection closed (code={close_status_code}, msg={close_msg})")
        self._schedule_reconnect()

    def _schedule_reconnect(self) -> None:
        """Schedule a reconnection attempt with exponential backoff."""
        import time
        
        def reconnect_with_backoff():
            backoff = self._reconnect_interval_sec
            while not self._stop_event.is_set():
                if self._connected:
                    break
                self.get_logger().warning(f"Reconnecting in {backoff:.1f}s...")
                self._stop_event.wait(backoff)
                if not self._stop_event.is_set():
                    with self._ws_lock:
                        if self._ws and not self._connected:
                            self._ws = None
                    self._start_websocket()
                    break
                backoff = min(backoff * 2.0, self._max_reconnect_interval_sec)
        
        reconnect_thread = threading.Thread(target=reconnect_with_backoff, daemon=True)
        reconnect_thread.start()

    def _update_prompt_parameter(self, prompt: str) -> None:
        """Update the 'prompt' parameter of the target node."""
        if not self._param_client.service_is_ready():
            self.get_logger().warning("Parameter service not ready, cannot update prompt")
            return

        try:
            # Use the simpler rclpy method - call set_parameters directly
            from rclpy.parameter import Parameter
            
            # Create a request with Parameter objects
            request = SetParameters.Request()
            request.parameters = [
                Parameter("prompt", value=prompt).to_parameter_msg(),
                Parameter("enable_request_action_flow", value=self._enable_request_action_flow).to_parameter_msg(),
            ]
            
            # Call async with callback
            future = self._param_client.call_async(request)
            future.add_done_callback(lambda f: self._on_param_update_result(f, prompt))
            
        except Exception as e:
            self.get_logger().error(f"Failed to update parameter: {e}")

    def _on_param_update_result(self, future, prompt: str) -> None:
        """Handle parameter update result."""
        try:
            result = future.result()
            if result.results and result.results[0].successful:
                self.get_logger().info(f"✓ Updated prompt to: '{prompt}'")
            else:
                self.get_logger().warning(f"✗ Failed to update prompt: {result.results[0].reason}")
        except Exception as e:
            self.get_logger().error(f"Parameter update failed: {e}")

    def _publish_status(self) -> None:
        """Publish status diagnostic message."""
        status = {
            "connected": self._connected,
            "websocket_host": self._websocket_host,
            "websocket_port": self._websocket_port,
            "target_node": self._target_node,
            "auto_update_enabled": self._enable_auto_update,
        }
        msg = String()
        msg.data = json.dumps(status)
        self._status_pub.publish(msg)

    def destroy_node(self) -> None:
        """Cleanup on node destruction."""
        self._stop_event.set()
        
        with self._ws_lock:
            if self._ws:
                self._ws.close()
        
        if self._ws_thread:
            self._ws_thread.join(timeout=3.0)
        
        super().destroy_node()


def main(args=None) -> None:
    """Main entry point."""
    rclpy.init(args=args)
    node = VoiceCommandNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
