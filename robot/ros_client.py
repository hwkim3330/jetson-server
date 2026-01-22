"""ROS WebSocket client for robot communication."""

import base64
import threading
from typing import Callable, Optional

import cv2
import numpy as np
import roslibpy


class ROSClient:
    """Client for ROS robot communication via WebSocket."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.ros: Optional[roslibpy.Ros] = None
        self.cmd_vel_pub: Optional[roslibpy.Topic] = None

        self._frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected and self.ros is not None and self.ros.is_connected

    def connect(self) -> bool:
        """Connect to ROS via WebSocket."""
        try:
            self.ros = roslibpy.Ros(host=self.host, port=self.port)
            self.ros.run()
            self._connected = self.ros.is_connected
            return self._connected
        except Exception as e:
            print(f"ROS connection error: {e}")
            return False

    def disconnect(self):
        """Disconnect from ROS."""
        if self.ros:
            self.ros.terminate()
            self._connected = False

    def setup_publishers(self, cmd_vel_topic: str = "/cmd_vel"):
        """Setup ROS publishers."""
        if not self.ros:
            return
        self.cmd_vel_pub = roslibpy.Topic(
            self.ros, cmd_vel_topic, "geometry_msgs/Twist"
        )

    def subscribe_camera(self, topic: str, callback: Optional[Callable] = None):
        """Subscribe to camera topic."""
        if not self.ros:
            return

        def on_image(msg):
            try:
                data = msg.get("data", "")
                img_bytes = base64.b64decode(data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    with self._frame_lock:
                        self._frame = img
                    if callback:
                        callback(img)
            except Exception:
                pass

        topic_sub = roslibpy.Topic(
            self.ros, topic, "sensor_msgs/CompressedImage"
        )
        topic_sub.subscribe(on_image)

    def get_frame(self) -> Optional[np.ndarray]:
        """Get latest camera frame."""
        with self._frame_lock:
            return self._frame.copy() if self._frame is not None else None

    def send_velocity(self, linear: float, angular: float):
        """Send velocity command to robot."""
        if not self.cmd_vel_pub:
            return

        msg = roslibpy.Message({
            "linear": {"x": float(linear), "y": 0.0, "z": 0.0},
            "angular": {"x": 0.0, "y": 0.0, "z": float(angular)}
        })
        self.cmd_vel_pub.publish(msg)

    def stop(self):
        """Stop the robot."""
        self.send_velocity(0.0, 0.0)
