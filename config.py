"""Configuration for Alpamayo Robot Server."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RobotConfig:
    """Robot ROS connection settings."""
    ros_host: str = "192.168.10.1"
    ros_port: int = 9090
    cmd_vel_topic: str = "/cmd_vel"
    camera_topic: str = "/camera/image_raw/compressed"


@dataclass
class ModelConfig:
    """Alpamayo model settings."""
    model_path: str = "/mnt/data/lfm_agi/models/alpamayo-r1-10b"
    dtype: str = "bfloat16"
    device: str = "cuda"
    max_generation_length: int = 32
    temperature: float = 0.6
    top_p: float = 0.98


@dataclass
class ControlConfig:
    """Control parameters."""
    max_linear_vel: float = 0.4  # m/s
    max_angular_vel: float = 1.0  # rad/s
    lookahead_steps: int = 10
    linear_scale: float = 0.4
    angular_scale: float = 0.6


@dataclass
class ServerConfig:
    """Web server settings."""
    host: str = "0.0.0.0"
    port: int = 8899
    video_quality: int = 85
    video_fps: int = 30


# Default configurations
ROBOT = RobotConfig()
MODEL = ModelConfig()
CONTROL = ControlConfig()
SERVER = ServerConfig()
