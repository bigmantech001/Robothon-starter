"""
PROMETHEUS Configuration — Centralized parameters for the surgical robot simulation.

All tunable parameters are defined here for clean separation of concerns.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from pathlib import Path


# ═══════════════════ PATH CONFIGURATION ═══════════════════

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "prometheus.xml"


# ═══════════════════ PHYSICS CONFIGURATION ═══════════════════

@dataclass
class PhysicsConfig:
    """MuJoCo physics simulation parameters."""
    timestep: float = 0.002
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)
    solver_iterations: int = 50
    solver_tolerance: float = 1e-10


# ═══════════════════ CONTROLLER CONFIGURATION ═══════════════════

@dataclass
class ControllerConfig:
    """PD controller gains and limits."""
    # Arm joint PD gains [Kp, Kd]
    arm_kp: float = 200.0
    arm_kd: float = 20.0

    # Finger joint PD gains [Kp, Kd]
    finger_kp: float = 5.0
    finger_kd: float = 0.5

    # Wrist PD gains (finer control)
    wrist_kp: float = 50.0
    wrist_kd: float = 5.0

    # Maximum joint velocity (rad/s)
    max_joint_velocity: float = 2.0

    # Trajectory interpolation speed
    trajectory_speed: float = 0.3  # fraction of max speed

    # Control frequency relative to simulation
    control_decimation: int = 5  # apply control every N sim steps


# ═══════════════════ JOINT NAME MAPPINGS ═══════════════════

# Right arm joint names (ordered)
RIGHT_ARM_JOINTS = [
    "r_shoulder_yaw", "r_shoulder_pitch", "r_shoulder_roll",
    "r_elbow_flex",
    "r_wrist_pronation", "r_wrist_flex", "r_wrist_dev",
]

# Left arm joint names (ordered)
LEFT_ARM_JOINTS = [
    "l_shoulder_yaw", "l_shoulder_pitch", "l_shoulder_roll",
    "l_elbow_flex",
    "l_wrist_pronation", "l_wrist_flex", "l_wrist_dev",
]

# Right hand joint names (ordered)
RIGHT_HAND_JOINTS = [
    # Thumb (4)
    "r_thumb_cmc_flex", "r_thumb_cmc_abd", "r_thumb_mcp_flex", "r_thumb_ip_flex",
    # Index (4)
    "r_index_abd", "r_index_mcp_flex", "r_index_pip_flex", "r_index_dip_flex",
    # Middle (4)
    "r_middle_abd", "r_middle_mcp_flex", "r_middle_pip_flex", "r_middle_dip_flex",
    # Ring (4)
    "r_ring_abd", "r_ring_mcp_flex", "r_ring_pip_flex", "r_ring_dip_flex",
    # Little (4)
    "r_little_abd", "r_little_mcp_flex", "r_little_pip_flex", "r_little_dip_flex",
]

# Left hand joint names (ordered)
LEFT_HAND_JOINTS = [
    # Thumb (4)
    "l_thumb_cmc_flex", "l_thumb_cmc_abd", "l_thumb_mcp_flex", "l_thumb_ip_flex",
    # Index (4)
    "l_index_abd", "l_index_mcp_flex", "l_index_pip_flex", "l_index_dip_flex",
    # Middle (4)
    "l_middle_abd", "l_middle_mcp_flex", "l_middle_pip_flex", "l_middle_dip_flex",
    # Ring (4)
    "l_ring_abd", "l_ring_mcp_flex", "l_ring_pip_flex", "l_ring_dip_flex",
    # Little (4)
    "l_little_abd", "l_little_mcp_flex", "l_little_pip_flex", "l_little_dip_flex",
]

# Actuator name mappings (joint_name + "_act")
RIGHT_ARM_ACTUATORS = [j + "_act" for j in RIGHT_ARM_JOINTS]
LEFT_ARM_ACTUATORS = [j + "_act" for j in LEFT_ARM_JOINTS]
RIGHT_HAND_ACTUATORS = [j + "_act" for j in RIGHT_HAND_JOINTS]
LEFT_HAND_ACTUATORS = [j + "_act" for j in LEFT_HAND_JOINTS]

# All actuators combined
ALL_ACTUATORS = (RIGHT_ARM_ACTUATORS + RIGHT_HAND_ACTUATORS +
                 LEFT_ARM_ACTUATORS + LEFT_HAND_ACTUATORS)

# Touch sensor names
RIGHT_TOUCH_SENSORS = [
    "r_thumb_contact", "r_index_contact", "r_middle_contact",
    "r_ring_contact", "r_little_contact", "r_palm_contact",
]
LEFT_TOUCH_SENSORS = [
    "l_thumb_contact", "l_index_contact", "l_middle_contact",
    "l_ring_contact", "l_little_contact", "l_palm_contact",
]


# ═══════════════════ HAND POSE PRESETS ═══════════════════

@dataclass
class HandPoses:
    """
    Predefined hand poses for surgical manipulation.
    Each pose is a dict of joint_name -> target_angle (radians).
    """

    @staticmethod
    def open_hand() -> Dict[str, float]:
        """Fully open hand — all fingers extended."""
        return {
            "thumb_cmc_flex": 0.0, "thumb_cmc_abd": 0.0,
            "thumb_mcp_flex": 0.0, "thumb_ip_flex": 0.0,
            "index_abd": 0.0, "index_mcp_flex": 0.0,
            "index_pip_flex": 0.0, "index_dip_flex": 0.0,
            "middle_abd": 0.0, "middle_mcp_flex": 0.0,
            "middle_pip_flex": 0.0, "middle_dip_flex": 0.0,
            "ring_abd": 0.0, "ring_mcp_flex": 0.0,
            "ring_pip_flex": 0.0, "ring_dip_flex": 0.0,
            "little_abd": 0.0, "little_mcp_flex": 0.0,
            "little_pip_flex": 0.0, "little_dip_flex": 0.0,
        }

    @staticmethod
    def pinch_grip() -> Dict[str, float]:
        """Surgical pinch grip — thumb + index for needle grasping."""
        return {
            "thumb_cmc_flex": 0.8, "thumb_cmc_abd": 0.3,
            "thumb_mcp_flex": 0.6, "thumb_ip_flex": 0.4,
            "index_abd": -0.1, "index_mcp_flex": 0.9,
            "index_pip_flex": 1.0, "index_dip_flex": 0.7,
            "middle_abd": 0.0, "middle_mcp_flex": 1.2,
            "middle_pip_flex": 1.4, "middle_dip_flex": 0.9,
            "ring_abd": 0.0, "ring_mcp_flex": 1.2,
            "ring_pip_flex": 1.4, "ring_dip_flex": 0.9,
            "little_abd": 0.0, "little_mcp_flex": 1.2,
            "little_pip_flex": 1.4, "little_dip_flex": 0.9,
        }

    @staticmethod
    def tripod_grip() -> Dict[str, float]:
        """Tripod grip — thumb + index + middle for stable holding."""
        return {
            "thumb_cmc_flex": 0.7, "thumb_cmc_abd": 0.2,
            "thumb_mcp_flex": 0.5, "thumb_ip_flex": 0.3,
            "index_abd": -0.05, "index_mcp_flex": 0.8,
            "index_pip_flex": 0.9, "index_dip_flex": 0.6,
            "middle_abd": 0.05, "middle_mcp_flex": 0.8,
            "middle_pip_flex": 0.9, "middle_dip_flex": 0.6,
            "ring_abd": 0.0, "ring_mcp_flex": 1.3,
            "ring_pip_flex": 1.5, "ring_dip_flex": 1.0,
            "little_abd": 0.0, "little_mcp_flex": 1.3,
            "little_pip_flex": 1.5, "little_dip_flex": 1.0,
        }

    @staticmethod
    def tissue_retraction() -> Dict[str, float]:
        """Spread fingers for tissue retraction / stabilization."""
        return {
            "thumb_cmc_flex": 0.5, "thumb_cmc_abd": -0.3,
            "thumb_mcp_flex": 0.2, "thumb_ip_flex": 0.1,
            "index_abd": -0.25, "index_mcp_flex": 0.3,
            "index_pip_flex": 0.2, "index_dip_flex": 0.1,
            "middle_abd": 0.0, "middle_mcp_flex": 0.3,
            "middle_pip_flex": 0.2, "middle_dip_flex": 0.1,
            "ring_abd": 0.15, "ring_mcp_flex": 0.3,
            "ring_pip_flex": 0.2, "ring_dip_flex": 0.1,
            "little_abd": 0.25, "little_mcp_flex": 0.3,
            "little_pip_flex": 0.2, "little_dip_flex": 0.1,
        }

    @staticmethod
    def power_grasp() -> Dict[str, float]:
        """Full hand close — power grasp for pulling thread."""
        return {
            "thumb_cmc_flex": 1.0, "thumb_cmc_abd": 0.3,
            "thumb_mcp_flex": 0.8, "thumb_ip_flex": 0.6,
            "index_abd": 0.0, "index_mcp_flex": 1.3,
            "index_pip_flex": 1.5, "index_dip_flex": 1.0,
            "middle_abd": 0.0, "middle_mcp_flex": 1.3,
            "middle_pip_flex": 1.5, "middle_dip_flex": 1.0,
            "ring_abd": 0.0, "ring_mcp_flex": 1.3,
            "ring_pip_flex": 1.5, "ring_dip_flex": 1.0,
            "little_abd": 0.0, "little_mcp_flex": 1.3,
            "little_pip_flex": 1.5, "little_dip_flex": 1.0,
        }


# ═══════════════════ TASK WAYPOINTS ═══════════════════

@dataclass
class SuturingConfig:
    """
    Suturing task configuration — defines the surgical procedure steps.
    """
    # Suture entry/exit points on tissue (x, y, z in world frame)
    # These correspond to the site positions defined in the MJCF
    suture_points: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = field(
        default_factory=lambda: [
            # (entry_point, exit_point) pairs
            ((-0.05 + 0.10, -0.45 + 0.06, 0.79), (0.05 - 0.10, -0.45 + 0.06, 0.79)),
            ((-0.05 + 0.10, -0.45 + 0.02, 0.79), (0.05 - 0.10, -0.45 + 0.02, 0.79)),
            ((-0.05 + 0.10, -0.45 - 0.02, 0.79), (0.05 - 0.10, -0.45 - 0.02, 0.79)),
            ((-0.05 + 0.10, -0.45 - 0.06, 0.79), (0.05 - 0.10, -0.45 - 0.06, 0.79)),
        ]
    )

    # Timing for each phase (in simulation seconds)
    approach_duration: float = 2.0
    grasp_duration: float = 1.5
    pierce_duration: float = 2.0
    pull_duration: float = 2.0
    tension_duration: float = 1.5
    retract_duration: float = 1.0

    # Heights for approach/retract
    approach_height: float = 0.10  # meters above tissue
    retract_height: float = 0.08


# ═══════════════════ CAMERA CONFIGURATION ═══════════════════

@dataclass
class CameraConfig:
    """Camera settings for recording and visualization."""
    # Named cameras in the MJCF model
    camera_names: List[str] = field(
        default_factory=lambda: [
            "overview", "top_down", "hands_close",
            "right_side", "left_side", "detail",
        ]
    )

    # Default camera for viewer
    default_camera: str = "overview"

    # Recording resolution
    width: int = 1280
    height: int = 720
    fps: int = 30

    # Camera cut schedule for demo video (camera_name, start_time_seconds)
    demo_cuts: List[Tuple[str, float]] = field(
        default_factory=lambda: [
            ("overview", 0.0),       # Opening shot — full scene
            ("hands_close", 4.0),    # Approach needle
            ("detail", 10.0),        # Grasp needle close-up
            ("right_side", 16.0),    # Lift and position
            ("hands_close", 22.0),   # Approach tissue
            ("detail", 28.0),        # Pierce tissue
            ("overview", 34.0),      # Pull-back wide shot
            ("top_down", 40.0),      # Top-down during thread pull
            ("hands_close", 46.0),   # Tension and next stitch
            ("overview", 52.0),      # Final wide shot
        ]
    )


# ═══════════════════ TELEOPERATION CONFIGURATION ═══════════════════

@dataclass
class TeleoperationConfig:
    """Keyboard teleoperation mappings."""
    # Translation step size (meters per keypress)
    translation_step: float = 0.005
    # Rotation step size (radians per keypress)
    rotation_step: float = 0.05
    # Finger open/close step (radians per keypress)
    finger_step: float = 0.1


# ═══════════════════ MASTER CONFIG ═══════════════════

@dataclass
class PrometheusConfig:
    """Master configuration combining all sub-configs."""
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    suturing: SuturingConfig = field(default_factory=SuturingConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    teleop: TeleoperationConfig = field(default_factory=TeleoperationConfig)
    hand_poses: HandPoses = field(default_factory=HandPoses)

    # Model path
    model_path: str = str(MODEL_PATH)


# Default config instance
DEFAULT_CONFIG = PrometheusConfig()
