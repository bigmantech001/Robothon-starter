"""
PROMETHEUS Teleoperation — Keyboard-based interactive control for the surgical robot.

Controls:
  ┌─────────────────────────────────────────────────────────┐
  │  ARM CONTROL (active arm)                               │
  │  W/S  — Move forward/backward     Q/E — Rotate wrist   │
  │  A/D  — Move left/right           R/F — Move up/down   │
  │  Z/X  — Elbow flex/extend                               │
  │                                                         │
  │  HAND CONTROL                                           │
  │  1 — Open hand        2 — Pinch grip                    │
  │  3 — Tripod grip      4 — Power grasp                   │
  │  5 — Tissue retraction                                  │
  │                                                         │
  │  MODE CONTROL                                           │
  │  TAB   — Switch active arm (left/right)                 │
  │  SPACE — Toggle autonomous/teleop mode                  │
  │  G     — Start autonomous suturing                      │
  │  ESC   — Exit                                           │
  └─────────────────────────────────────────────────────────┘
"""

import numpy as np
from typing import Optional

from .config import TeleoperationConfig, HandPoses
from .controller import SuturingController
from .utils import clamp


class TeleoperationHandler:
    """
    Processes keyboard input and translates to robot joint commands.
    Works in conjunction with the SuturingController.
    """

    def __init__(self, controller: SuturingController,
                 config: Optional[TeleoperationConfig] = None):
        """
        Initialize teleoperation handler.

        Args:
            controller: The suturing controller to command
            config: Teleoperation configuration
        """
        self.controller = controller
        self.config = config or TeleoperationConfig()
        self.hand_poses = HandPoses()

        # Active arm selection (True = right, False = left)
        self.right_arm_active = True

        # Key state tracking
        self.keys_pressed = set()

    @property
    def active_arm_name(self) -> str:
        """Get the name of the currently active arm."""
        return "RIGHT" if self.right_arm_active else "LEFT"

    def handle_key_press(self, key: int):
        """
        Process a key press event from MuJoCo viewer.

        Args:
            key: Key code from glfw
        """
        self.keys_pressed.add(key)
        self._process_discrete_keys(key)

    def handle_key_release(self, key: int):
        """Process a key release event."""
        self.keys_pressed.discard(key)

    def _process_discrete_keys(self, key: int):
        """Handle one-shot key presses (hand poses, mode switches)."""
        # Import glfw key constants (mapped to ASCII for simplicity)
        # MuJoCo viewer uses glfw, key codes match ASCII for letters

        # TAB (9) — Switch active arm
        if key == 9:  # Tab
            self.right_arm_active = not self.right_arm_active
            return

        # SPACE (32) — Toggle autonomous/teleop
        if key == 32:  # Space
            self.controller.teleop_active = not self.controller.teleop_active
            return

        # G (71) — Start autonomous mode
        if key == 71 or key == 103:  # 'G' or 'g'
            self.controller.start_autonomous()
            return

        # Hand pose shortcuts
        hand_target = None

        if key == 49 or key == 257:  # '1'
            hand_target = self.hand_poses.open_hand()
        elif key == 50 or key == 258:  # '2'
            hand_target = self.hand_poses.pinch_grip()
        elif key == 51 or key == 259:  # '3'
            hand_target = self.hand_poses.tripod_grip()
        elif key == 52 or key == 260:  # '4'
            hand_target = self.hand_poses.power_grasp()
        elif key == 53 or key == 261:  # '5'
            hand_target = self.hand_poses.tissue_retraction()

        if hand_target is not None:
            arr = self.controller._pose_to_array(hand_target)
            if self.right_arm_active:
                self.controller.right_hand_target = arr
            else:
                self.controller.left_hand_target = arr

    def update_continuous(self):
        """
        Process continuous key holds for arm movement.
        Called every control step.
        """
        if not self.controller.teleop_active:
            return

        step = self.config.translation_step
        rot_step = self.config.rotation_step

        # Get reference to active arm target
        if self.right_arm_active:
            target = self.controller.right_arm_target
        else:
            target = self.controller.left_arm_target

        # Map held keys to joint adjustments
        # W/S (87/83) — Shoulder pitch (forward/backward)
        if 87 in self.keys_pressed or 119 in self.keys_pressed:  # W
            target[1] += step * 2
        if 83 in self.keys_pressed or 115 in self.keys_pressed:  # S
            target[1] -= step * 2

        # A/D (65/68) — Shoulder yaw (left/right)
        if 65 in self.keys_pressed or 97 in self.keys_pressed:  # A
            target[0] -= step * 2
        if 68 in self.keys_pressed or 100 in self.keys_pressed:  # D
            target[0] += step * 2

        # R/F (82/70) — Shoulder roll (up/down)
        if 82 in self.keys_pressed or 114 in self.keys_pressed:  # R
            target[2] += step * 2
        if 70 in self.keys_pressed or 102 in self.keys_pressed:  # F
            target[2] -= step * 2

        # Z/X (90/88) — Elbow flex/extend
        if 90 in self.keys_pressed or 122 in self.keys_pressed:  # Z
            target[3] += step * 2
        if 88 in self.keys_pressed or 120 in self.keys_pressed:  # X
            target[3] -= step * 2

        # Q/E (81/69) — Wrist rotation
        if 81 in self.keys_pressed or 113 in self.keys_pressed:  # Q
            target[4] -= rot_step
        if 69 in self.keys_pressed or 101 in self.keys_pressed:  # E
            target[4] += rot_step

        # Write back
        if self.right_arm_active:
            self.controller.right_arm_target = target
        else:
            self.controller.left_arm_target = target

    def get_hud_text(self) -> str:
        """Generate HUD overlay text for teleoperation."""
        status = self.controller.get_status()
        mode = "🎮 TELEOPERATION" if self.controller.teleop_active else "🤖 AUTONOMOUS"
        arm = f"Active: {'RIGHT ▶' if self.right_arm_active else '◀ LEFT'}"

        lines = [
            f"═══ PROMETHEUS ═══",
            f"Mode: {mode}",
            f"State: {status['state']}",
            f"Stitch: {status['stitch']}",
            f"Progress: {status['progress']}",
            f"Time: {status['time']}",
            f"{arm}",
            f"",
            f"Controls: WASD=Move QE=Rotate",
            f"1-5=Hand Poses TAB=Switch Arm",
            f"SPACE=Toggle Mode G=Auto Start",
        ]
        return "\n".join(lines)
