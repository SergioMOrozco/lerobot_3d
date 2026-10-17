"""Keyboard end-effector teleop for the YAM arms, as a LeRobot :class:`~lerobot.teleoperators.Teleoperator`.

Each key press nudges the active arm's end-effector target (``<side>_gripper``); IK turns the targets
into joint angles every step. Keys come from the terminal running ``lerobot-teleop`` (it needs focus),
fed in through :meth:`YamKeyboard.handle_keys`, so they share the terminal with the recording keys
(Enter / Space / y / n), which this teleop leaves alone.

Translations are along the world (station URDF) axes; rotations are about the end effector's own axes.
Uppercase (Shift) moves 4x further. Targets the arm can't reach are pulled back to the closest pose it
can, so holding a key against a joint limit doesn't wind up.

:meth:`YamKeyboard.home` (the viser **Home** button, or ``0`` for the active arm) eases arms back to
the rest pose over ``home_duration_s`` instead of jumping; motion keys are ignored meanwhile.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.teleoperator import Teleoperator
from scipy.spatial.transform import Rotation

from lerobot_3d.paths import YAM_URDF_PATH
from lerobot_3d.point_clouds.yam_kinematics import YamArmKinematics
from lerobot_3d.point_clouds.yam_robot_state import joint_keys

KEYBOARD_HELP = (
    "w/s: +x/-x  a/d: +y/-y  r/f: +z/-z  i/k: pitch  j/l: yaw  u/o: roll  g/h: gripper open/close  "
    "Tab: switch arm  0: home arm  (Shift: 4x)"
)

_TRANSLATE = {"w": (0, 1), "s": (0, -1), "a": (1, 1), "d": (1, -1), "r": (2, 1), "f": (2, -1)}
"""key -> (world axis, sign)"""
_ROTATE = {"i": (1, 1), "k": (1, -1), "j": (2, 1), "l": (2, -1), "u": (0, 1), "o": (0, -1)}
"""key -> (end-effector axis, sign): pitch about y, yaw about z, roll about x."""
_GRIPPER = {"g": 1, "h": -1}
_FAST = 4.0


@TeleoperatorConfig.register_subclass("yam_keyboard")
@dataclass
class YamKeyboardConfig(TeleoperatorConfig):
    sides: tuple[str, ...] = ("left", "right")
    """Arms this keyboard drives; Tab cycles the active one."""
    urdf_path: str | None = None
    """``None`` -> the bundled YAM station URDF."""
    step_m: float = 0.005
    """End-effector translation per key press (metres)."""
    step_deg: float = 3.0
    """End-effector rotation per key press (degrees)."""
    gripper_step: float = 0.05
    """Gripper opening change per key press (0 closed .. 1 fully open)."""
    home_duration_s: float = 2.0
    """How long :meth:`YamKeyboard.home` takes to ease an arm back to the rest pose."""
    base_poses: dict | None = None
    """Calibrated ``{side: X_W_base}`` arm base poses (see ``yam_robot_state.yam_base_poses``), so IK
    targets live in the same world as the arms drawn in viser; ``None`` -> the URDF's."""


class YamKeyboard(Teleoperator):
    config_class = YamKeyboardConfig
    name = "yam_keyboard"

    def __init__(self, config: YamKeyboardConfig):
        super().__init__(config)
        self.config = config
        self.sides = tuple(config.sides)
        self.active = self.sides[0]
        self._kinematics: dict[str, YamArmKinematics] = {}
        self._q: dict[str, np.ndarray] = {}
        self._target: dict[str, np.ndarray] = {}
        self._gripper: dict[str, float] = {}
        self._homing: dict[str, tuple[np.ndarray, float, float]] = {}
        """side -> (joints, gripper, start time) the arm is easing home from."""

    @property
    def action_features(self) -> dict:
        return {key: float for side in self.sides for key in joint_keys(side)}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return bool(self._kinematics)

    def connect(self, calibrate: bool = True) -> None:
        urdf = self.config.urdf_path or str(YAM_URDF_PATH)
        poses = self.config.base_poses or {}
        self._kinematics = {side: YamArmKinematics(urdf, side, base_pose=poses.get(side)) for side in self.sides}
        for side in self.sides:
            self.reset(side)
            self._homing.pop(side, None)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def reset(self, side: str) -> None:
        """Put ``side`` back at the URDF rest pose (every joint 0, fingers closed) immediately."""
        self._q[side] = np.zeros(6)
        self._target[side] = self._kinematics[side].fk(self._q[side])
        self._gripper[side] = 0.0

    def set_joints(self, observation: dict[str, float]) -> None:
        """Take the current joints (e.g. a real arm's measured ones) as the starting pose for every
        arm whose keys are in ``observation``."""
        for side in self.sides:
            keys = joint_keys(side)
            if all(key in observation for key in keys):
                kinematics = self._kinematics[side]
                q = np.array([observation[key] for key in keys[:6]], dtype=np.float64)
                self._q[side] = np.clip(q, kinematics.lower, kinematics.upper)
                self._target[side] = kinematics.fk(self._q[side])
                self._gripper[side] = float(np.clip(observation[keys[6]], 0.0, 1.0))
                self._homing.pop(side, None)

    def home(self, sides=None) -> None:
        """Ease ``sides`` (default: every arm) back to the rest pose over ``home_duration_s``."""
        now = time.monotonic()
        for side in self.sides if sides is None else sides:
            self._homing[side] = (self._q[side].copy(), self._gripper[side], now)

    @property
    def homing(self) -> bool:
        return bool(self._homing)

    def handle_key(self, key: str) -> bool:
        """Apply one key press; returns False for keys this teleop doesn't use."""
        if key == "\t":
            self.active = self.sides[(self.sides.index(self.active) + 1) % len(self.sides)]
            return True
        if key == "0":
            self.home([self.active])
            return True
        lower = key.lower()
        if lower not in _TRANSLATE and lower not in _ROTATE and lower not in _GRIPPER:
            return False
        if self.active in self._homing:
            return True  # motion keys wait until the arm is home
        scale = _FAST if key != lower else 1.0
        target = self._target[self.active]
        if lower in _TRANSLATE:
            axis, sign = _TRANSLATE[lower]
            target[axis, 3] += sign * scale * self.config.step_m
        elif lower in _ROTATE:
            axis, sign = _ROTATE[lower]
            rotvec = np.zeros(3)
            rotvec[axis] = sign * scale * np.radians(self.config.step_deg)
            target[:3, :3] = target[:3, :3] @ Rotation.from_rotvec(rotvec).as_matrix()
        else:
            step = _GRIPPER[lower] * scale * self.config.gripper_step
            self._gripper[self.active] = float(np.clip(self._gripper[self.active] + step, 0.0, 1.0))
        return True

    def handle_keys(self, keys) -> list[str]:
        """Apply every key this teleop uses; returns the rest (e.g. for the recording controls)."""
        return [key for key in keys if not self.handle_key(key)]

    def get_action(self) -> dict[str, Any]:
        """Joint targets for every arm, solved from the end-effector targets."""
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")
        action = {}
        now = time.monotonic()
        for side in self.sides:
            kinematics = self._kinematics[side]
            if side in self._homing:
                q0, gripper0, t0 = self._homing[side]
                duration = self.config.home_duration_s
                alpha = 1.0 if duration <= 0 else min(1.0, (now - t0) / duration)
                self._q[side] = (1.0 - alpha) * q0
                self._gripper[side] = (1.0 - alpha) * gripper0
                if alpha >= 1.0:
                    del self._homing[side]
            else:
                self._q[side] = kinematics.ik(self._target[side], self._q[side])
            # Pull an unreachable target back to what was reached, so it can't wind up.
            self._target[side] = kinematics.fk(self._q[side])
            keys = joint_keys(side)
            action.update({key: float(v) for key, v in zip(keys[:6], self._q[side])})
            action[keys[6]] = self._gripper[side]
        return action

    def status_text(self) -> str:
        if self._homing:
            return f"Keyboard: homing {', '.join(side for side in self.sides if side in self._homing)}..."
        p = self._target[self.active][:3, 3]
        return (f"Keyboard: {self.active} arm at ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}) m, "
                f"gripper {self._gripper[self.active]:.0%} open")

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        self._kinematics = {}
