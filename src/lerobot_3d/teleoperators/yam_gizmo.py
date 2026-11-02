"""Drag-gizmo end-effector teleop for the YAM arms: a viser transform handle on each gripper.

The same end-effector teleop as :class:`~lerobot_3d.teleoperators.yam_keyboard.YamKeyboard` (an IK
target per arm, solved every step, pulled back to what the arm reached when it's out of reach, Home
easing arms to rest) with the targets set by dragging a 3D handle in the viser scene instead of key
presses, and each gripper by a slider. The keyboard keys still work alongside (the handles follow).

A handle being dragged stays where the mouse puts it, even past the arm's reach, and the arm follows
as closely as it can; on release (or whenever it isn't being dragged) the handle snaps to the pose the
arm actually reached, so it always shows where the gripper is.

Handles live in the scene's world frame, the station URDF's, with the calibrated arm base poses -
the frame the arms are drawn in. :meth:`YamGizmo.attach` adds the handles and sliders to a viser
server once the teleop is connected (``TeleopPointCloudSystem.connect`` does this).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from lerobot.teleoperators.config import TeleoperatorConfig
from scipy.spatial.transform import Rotation

from lerobot_3d.teleoperators.yam_keyboard import YamKeyboard, YamKeyboardConfig

GIZMO_HELP = "Drag a gripper's handle to move that arm; sliders set the grippers. The keyboard keys also work."


def _wxyz(matrix: np.ndarray) -> np.ndarray:
    x, y, z, w = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return np.array([w, x, y, z])


def _pose(position, wxyz) -> np.ndarray:
    w, x, y, z = wxyz
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    pose[:3, 3] = position
    return pose


@TeleoperatorConfig.register_subclass("yam_gizmo")
@dataclass
class YamGizmoConfig(YamKeyboardConfig):
    gizmo_scale: float = 0.12
    """Size of each handle's arrows and rings (metres)."""


class YamGizmo(YamKeyboard):
    config_class = YamGizmoConfig
    name = "yam_gizmo"

    def __init__(self, config: YamGizmoConfig):
        super().__init__(config)
        self._handles: dict[str, Any] = {}
        self._sliders: dict[str, Any] = {}
        self._dragging: set[str] = set()
        self._shown: dict[str, np.ndarray] = {}
        """side -> the pose its handle was last moved to by :meth:`_sync` (to skip unchanged updates)."""

    def attach(self, server) -> None:
        """Adds a handle per arm at its current end-effector target, and a gripper slider per arm."""
        if not self.is_connected:
            raise ConnectionError(f"{self} must be connected before attach().")
        for side in self.sides:
            target = self._target[side]
            handle = server.scene.add_transform_controls(
                f"/yam_gizmo/{side}", scale=self.config.gizmo_scale, position=target[:3, 3], wxyz=_wxyz(target)
            )
            self._shown[side] = target.copy()
            slider = server.gui.add_slider(
                f"{side} gripper", min=0.0, max=1.0, step=0.01, initial_value=self._gripper[side]
            )
            self._bind(side, handle, slider)
            self._handles[side], self._sliders[side] = handle, slider

    def _bind(self, side: str, handle, slider) -> None:
        # Callbacks run on viser's thread; each replaces a whole entry, which get_action reads.
        @handle.on_drag_start
        def _(_) -> None:
            self._dragging.add(side)

        @handle.on_drag_end
        def _(_) -> None:
            self._dragging.discard(side)

        @handle.on_update
        def _(_) -> None:
            if side not in self._homing:
                self._target[side] = _pose(handle.position, handle.wxyz)

        @slider.on_update
        def _(_) -> None:
            if side not in self._homing:
                self._gripper[side] = float(slider.value)

    def get_action(self) -> dict[str, Any]:
        action = super().get_action()
        self._sync()
        return action

    def _sync(self) -> None:
        """Handles not being dragged go to the pose their arm reached; sliders show the grippers."""
        for side, handle in self._handles.items():
            target = self._target[side]
            if side not in self._dragging and not np.allclose(target, self._shown[side], atol=1e-5):
                handle.position, handle.wxyz = target[:3, 3], _wxyz(target)
                self._shown[side] = target.copy()
            slider = self._sliders[side]
            if abs(slider.value - self._gripper[side]) > 1e-3:
                slider.value = self._gripper[side]

    def status_text(self) -> str:
        if self._homing:
            return f"Gizmo: homing {', '.join(side for side in self.sides if side in self._homing)}..."
        parts = []
        for side in self.sides:
            p = self._target[side][:3, 3]
            parts.append(f"{side} ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}) m, gripper {self._gripper[side]:.0%}")
        return "Gizmo: " + "  |  ".join(parts)

    def disconnect(self) -> None:
        for handle in self._handles.values():
            handle.remove()
        for slider in self._sliders.values():
            slider.remove()
        self._handles, self._sliders, self._shown = {}, {}, {}
        self._dragging.clear()
        super().disconnect()
