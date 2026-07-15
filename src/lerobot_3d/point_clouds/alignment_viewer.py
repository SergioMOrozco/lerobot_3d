"""Viser-backed interactive point-cloud alignment for camera-extrinsic calibration.

Used by icp.py in place of the old Open3D VisualizerWithKeyCallback keyboard-driven
window: GUI buttons for translate/rotate instead of arrow/number keys.
"""
from __future__ import annotations

import time

import numpy as np
import viser
from scipy.spatial.transform import Rotation

from lerobot_3d.point_clouds.viser_viewer import _as_viser_colors

_TRANSLATE_BUTTONS = (
    ("+X", (1, 0, 0)), ("-X", (-1, 0, 0)),
    ("+Y", (0, 1, 0)), ("-Y", (0, -1, 0)),
    ("+Z", (0, 0, 1)), ("-Z", (0, 0, -1)),
)
_ROTATE_BUTTONS = (
    ("+Rx", (1, 0, 0)), ("-Rx", (-1, 0, 0)),
    ("+Ry", (0, 1, 0)), ("-Ry", (0, -1, 0)),
    ("+Rz", (0, 0, 1)), ("-Rz", (0, 0, -1)),
)


def _world_centroid(T, pts_cam):
    return (T[:3, :3] @ pts_cam.T).T.mean(axis=0) + T[:3, 3]


def _rotate_about_centroid(T_current, pts_cam, axis, angle_deg):
    """Rotate T_current about the world-frame centroid of pts_cam transformed by T_current."""
    centroid = _world_centroid(T_current, pts_cam)
    axis = np.array(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    Rm = Rotation.from_rotvec(axis * np.deg2rad(angle_deg)).as_matrix()
    T_rot = np.eye(4)
    T_rot[:3, :3] = Rm
    T_rot[:3, 3] = centroid - Rm @ centroid
    return T_rot @ T_current


def _translate_world(T_current, delta):
    T_t = np.eye(4)
    T_t[:3, 3] = delta
    return T_t @ T_current


def _pose_delta_summary(T_current, T_init, pts_cam):
    """Centroid displacement (cm) and rotation angle (deg) of T_current relative to T_init.

    Uses centroid displacement rather than the raw matrix translation column so a
    pure in-place rotation (pivoting off-origin) correctly reports ~0cm of "movement".
    """
    trans_cm = np.linalg.norm(
        _world_centroid(T_current, pts_cam) - _world_centroid(T_init, pts_cam)
    ) * 100.0
    T_delta = T_current @ np.linalg.inv(T_init)
    trace = np.clip((np.trace(T_delta[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    rot_deg = np.degrees(np.arccos(trace))
    return trans_cm, rot_deg


class AlignmentViewer:
    """One long-lived viser server for a whole calibration session."""

    def __init__(self, port: int = 8081, host: str = "0.0.0.0"):
        self.server = viser.ViserServer(host=host, port=port)
        print(f"Viser running at http://localhost:{port}")
        self._handles: dict[str, object] = {}

    def show(
        self, name: str, points: np.ndarray, colors: np.ndarray | None, point_size: float = 0.003
    ):
        """Upsert a named point cloud (e.g. '/target', '/moving', '/merged', '/live')."""
        points = np.asarray(points, dtype=np.float32)
        colors = _as_viser_colors(points, colors)
        handle = self._handles.get(name)
        if handle is None:
            handle = self.server.scene.add_point_cloud(
                name=name,
                points=points,
                colors=colors,
                point_size=point_size,
                point_shape="circle",
            )
            self._handles[name] = handle
        else:
            handle.points = points
            handle.colors = colors
        return handle

    def remove(self, name: str) -> None:
        handle = self._handles.pop(name, None)
        if handle is not None:
            handle.remove()

    def align(
        self,
        pts_cam: np.ndarray,
        T_init: np.ndarray,
        T_fallback: np.ndarray | None = None,
        title: str = "Manual align",
        translate_steps=(0.002, 0.01, 0.05),  # meters: fine / medium / coarse
        rotate_steps_deg=(1.0, 5.0, 20.0),
    ) -> np.ndarray:
        """Interactive world-frame nudge of pts_cam (via T_init) onto whatever '/target' shows.

        GUI buttons replace the old keyboard shortcuts: translate/rotate about each axis,
        cycle step size, reset, confirm, abort. Rotations pivot about pts_cam's current
        world-frame centroid, so the cloud spins in place regardless of how far it's
        already been dragged.

        Blocks until Confirm or Abort is clicked. Abort reverts to T_fallback (defaults
        to T_init) -- pass e.g. a pre-ICP pose here when confirming an ICP result, so a
        bad ICP jump can be visually rejected, not just caught by a distance heuristic.

        Returns the confirmed 4x4 transform, or T_fallback if aborted.
        """
        if T_fallback is None:
            T_fallback = T_init
        pts_cam = np.asarray(pts_cam)

        state = {"T": T_init.copy(), "step_idx": 0, "done": False, "confirmed": False}

        def status_text() -> str:
            trans_cm, rot_deg = _pose_delta_summary(state["T"], T_init, pts_cam)
            t_step = translate_steps[state["step_idx"]]
            r_step = rotate_steps_deg[state["step_idx"]]
            return (
                f"step: translate={t_step * 100:.1f}cm rotate={r_step:.1f}deg\n"
                f"delta from initial guess: {trans_cm:.2f}cm, {rot_deg:.2f}deg"
            )

        def refresh() -> None:
            T = state["T"]
            pts_world = (T[:3, :3] @ pts_cam.T).T + T[:3, 3]
            self.show("/moving", pts_world, None)
            text = status_text()
            status.value = text
            print(f"  {text}".replace("\n", "  |  "))

        folder = self.server.gui.add_folder(title)
        with folder:
            step_button = self.server.gui.add_button("Cycle step size")
            translate_buttons = [
                (axis, self.server.gui.add_button(f"Translate {label}"))
                for label, axis in _TRANSLATE_BUTTONS
            ]
            rotate_buttons = [
                (axis, self.server.gui.add_button(f"Rotate {label}"))
                for label, axis in _ROTATE_BUTTONS
            ]
            reset_button = self.server.gui.add_button("Reset")
            confirm_button = self.server.gui.add_button("Confirm", color="green")
            abort_button = self.server.gui.add_button("Abort", color="red")
            status = self.server.gui.add_text("Status", initial_value="", disabled=True)

        def make_translate(axis_vec):
            def _cb(_) -> None:
                step = translate_steps[state["step_idx"]]
                state["T"] = _translate_world(state["T"], np.array(axis_vec) * step)
                refresh()

            return _cb

        def make_rotate(axis_vec):
            def _cb(_) -> None:
                step = rotate_steps_deg[state["step_idx"]]
                state["T"] = _rotate_about_centroid(state["T"], pts_cam, axis_vec, step)
                refresh()

            return _cb

        for axis_vec, button in translate_buttons:
            button.on_click(make_translate(axis_vec))
        for axis_vec, button in rotate_buttons:
            button.on_click(make_rotate(axis_vec))

        @step_button.on_click
        def _on_cycle_step(_) -> None:
            state["step_idx"] = (state["step_idx"] + 1) % len(translate_steps)
            refresh()

        @reset_button.on_click
        def _on_reset(_) -> None:
            state["T"] = T_init.copy()
            refresh()

        @confirm_button.on_click
        def _on_confirm(_) -> None:
            state["confirmed"] = True
            state["done"] = True

        @abort_button.on_click
        def _on_abort(_) -> None:
            state["confirmed"] = False
            state["done"] = True

        refresh()
        print(f"{title}: use the GUI buttons to translate/rotate, then Confirm or Abort.")

        while not state["done"]:
            time.sleep(0.05)

        folder.remove()

        return state["T"] if state["confirmed"] else T_fallback

    def wait_for_confirmation(self, title: str, button_label: str = "Done") -> None:
        """Block until the user acknowledges what's currently shown (e.g. a final preview)."""
        state = {"done": False}

        folder = self.server.gui.add_folder(title)
        with folder:
            button = self.server.gui.add_button(button_label, color="green")

        @button.on_click
        def _on_done(_) -> None:
            state["done"] = True

        print(f"{title}: click '{button_label}' in the viser GUI to continue.")
        while not state["done"]:
            time.sleep(0.05)

        folder.remove()

    def close(self) -> None:
        self.server.stop()
