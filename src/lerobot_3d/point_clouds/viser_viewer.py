"""Viser-backed live viewer: scene / robot / per-link point clouds, plus GUI controls.

One browser tab replaces the old Open3D window + Foxglove publishing + Tk control panel
used by ``SystemStateViewer``.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import viser

if TYPE_CHECKING:
    from lerobot_3d.common.types import RobotSnapshot

ROBOT_PCD_COLOR = (1.0, 0.1, 0.1)
"""Float RGB used for every robot's sampled mesh point cloud."""


def grid_offsets(n: int, spacing: float) -> list[np.ndarray]:
    """World-frame base positions for ``n`` robots on a near-square grid ``spacing`` apart.

    Robot 0 sits at the origin (so a camera scene calibrated to it still lines up); the rest
    fill ``ceil(sqrt(n))`` columns along +x, then rows along +y.
    """
    cols = max(1, math.ceil(math.sqrt(n)))
    return [
        np.array([(i % cols) * spacing, (i // cols) * spacing, 0.0], dtype=np.float64)
        for i in range(n)
    ]


def _as_viser_colors(points: np.ndarray, colors: np.ndarray | None) -> np.ndarray:
    """Uint8 RGB for viser, from either float [0, 1] or already-uint8 colors."""
    if colors is None:
        return np.full((points.shape[0], 3), 178, dtype=np.uint8)  # mid-gray
    colors = np.asarray(colors)
    if colors.dtype == np.uint8:
        return colors
    return np.clip(colors * 255.0, 0, 255).astype(np.uint8)


class ViserSceneViewer:
    def __init__(
        self,
        point_size: float = 0.003,
        port: int = 8080,
        host: str = "0.0.0.0",
        controls: bool = True,
    ):
        self.server = viser.ViserServer(host=host, port=port)
        print(f"Viser running at http://localhost:{port}")

        self.point_size = point_size
        self._scene_handle = None
        self._robot_frames: dict[int, object] = {}
        self._robot_handles: dict[int, object] = {}
        self._link_handles: dict[tuple[int, str], object] = {}
        self._link_frames: dict[tuple[int, str], object] = {}

        self.quit = False
        self.capture = False
        self.save_subgoal = False
        self._status_handle = None
        self._status_text = None

        if controls:
            self._add_controls()

    def _add_controls(self) -> None:
        quit_button = self.server.gui.add_button("Quit", color="red")
        capture_button = self.server.gui.add_button("Capture")
        save_subgoal_button = self.server.gui.add_button("Save subgoal")

        @quit_button.on_click
        def _on_quit(_) -> None:
            self.quit = True

        @capture_button.on_click
        def _on_capture(_) -> None:
            self.capture = True

        @save_subgoal_button.on_click
        def _on_save_subgoal(_) -> None:
            self.save_subgoal = True

    def set_status(self, text: str) -> None:
        """Show ``text`` in the GUI sidebar (e.g. the episode recording timer)."""
        if text == self._status_text:
            return
        self._status_text = text
        content = f"**{text}**"
        if self._status_handle is None:
            self._status_handle = self.server.gui.add_markdown(content)
        else:
            self._status_handle.content = content

    def _upsert_point_cloud(self, handle, name: str, points: np.ndarray, colors: np.ndarray | None):
        points = np.asarray(points, dtype=np.float32)
        colors = _as_viser_colors(points, colors)
        if handle is None:
            return self.server.scene.add_point_cloud(
                name=name,
                points=points,
                colors=colors,
                point_size=self.point_size,
                point_shape="circle",
            )
        handle.points = points
        handle.colors = colors
        return handle

    def _robot_root(self, robot_index: int) -> str:
        return f"/robots/robot_{robot_index}"

    def _ensure_robot_frame(self, robot_index: int, base_offset=None) -> None:
        """Parent frame for one robot, placed at its grid offset. Everything under it (link
        frames, point clouds) is sent in that robot's base frame."""
        if robot_index in self._robot_frames:
            return
        position = np.zeros(3) if base_offset is None else np.asarray(base_offset)
        self._robot_frames[robot_index] = self.server.scene.add_frame(
            self._robot_root(robot_index),
            position=tuple(float(v) for v in position),
            axes_length=0.0,
            show_axes=False,
        )

    def load_static_meshes(
        self,
        meshes: list[tuple[str, str, np.ndarray, np.ndarray]],
        robot_index: int = 0,
        base_offset: np.ndarray | None = None,
    ) -> None:
        """Mount each URDF visual mesh once, in its local rest pose, for one robot.

        Call this once per robot at startup (see ``RobotState.get_static_meshes``). Each
        mesh is added as a child of a per-link frame node under the robot's grid frame;
        animating the robot afterwards only needs :func:`update_link_poses`, not
        re-uploading vertex data every frame -- these meshes are tens of thousands of
        vertices each, so resending them per frame (instead of just moving a frame) was
        the dominant per-frame cost.
        """
        self._ensure_robot_frame(robot_index, base_offset)
        root = self._robot_root(robot_index)
        for link_name, mesh_name, vertices, faces in meshes:
            key = (robot_index, link_name)
            if key not in self._link_frames:
                self._link_frames[key] = self.server.scene.add_frame(
                    f"{root}/urdf/{link_name}", axes_length=0.0, show_axes=False
                )
            self.server.scene.add_mesh_simple(
                name=f"{root}/urdf/{link_name}/{mesh_name}",
                vertices=np.asarray(vertices, dtype=np.float32),
                faces=np.asarray(faces, dtype=np.uint32),
                color=(200, 200, 200),
                flat_shading=True,
            )

    def update_link_poses(
        self,
        link_poses: dict[str, tuple[np.ndarray, np.ndarray]] | None,
        robot_index: int = 0,
    ) -> None:
        """Move each link's mesh rigidly by updating its frame's pose (cheap: 7 floats/link).

        Poses are in the robot's base frame; its grid frame supplies the offset."""
        for link_name, (translation, quat_wxyz) in (link_poses or {}).items():
            frame = self._link_frames.get((robot_index, link_name))
            if frame is None:
                continue
            frame.position = np.asarray(translation, dtype=np.float32)
            frame.wxyz = np.asarray(quat_wxyz, dtype=np.float32)

    def update(
        self,
        scene_points: np.ndarray,
        scene_colors: np.ndarray | None,
        robots: list[RobotSnapshot],
    ) -> None:
        """Push one frame: the world-frame scene cloud plus each robot's clouds and pose."""
        self._scene_handle = self._upsert_point_cloud(
            self._scene_handle, "/scene_pcd", scene_points, scene_colors
        )
        for robot in robots:
            i = robot.index
            self._ensure_robot_frame(i, robot.base_offset)
            self._robot_frames[i].position = np.asarray(robot.base_offset)
            self._robot_frames[i].wxyz = np.asarray(robot.base_wxyz)
            root = self._robot_root(i)
            robot_points = np.asarray(robot.pcd, dtype=np.float64)
            robot_colors = np.tile(np.array([ROBOT_PCD_COLOR]), (robot_points.shape[0], 1))
            self._robot_handles[i] = self._upsert_point_cloud(
                self._robot_handles.get(i), f"{root}/pcd", robot_points, robot_colors
            )
            for link_name, pts in robot.link_pcds.items():
                key = (i, link_name)
                self._link_handles[key] = self._upsert_point_cloud(
                    self._link_handles.get(key), f"{root}/links/{link_name}", pts, None
                )
            self.update_link_poses(robot.link_poses, robot_index=i)

    def close(self) -> None:
        self.server.stop()
