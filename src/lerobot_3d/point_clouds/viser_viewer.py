"""Viser-backed live viewer: scene / robot / per-link point clouds, plus GUI controls.

One browser tab replaces the old Open3D window + Foxglove publishing + Tk control panel
used by ``SystemStateViewer``.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import viser
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from lerobot_3d.common.types import RobotSnapshot
    from lerobot_3d.table_plane import TablePlane

ROBOT_PCD_COLOR = (1.0, 0.1, 0.1)
"""Float RGB used for every robot's sampled mesh point cloud."""
TABLE_COLOR = (60, 160, 255)
"""RGB of the fitted table plane overlay."""
TABLE_GRID_CELL_M = 0.05
"""Spacing (m) of the grid drawn on the table plane; every 5th line is darker."""
TABLE_GRID_SIZE_M = 4.0
"""Side (m) of the square grid drawn on the table plane, centered on the table (not limited to its extent)."""



@dataclass
class CameraView:
    """One camera as drawn in viser: pose, field of view, and its latest color image."""

    X_WC: np.ndarray
    """Camera -> world, OpenCV optical frame (+z forward, -y up)."""
    fov: float
    """Vertical field of view (radians)."""
    aspect: float
    """Image width / height."""
    image: np.ndarray | None = None
    """``(H, W, 3)`` uint8 **RGB**."""


def _shrink(image: np.ndarray, width: int) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= width:
        return image
    import cv2

    return cv2.resize(image, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)


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
        self.fit_table = False
        self._status_handle = None
        self._status_text = None
        self.camera_frustum_scale = 0.08
        """Distance (m) from each camera to its image plane in the scene."""
        self.camera_image_width = 480
        """Camera images are shrunk to at most this many pixels wide before sending."""
        self.camera_image_period_s = 0.1
        """Each camera's image is re-sent at most this often."""
        self._cameras: dict[str, CameraView] = {}
        self._camera_frustums: dict[str, object] = {}
        self._camera_buttons: dict[str, object] = {}
        self._camera_image_times: dict[str, float] = {}
        self._cameras_folder = None

        if controls:
            self._add_controls()

    def update_cameras(self, cameras: dict[str, CameraView]) -> None:
        """Draw every camera at its pose with its live image, plus one sidebar button per camera.

        Each camera is a frustum carrying its color image (re-sent at most every
        :attr:`camera_image_period_s`). Its button moves that browser's view to the camera's pose and
        field of view, once: the view isn't locked, so you can orbit away at any time. Cameras that
        stop being reported are hidden and their buttons disabled.
        """
        now = time.monotonic()
        for name, camera in cameras.items():
            X_WC = np.asarray(camera.X_WC, dtype=np.float64)
            wxyz = Rotation.from_matrix(X_WC[:3, :3]).as_quat()[[3, 0, 1, 2]]
            frustum = self._camera_frustums.get(name)
            if frustum is None:
                frustum = self._camera_frustums[name] = self.server.scene.add_camera_frustum(
                    f"/cameras/{name}", fov=camera.fov, aspect=camera.aspect, scale=self.camera_frustum_scale,
                    color=(40, 40, 40), format="jpeg", jpeg_quality=75,
                )
                self._add_camera_button(name)
            frustum.position = X_WC[:3, 3]
            frustum.wxyz = wxyz
            frustum.fov, frustum.aspect, frustum.visible = camera.fov, camera.aspect, True
            if camera.image is not None and now - self._camera_image_times.get(name, -np.inf) >= self.camera_image_period_s:
                frustum.image = _shrink(camera.image, self.camera_image_width)
                self._camera_image_times[name] = now
            self._camera_buttons[name].disabled = False
        for name, frustum in self._camera_frustums.items():
            if name not in cameras:
                frustum.visible = False
                self._camera_buttons[name].disabled = True
        self._cameras = dict(cameras)

    def _add_camera_button(self, name: str) -> None:
        if self._cameras_folder is None:
            self._cameras_folder = self.server.gui.add_folder("Camera views")
        with self._cameras_folder:
            button = self.server.gui.add_button(name, hint=f"Look through {name} (then move away freely)")

        @button.on_click
        def _on_click(event) -> None:
            self.look_through(name, [event.client] if event.client is not None else None)

        self._camera_buttons[name] = button

    def look_through(self, name: str, clients=None) -> None:
        """Move ``clients``' views (default: every browser) to camera ``name``'s pose and field of view."""
        camera = self._cameras.get(name)
        if camera is None:
            return
        X_WC = np.asarray(camera.X_WC, dtype=np.float64)
        position = X_WC[:3, 3]
        for client in self.server.get_clients().values() if clients is None else clients:
            try:
                with client.atomic():
                    # OpenCV camera frame, like viser's: +z looks forward, -y is up.
                    client.camera.position = position
                    client.camera.look_at = position + 0.5 * X_WC[:3, 2]
                    client.camera.up_direction = -X_WC[:3, 1]
                    client.camera.fov = float(camera.fov)
            except AssertionError:
                pass  # this browser hasn't reported its camera yet

    def _add_controls(self) -> None:
        quit_button = self.server.gui.add_button("Quit", color="red")
        capture_button = self.server.gui.add_button("Capture")
        save_subgoal_button = self.server.gui.add_button("Save subgoal")
        fit_table_button = self.server.gui.add_button(
            "Fit table", hint="Click the table in each camera's window, then fit and save its plane"
        )

        @quit_button.on_click
        def _on_quit(_) -> None:
            self.quit = True

        @capture_button.on_click
        def _on_capture(_) -> None:
            self.capture = True

        @save_subgoal_button.on_click
        def _on_save_subgoal(_) -> None:
            self.save_subgoal = True

        @fit_table_button.on_click
        def _on_fit_table(_) -> None:
            self.fit_table = True

    def draw_table_plane(self, plane: TablePlane) -> None:
        """Show the fitted table top as a translucent rectangle with its frame (``+z`` = normal)."""
        X_W_table = np.asarray(plane.X_W_table, dtype=np.float64)
        wxyz = Rotation.from_matrix(X_W_table[:3, :3]).as_quat()[[3, 0, 1, 2]]
        hx, hy = 0.5 * np.asarray(plane.extent, dtype=np.float64)
        self.server.scene.add_frame(
            "/table", position=X_W_table[:3, 3], wxyz=wxyz, axes_length=0.1, axes_radius=0.004
        )
        self.server.scene.add_mesh_simple(
            "/table/top",
            vertices=np.array([[-hx, -hy, 0.0], [hx, -hy, 0.0], [hx, hy, 0.0], [-hx, hy, 0.0]], dtype=np.float32),
            faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32),
            color=TABLE_COLOR,
            opacity=0.35,
            side="double",
            cast_shadow=False,
            receive_shadow=False,
        )
        # Lifted 1 mm so it doesn't z-fight with the translucent top.
        self.server.scene.add_grid(
            "/table/grid",
            width=TABLE_GRID_SIZE_M,
            height=TABLE_GRID_SIZE_M,
            plane="xy",
            cell_size=TABLE_GRID_CELL_M,
            cell_color=TABLE_COLOR,
            section_size=5 * TABLE_GRID_CELL_M,
            section_color=(20, 90, 200),
            section_thickness=1.5,
            position=(0.0, 0.0, 0.001),
        )

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
        meshes: list[tuple],
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

        Each mesh is ``(link, mesh name, vertices, faces)``, drawn grey, or
        ``(link, mesh name, vertices, faces, rgb)`` in its own 0-255 color.
        """
        self._ensure_robot_frame(robot_index, base_offset)
        root = self._robot_root(robot_index)
        for link_name, mesh_name, vertices, faces, *color in meshes:
            key = (robot_index, link_name)
            if key not in self._link_frames:
                self._link_frames[key] = self.server.scene.add_frame(
                    f"{root}/urdf/{link_name}", axes_length=0.0, show_axes=False
                )
            self.server.scene.add_mesh_simple(
                name=f"{root}/urdf/{link_name}/{mesh_name}",
                vertices=np.asarray(vertices, dtype=np.float32),
                faces=np.asarray(faces, dtype=np.uint32),
                color=tuple(color[0]) if color else (200, 200, 200),
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
            robot_colors = robot.pcd_colors
            if robot_colors is None:
                robot_colors = np.tile(np.array([ROBOT_PCD_COLOR]), (robot_points.shape[0], 1))
            self._robot_handles[i] = self._upsert_point_cloud(
                self._robot_handles.get(i), f"{root}/pcd", robot_points, robot_colors
            )
            for link_name, pts in robot.link_pcds.items():
                key = (i, link_name)
                self._link_handles[key] = self._upsert_point_cloud(
                    self._link_handles.get(key), f"{root}/links/{link_name}", pts,
                    (robot.link_pcd_colors or {}).get(link_name),
                )
            self.update_link_poses(robot.link_poses, robot_index=i)

    def close(self) -> None:
        self.server.stop()
