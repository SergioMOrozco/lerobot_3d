"""Viser-backed live viewer: scene / robot / per-link point clouds, plus GUI controls.

One browser tab replaces the old Open3D window + Foxglove publishing + Tk control panel
used by ``SystemStateViewer``.
"""
from __future__ import annotations

import numpy as np
import viser


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
        self._robot_handle = None
        self._link_handles: dict[str, object] = {}
        self._link_frames: dict[str, object] = {}

        self.quit = False
        self.capture = False
        self.save_subgoal = False

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

    def load_static_meshes(self, meshes: list[tuple[str, str, np.ndarray, np.ndarray]]) -> None:
        """Mount each URDF visual mesh once, in its local rest pose.

        Call this once at startup (see ``RobotState.get_static_meshes``). Each mesh is
        added as a child of a per-link frame node; animating the robot afterwards only
        needs :func:`update_link_poses`, not re-uploading vertex data every frame --
        these meshes are tens of thousands of vertices each, so resending them per
        frame (instead of just moving a frame) was the dominant per-frame cost.
        """
        for link_name, mesh_name, vertices, faces in meshes:
            if link_name not in self._link_frames:
                self._link_frames[link_name] = self.server.scene.add_frame(
                    f"/robot_urdf/{link_name}", axes_length=0.0, show_axes=False
                )
            self.server.scene.add_mesh_simple(
                name=f"/robot_urdf/{link_name}/{mesh_name}",
                vertices=np.asarray(vertices, dtype=np.float32),
                faces=np.asarray(faces, dtype=np.uint32),
                color=(200, 200, 200),
                flat_shading=True,
            )

    def update_link_poses(self, link_poses: dict[str, tuple[np.ndarray, np.ndarray]] | None) -> None:
        """Move each link's mesh rigidly by updating its frame's pose (cheap: 7 floats/link)."""
        for link_name, (translation, quat_wxyz) in (link_poses or {}).items():
            frame = self._link_frames.get(link_name)
            if frame is None:
                continue
            frame.position = np.asarray(translation, dtype=np.float32)
            frame.wxyz = np.asarray(quat_wxyz, dtype=np.float32)

    def update(
        self,
        scene_points: np.ndarray,
        scene_colors: np.ndarray | None,
        robot_points: np.ndarray,
        robot_colors: np.ndarray | None,
        link_pcds: dict[str, np.ndarray] | None = None,
        link_poses: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> None:
        """Push one frame's scene/robot/per-link point clouds and robot pose to viser."""
        self._scene_handle = self._upsert_point_cloud(
            self._scene_handle, "/scene_pcd", scene_points, scene_colors
        )
        self._robot_handle = self._upsert_point_cloud(
            self._robot_handle, "/robot_pcd", robot_points, robot_colors
        )
        for link_name, pts in (link_pcds or {}).items():
            self._link_handles[link_name] = self._upsert_point_cloud(
                self._link_handles.get(link_name), f"/robot_links/{link_name}", pts, None
            )
        self.update_link_poses(link_poses)

    def close(self) -> None:
        self.server.stop()
