"""Live YAM data visualizer: measured joints pose the station URDF, wrist RGB-D streams are drawn
at the URDF camera links, all in one viser page. Read-only: it never commands the robot.

    python -m lerobot_3d.point_clouds.yam_live_viewer --urdf "$YAM_URDF"

Joint angles and finger travel are measured (joint topic); camera poses are the URDF's nominal
mounts, *not* a calibration. The sidebar says which is which.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot_3d.point_clouds.rgbd import colored_cloud
from lerobot_3d.point_clouds.viser_viewer import ViserSceneViewer
from lerobot_3d.point_clouds.yam_robot_state import YamRobotState

DEPTH_SCALE_MM = 0.001
"""ROS ``16UC1`` depth is millimetres."""


def _wxyz(R: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(R).as_quat()[[3, 0, 1, 2]]


def _thumbnail(image: np.ndarray, width: int = 320) -> np.ndarray:
    import cv2

    h, w = image.shape[:2]
    if w <= width:
        return image
    return cv2.resize(image, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)


class YamLiveViewer:
    def __init__(
        self,
        urdf_path,
        source,
        camera_links: dict[str, str],
        *,
        port: int = 8080,
        stride: int = 4,
        max_depth: float = 1.0,
        stale_s: float = 1.0,
        point_size: float = 0.002,
    ):
        self.source = source
        self.sides = ("left", "right")
        self.camera_links = dict(camera_links)
        self.stride, self.max_depth, self.stale_s = stride, max_depth, stale_s
        self.states = {side: YamRobotState(urdf_path, side) for side in self.sides}
        self.index = {side: i for i, side in enumerate(self.sides)}
        self.viewer = ViserSceneViewer(point_size=point_size, port=port, controls=False)
        self.server = self.viewer.server
        self._meshes_loaded: set[str] = set()
        self._cam_nodes: dict[str, dict] = {}
        self._seen_counts: dict[str, tuple[int, int]] = {}
        self._status = self.server.gui.add_markdown("Waiting for data...")
        self._images_folder = self.server.gui.add_folder("Camera images")

    # --- robots ------------------------------------------------------------------------------
    def _arm_snapshots(self, arms, now):
        snapshots, lines, joints_cfg = [], [], {}
        for side in self.sides:
            sample = arms.get(side)
            if sample is None:
                lines.append(f"- **{side}**: waiting for `{side}_joint1..6` on the joint topic")
                continue
            state = self.states[side]
            try:
                joints = state.get_joint_positions(sample.observation)
                snapshot = state.get_robot_snapshot(sample.observation, index=self.index[side])
            except ValueError as e:
                lines.append(f"- **{side}**: not drawn: {e}")
                continue
            if side not in self._meshes_loaded:
                self.viewer.load_static_meshes(state.get_static_meshes(), self.index[side], snapshot.base_offset)
                self._meshes_loaded.add(side)
            snapshots.append(snapshot)
            joints_cfg.update(joints)
            age = now - sample.arrival
            q = np.degrees(sample.observation["position_rad"])
            grip = (f"{sample.observation['gripper_open']:.2f} open (from {side}_joint7)"
                    if sample.gripper_reported else "not reported (drawn closed)")
            stale = f" **STALE {age:.1f}s**" if age > self.stale_s else ""
            lines.append(f"- **{side}**{stale}: " + " ".join(f"{v:.1f}°" for v in q) + f"; gripper {grip}")
        return snapshots, joints_cfg, lines

    # --- cameras -----------------------------------------------------------------------------
    def _camera_node(self, ns):
        if ns not in self._cam_nodes:
            root = "/cameras/" + ns.strip("/").replace("/", "_")
            self._cam_nodes[ns] = {
                "root": root,
                "frame": self.server.scene.add_frame(root, show_axes=False, visible=False),
                "cloud": None,
                "frustum": None,
                "image": None,
            }
        return self._cam_nodes[ns]

    def _update_camera(self, ns, link, cam, joints_cfg, now):
        node = self._camera_node(ns)
        counts = (cam.color_count, cam.depth_count)
        fresh = counts != self._seen_counts.get(ns)
        self._seen_counts[ns] = counts

        if cam.color is not None and fresh:
            thumb = _thumbnail(cam.color)
            if node["image"] is None:
                with self._images_folder:
                    node["image"] = self.server.gui.add_image(thumb, label=ns, format="jpeg", jpeg_quality=70)
            else:
                node["image"].image = thumb

        side = link.split("_")[0]
        if f"{side}_joint1" not in joints_cfg:
            node["frame"].visible = False
            return f"- `{ns}` → `{link}`: images only; waiting for {side} joints to place it"
        T_world_cam = self.states[side].robot_urdf.link_fk(cfg=joints_cfg, use_names=True)[link]
        node["frame"].position = T_world_cam[:3, 3]
        node["frame"].wxyz = _wxyz(T_world_cam[:3, :3])
        node["frame"].visible = True

        notes = []
        if node["frustum"] is None and cam.K_color is not None and cam.color is not None:
            h, w = cam.color.shape[:2]
            fov = 2 * np.arctan2(h / 2, cam.K_color[1, 1])
            node["frustum"] = self.server.scene.add_camera_frustum(
                f"{node['root']}/frustum", fov=float(fov), aspect=w / h, scale=0.04)

        if cam.depth is None or cam.K_depth is None:
            notes.append("waiting for depth + camera_info")
        elif fresh:
            T_color_depth = cam.T_color_depth
            if T_color_depth is None:
                T_color_depth = np.eye(4)
                notes.append("depth→color TF not received (identity used)")
            points, colors = colored_cloud(
                cam.depth, cam.K_depth, DEPTH_SCALE_MM, cam.color, cam.K_color, T_color_depth,
                stride=self.stride, max_depth=self.max_depth)
            if node["cloud"] is None:
                node["cloud"] = self.server.scene.add_point_cloud(
                    f"{node['root']}/cloud", points=points.astype(np.float32), colors=colors,
                    point_size=self.viewer.point_size, point_shape="circle")
            else:
                node["cloud"].points = points.astype(np.float32)
                node["cloud"].colors = colors

        age = now - max(cam.color_arrival, cam.depth_arrival) if cam.color_count or cam.depth_count else None
        age_text = "no frames yet" if age is None else (f"**STALE {age:.1f}s**" if age > self.stale_s else "live")
        errors = "; ".join(cam.errors[-2:])
        extra = "; ".join(notes + ([errors] if errors else []))
        return (f"- `{ns}` at `{link}` (nominal URDF mount, not calibrated): {age_text}, "
                f"{cam.color_count} color / {cam.depth_count} depth" + (f"; {extra}" if extra else ""))

    # --- loop --------------------------------------------------------------------------------
    def tick(self) -> None:
        now = time.monotonic()
        arms, cams = self.source.latest()
        snapshots, joints_cfg, lines = self._arm_snapshots(arms, now)
        if snapshots:
            self.viewer.update(np.empty((0, 3)), None, snapshots)
        for ns, link in self.camera_links.items():
            cam = cams.get(ns)
            if cam is None:
                continue
            lines.append(self._update_camera(ns, link, cam, joints_cfg, now))
        self._status.content = "**Measured joints**\n" + "\n".join(
            l for l in lines if not l.startswith("- `")) + "\n\n**Cameras**\n" + "\n".join(
            l for l in lines if l.startswith("- `")) + "\n\nRead-only viewer: no motor commands."

    def close(self) -> None:
        self.viewer.close()


def parse_camera_links(values) -> dict[str, str]:
    """``["/camera/left_wrist=left_camera", ...]`` -> ``{namespace: urdf_link}``."""
    out = {}
    for value in values:
        ns, sep, link = value.partition("=")
        if not sep or not ns or link.split("_")[0] not in ("left", "right"):
            raise ValueError(f"--camera expects NAMESPACE=<left|right>_LINK, got {value!r}")
        out[ns.rstrip("/")] = link
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--urdf", required=True, help="Composed YAM station URDF (see README)")
    parser.add_argument("--joint-topic", default="/joint_states")
    parser.add_argument(
        "--camera", action="append", metavar="NAMESPACE=URDF_LINK",
        help="Camera namespace and the arm URDF link it is mounted on (repeatable). "
             "Default: /camera/left_wrist=left_camera and /camera/right_wrist=right_camera")
    parser.add_argument("--stride", type=int, default=4, help="Depth pixel subsampling for the clouds")
    parser.add_argument("--max-depth", type=float, default=1.0, help="Drop depth beyond this (m)")
    parser.add_argument("--point-size", type=float, default=0.002, help="Point size in metres")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)

    camera_links = parse_camera_links(
        args.camera or ["/camera/left_wrist=left_camera", "/camera/right_wrist=right_camera"])

    from lerobot_3d.point_clouds.ros_yam_source import RosYamSource

    viewer = None
    source = None
    try:
        viewer = YamLiveViewer(args.urdf, None, camera_links, port=args.port, stride=args.stride,
                               max_depth=args.max_depth, point_size=args.point_size)
        # The URDF's finger range converts measured finger travel (m) to a 0..1 opening.
        travel = {side: float(state.joints[6].limit.upper) for side, state in viewer.states.items()}
        source = RosYamSource(list(camera_links), joint_topic=args.joint_topic, finger_travel=travel)
        viewer.source = source
        period = 0.1  # 10 Hz
        print("Ctrl-C to quit.")
        while True:
            t0 = time.monotonic()
            viewer.tick()
            time.sleep(max(0.0, period - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            viewer.close()
        if source is not None:
            source.close()


if __name__ == "__main__":
    main()
