"""Live YAM joint states and wrist RGB-D frames from ROS 2 topics (read-only, no commands).

Joints: one ``sensor_msgs/JointState`` topic with the YAM station URDF names (``<side>_joint1..6``
in radians, optional ``<side>_joint7`` finger travel in metres). Cameras: per namespace, the
topics below plus the depth-to-color extrinsics from ``/tf_static``. Only the latest sample and
its arrival time are kept, so a viewer can show staleness.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from lerobot_3d.point_clouds.rgbd import decode_color, decode_depth

SIDES = ("left", "right")
COLOR_TOPIC = "color/image_raw/compressed"
COLOR_INFO_TOPIC = "color/camera_info"
DEPTH_TOPIC = "depth_raw/image_rect_raw/compressedDepth"
DEPTH_INFO_TOPIC = "depth_raw/camera_info"


def joint_observation(names, positions, side: str, finger_travel: float | None):
    """``JointState`` names/positions -> YamRobotState observation for one arm, or ``None``.

    Returns ``(observation, gripper_reported)``. ``observation`` is
    ``{"position_rad": (6,), "gripper_open": float}``; when the message carries no finger joint
    the gripper is drawn closed and ``gripper_reported`` is False.
    """
    by_name = dict(zip(names, positions))
    arm = [f"{side}_joint{i}" for i in range(1, 7)]
    if not all(n in by_name for n in arm):
        return None
    q = np.array([float(by_name[n]) for n in arm])
    finger = by_name.get(f"{side}_joint7")
    if finger is None or not finger_travel:
        return {"position_rad": q, "gripper_open": 0.0}, False
    opening = float(np.clip(float(finger) / finger_travel, 0.0, 1.0))
    return {"position_rad": q, "gripper_open": opening}, True


def tf_matrix(translation, rotation_xyzw) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(rotation_xyzw).as_matrix()
    T[:3, 3] = translation
    return T


def lookup_transform(edges: dict, parent: str, child: str) -> np.ndarray | None:
    """Compose ``parent <- child`` through a graph of static transforms keyed ``(parent, child)``."""
    pending, seen = [(parent, np.eye(4))], set()
    while pending:
        frame, matrix = pending.pop()
        if frame == child:
            return matrix
        if frame in seen:
            continue
        seen.add(frame)
        for (p, c), T in edges.items():
            if p == frame:
                pending.append((c, matrix @ T))
            elif c == frame:
                pending.append((p, matrix @ np.linalg.inv(T)))
    return None


@dataclass
class CameraFrame:
    """Latest decoded frame for one camera. ``T_color_depth`` is ``None`` until TF arrives."""

    color: np.ndarray | None = None
    depth: np.ndarray | None = None
    K_color: np.ndarray | None = None
    K_depth: np.ndarray | None = None
    color_frame_id: str | None = None
    depth_frame_id: str | None = None
    T_color_depth: np.ndarray | None = None
    color_arrival: float = 0.0
    depth_arrival: float = 0.0
    color_count: int = 0
    depth_count: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class ArmSample:
    observation: dict
    gripper_reported: bool
    arrival: float


class RosYamSource:
    """Background ``rclpy`` subscriber. ``latest()`` returns copies safe to use from another thread."""

    def __init__(self, cameras, *, joint_topic: str = "/joint_states", finger_travel: dict[str, float]):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CameraInfo, CompressedImage, JointState
        from tf2_msgs.msg import TFMessage

        if not rclpy.ok():
            rclpy.init()
        self._rclpy = rclpy
        self.finger_travel = finger_travel
        self._lock = threading.Lock()
        self._arms: dict[str, ArmSample] = {}
        self._cameras: dict[str, CameraFrame] = {ns: CameraFrame() for ns in cameras}
        self._tf_edges: dict[tuple[str, str], np.ndarray] = {}

        self.node = rclpy.create_node("lerobot_3d_yam_viewer")
        sensor_qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.node.create_subscription(JointState, joint_topic, self._on_joints, 10)
        self.node.create_subscription(
            TFMessage, "/tf_static", self._on_tf_static,
            QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        for ns in cameras:
            ns = ns.rstrip("/")
            self.node.create_subscription(
                CompressedImage, f"{ns}/{COLOR_TOPIC}", lambda m, n=ns: self._on_color(n, m), sensor_qos)
            self.node.create_subscription(
                CompressedImage, f"{ns}/{DEPTH_TOPIC}", lambda m, n=ns: self._on_depth(n, m), sensor_qos)
            self.node.create_subscription(
                CameraInfo, f"{ns}/{COLOR_INFO_TOPIC}", lambda m, n=ns: self._on_info(n, m, "color"), sensor_qos)
            self.node.create_subscription(
                CameraInfo, f"{ns}/{DEPTH_INFO_TOPIC}", lambda m, n=ns: self._on_info(n, m, "depth"), sensor_qos)

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    # --- callbacks (executor thread) ---------------------------------------------------------
    def _spin(self) -> None:
        while not self._stop.is_set() and self._rclpy.ok():
            self._executor.spin_once(timeout_sec=0.05)

    def _on_joints(self, msg) -> None:
        now = time.monotonic()
        with self._lock:
            for side in SIDES:
                parsed = joint_observation(msg.name, msg.position, side, self.finger_travel.get(side))
                if parsed is not None:
                    self._arms[side] = ArmSample(parsed[0], parsed[1], now)

    def _on_tf_static(self, msg) -> None:
        with self._lock:
            for t in msg.transforms:
                r, p = t.transform.rotation, t.transform.translation
                self._tf_edges[(t.header.frame_id, t.child_frame_id)] = tf_matrix(
                    [p.x, p.y, p.z], [r.x, r.y, r.z, r.w])
            self._resolve_extrinsics_locked()

    def _on_info(self, ns, msg, kind) -> None:
        K = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        with self._lock:
            cam = self._cameras[ns]
            if kind == "color":
                cam.K_color, cam.color_frame_id = K, msg.header.frame_id
            else:
                cam.K_depth, cam.depth_frame_id = K, msg.header.frame_id
            self._resolve_extrinsics_locked()

    def _resolve_extrinsics_locked(self) -> None:
        for cam in self._cameras.values():
            if cam.T_color_depth is None and cam.color_frame_id and cam.depth_frame_id:
                if cam.color_frame_id == cam.depth_frame_id:
                    cam.T_color_depth = np.eye(4)
                else:
                    cam.T_color_depth = lookup_transform(self._tf_edges, cam.color_frame_id, cam.depth_frame_id)

    def _on_color(self, ns, msg) -> None:
        try:
            image = decode_color(msg.format, msg.data)
        except ValueError as e:
            self._note_error(ns, str(e))
            return
        with self._lock:
            cam = self._cameras[ns]
            cam.color, cam.color_arrival = image, time.monotonic()
            cam.color_count += 1

    def _on_depth(self, ns, msg) -> None:
        try:
            depth = decode_depth(msg.format, msg.data)
        except ValueError as e:
            self._note_error(ns, str(e))
            return
        with self._lock:
            cam = self._cameras[ns]
            cam.depth, cam.depth_arrival = depth, time.monotonic()
            cam.depth_count += 1

    def _note_error(self, ns, text) -> None:
        with self._lock:
            errors = self._cameras[ns].errors
            if text not in errors:
                errors.append(text)

    # --- consumer API ------------------------------------------------------------------------
    def latest(self) -> tuple[dict[str, ArmSample], dict[str, CameraFrame]]:
        with self._lock:
            arms = {k: ArmSample(dict(v.observation), v.gripper_reported, v.arrival)
                    for k, v in self._arms.items()}
            cams = {k: CameraFrame(**vars(v)) for k, v in self._cameras.items()}
            for cam in cams.values():
                cam.errors = list(cam.errors)
        return arms, cams

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._executor.remove_node(self.node)
        self.node.destroy_node()
