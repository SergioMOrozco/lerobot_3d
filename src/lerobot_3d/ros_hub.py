"""Process-wide, read-only ROS 2 subscriber for YAM joint states and RGB-D cameras.

One node serves every consumer in the process: each :class:`~lerobot_3d.robots.yam_follower.YamFollower`
arm and each ROS camera asks :func:`ros_hub` for what it needs, and subscriptions are made once.
Only the latest sample (and its arrival time) is kept per topic, so consumers can judge staleness.

Joints: ``sensor_msgs/JointState`` per arm (e.g. ``/left/joint_states``) with the YAM station URDF names:
``<side>_joint1..6`` in radians and ``<side>_gripper`` as an opening from 0 (closed) to 1 (open); a
``<side>_joint7`` finger travel in metres is accepted instead. Commands go out the same way, as a
``JointState`` on the arm's command topic (e.g. ``/left/cmd_joint``). Cameras: per namespace, compressed
color and lossless compressed depth, both ``camera_info`` topics, and depth->color extrinsics from
``/tf_static``.

``rclpy`` is imported only when the hub is first created, so nothing here needs ROS installed
until a YAM robot or ROS camera is actually configured.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from lerobot_3d.point_clouds.rgbd import decode_color, decode_depth
from lerobot_3d.teleop_config import (
    ROS_COLOR_INFO_TOPIC,
    ROS_COLOR_TOPIC,
    ROS_DEPTH_INFO_TOPIC,
    ROS_DEPTH_TOPIC,
    YAM_SIDES,
)

YAM_FINGER_TRAVEL_M = 0.04695
"""Full linear_4310 finger travel (the station URDF's ``<side>_joint7`` upper limit), for converting a
``<side>_joint7`` reading in metres to a 0..1 opening."""


def joint_observation(names, positions, side: str) -> tuple[dict[str, float], bool] | None:
    """``JointState`` names/positions -> one YAM arm's observation, or ``None`` if it's absent.

    Returns ``(observation, gripper_reported)`` with ``observation`` keyed like a LeRobot robot:
    ``"<side>_joint1.pos" .. "<side>_joint6.pos"`` (radians) and ``"<side>_gripper.pos"`` (opening,
    0 closed .. 1 open) from ``<side>_gripper``, or else from ``<side>_joint7`` metres. With neither,
    the gripper reads closed (``0.0``) and ``gripper_reported`` is False.
    """
    by_name = dict(zip(names, positions))
    arm = [f"{side}_joint{i}" for i in range(1, 7)]
    if not all(n in by_name for n in arm):
        return None
    observation = {f"{n}.pos": float(by_name[n]) for n in arm}
    if f"{side}_gripper" in by_name:
        opening = float(by_name[f"{side}_gripper"])
    elif f"{side}_joint7" in by_name:
        opening = float(by_name[f"{side}_joint7"]) / YAM_FINGER_TRAVEL_M
    else:
        opening = None
    observation[f"{side}_gripper.pos"] = 0.0 if opening is None else opening
    return observation, opening is not None


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
    """``(H, W, 3)`` uint8 **RGB**."""
    depth: np.ndarray | None = None
    """``(h, w)`` uint16 millimetres, in the depth camera's own image grid."""
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

    @property
    def ready(self) -> bool:
        """Has everything needed to build a color-registered RGB-D frame."""
        return (
            self.color is not None
            and self.depth is not None
            and self.K_color is not None
            and self.K_depth is not None
            and self.T_color_depth is not None
        )


@dataclass
class ArmSample:
    observation: dict[str, float]
    gripper_reported: bool
    arrival: float


class RosHub:
    """Background ``rclpy`` node; ``watch_*`` subscribe once, ``latest_*`` return safe copies."""

    def __init__(self):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor

        if not rclpy.ok():
            rclpy.init()
        self._rclpy = rclpy
        self._lock = threading.Lock()
        """Guards the latest-sample state below (callbacks vs. consumers)."""
        self._spin_lock = threading.Lock()
        """Serializes spinning with creating subscriptions."""
        self._arms: dict[tuple[str, str], ArmSample] = {}
        self._joint_topics: set[str] = set()
        self._cameras: dict[str, CameraFrame] = {}
        self._tf_edges: dict[tuple[str, str], np.ndarray] = {}
        self._tf_subscribed = False
        self._publishers: dict[str, object] = {}

        self.node = rclpy.create_node("lerobot_3d")
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    # --- subscriptions -----------------------------------------------------------------------
    def watch_joints(self, topic: str) -> None:
        from sensor_msgs.msg import JointState

        with self._spin_lock:
            if topic in self._joint_topics:
                return
            self._joint_topics.add(topic)
            self.node.create_subscription(JointState, topic, lambda m, t=topic: self._on_joints(t, m), 10)

    def watch_camera(
        self,
        namespace: str,
        *,
        color_topic: str = ROS_COLOR_TOPIC,
        depth_topic: str = ROS_DEPTH_TOPIC,
        color_info_topic: str = ROS_COLOR_INFO_TOPIC,
        depth_info_topic: str = ROS_DEPTH_INFO_TOPIC,
    ) -> None:
        """Subscribe to one camera; topic names are relative to ``namespace``."""
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CameraInfo, CompressedImage
        from tf2_msgs.msg import TFMessage

        ns = namespace.rstrip("/")
        sensor_qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
        with self._spin_lock:
            if ns in self._cameras:
                return
            with self._lock:
                self._cameras[ns] = CameraFrame()
            if not self._tf_subscribed:
                self._tf_subscribed = True
                self.node.create_subscription(
                    TFMessage, "/tf_static", self._on_tf_static,
                    QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL),
                )
            self.node.create_subscription(
                CompressedImage, f"{ns}/{color_topic}", lambda m: self._on_color(ns, m), sensor_qos)
            self.node.create_subscription(
                CompressedImage, f"{ns}/{depth_topic}", lambda m: self._on_depth(ns, m), sensor_qos)
            self.node.create_subscription(
                CameraInfo, f"{ns}/{color_info_topic}", lambda m: self._on_info(ns, m, "color"), sensor_qos)
            self.node.create_subscription(
                CameraInfo, f"{ns}/{depth_info_topic}", lambda m: self._on_info(ns, m, "depth"), sensor_qos)

    def publish_joint_command(self, topic: str, names, positions) -> None:
        """Publish a ``sensor_msgs/JointState`` command (names + positions only) on ``topic``."""
        from sensor_msgs.msg import JointState

        with self._spin_lock:
            publisher = self._publishers.get(topic)
            if publisher is None:
                publisher = self._publishers[topic] = self.node.create_publisher(JointState, topic, 10)
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = list(names)
        msg.position = [float(p) for p in positions]
        publisher.publish(msg)

    # --- callbacks (spin thread) ----------------------------------------------------------------
    def _spin(self) -> None:
        while not self._stop.is_set() and self._rclpy.ok():
            with self._spin_lock:
                self._executor.spin_once(timeout_sec=0.02)
            time.sleep(0.001)  # let watch_* take the lock between spins

    def _on_joints(self, topic, msg) -> None:
        now = time.monotonic()
        with self._lock:
            for side in YAM_SIDES:
                parsed = joint_observation(msg.name, msg.position, side)
                if parsed is not None:
                    self._arms[(topic, side)] = ArmSample(parsed[0], parsed[1], now)

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
    def latest_arm(self, topic: str, side: str) -> ArmSample | None:
        with self._lock:
            sample = self._arms.get((topic, side))
            return None if sample is None else ArmSample(dict(sample.observation), sample.gripper_reported,
                                                         sample.arrival)

    def latest_camera(self, namespace: str) -> CameraFrame | None:
        with self._lock:
            cam = self._cameras.get(namespace.rstrip("/"))
            if cam is None:
                return None
            copy = CameraFrame(**vars(cam))
            copy.errors = list(cam.errors)
            return copy

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._executor.remove_node(self.node)
        self.node.destroy_node()


def wait_until(predicate, timeout_s: float, poll_s: float = 0.05) -> bool:
    """Poll ``predicate()`` until it's truthy or ``timeout_s`` passes; returns the last result."""
    deadline = time.monotonic() + timeout_s
    while True:
        if predicate():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll_s)


_hub: RosHub | None = None
_hub_lock = threading.Lock()


def ros_hub() -> RosHub:
    """The process's shared :class:`RosHub`, created on first use."""
    global _hub
    with _hub_lock:
        if _hub is None:
            _hub = RosHub()
        return _hub


def shutdown_ros_hub() -> None:
    """Stop the shared hub if one was created (safe to call when ROS was never used)."""
    global _hub
    with _hub_lock:
        if _hub is not None:
            _hub.close()
            _hub = None
