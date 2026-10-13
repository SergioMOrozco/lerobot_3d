"""YAM over ROS 2 in the main pipeline, against a fake ROS hub: RGB-D helpers, ROS message parsing,
the YamFollower robot, ROS cameras, SystemStateViewer (observe, Capture, calibrated wrist mounts),
teleop steps and two-arm dataset recording."""

import json
import socket

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("urchin")
pytest.importorskip("viser")

from lerobot_3d import ros_hub as ros_hub_module
from lerobot_3d.point_clouds.rgbd import (
    colored_cloud,
    decode_color,
    decode_depth,
    depth_to_points,
    register_depth_to_color,
    sample_colors,
)
from lerobot_3d.ros_hub import ArmSample, CameraFrame, joint_observation, lookup_transform
from lerobot_3d.teleop_config import RosCameraConfig, TeleopSystemConfig, YamArmConfig

pytestmark = pytest.mark.hardware_stack

K = np.array([[400.0, 0, 320], [0, 400.0, 240], [0, 0, 1]])


def test_depth_to_points_backprojects_principal_point_and_drops_invalid():
    depth = np.zeros((480, 640), np.uint16)
    depth[240, 320] = 500  # 0.5 m at the principal point
    depth[240, 720 - 320] = 500  # x = (400-320)*0.5/400 = 0.1
    depth[0, 0] = 3000  # beyond max_depth
    pts = depth_to_points(depth, K, 0.001, max_depth=1.0)
    assert pts.shape == (2, 3)
    np.testing.assert_allclose(sorted(pts[:, 0]), [0.0, 0.1], atol=1e-6)
    np.testing.assert_allclose(pts[:, 2], 0.5)


def test_sample_colors_projects_and_grays_out_of_view():
    rgb = np.zeros((480, 640, 3), np.uint8)
    rgb[240, 320] = (255, 0, 0)
    pts = np.array([[0.0, 0.0, 1.0], [10.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    np.testing.assert_array_equal(sample_colors(pts, rgb, K), [[255, 0, 0], [128] * 3, [128] * 3])


def test_colored_cloud_applies_depth_to_color_extrinsics():
    depth = np.zeros((480, 640), np.uint16)
    depth[240, 320] = 1000
    T = np.eye(4)
    T[0, 3] = 0.05
    pts, colors = colored_cloud(depth, K, 0.001, None, None, T, stride=1)
    np.testing.assert_allclose(pts, [[0.05, 0.0, 1.0]])
    assert colors.dtype == np.uint8


def test_register_depth_to_color_applies_extrinsics_and_keeps_nearest():
    depth = np.zeros((480, 640), np.uint16)
    depth[240, 320] = 1000  # 1 m straight ahead of the depth camera
    T = np.eye(4)
    T[0, 3] = 0.04  # ...is at x = 4 cm in the color frame
    K_color = np.array([[500.0, 0, 160], [0, 500.0, 120], [0, 0, 1]])
    out = register_depth_to_color(depth, K, 0.001, T, K_color, (240, 320, 3))
    assert out.shape == (240, 320) and out.dtype == np.uint16
    assert out[120, 160 + 20] == 1000 and np.count_nonzero(out) == 1  # u = 0.04 * 500 / 1 + 160

    # Two depth pixels landing on one coarse color pixel: the nearer one wins.
    depth[240, 321] = 2000
    coarse = np.array([[100.0, 0, 160], [0, 100.0, 120], [0, 0, 1]])
    out = register_depth_to_color(depth, K, 0.001, np.eye(4), coarse, (240, 320))
    assert out[120, 160] == 1000 and np.count_nonzero(out) == 1


def test_decode_ros_compressed_formats():
    depth = (np.arange(48 * 64, dtype=np.uint16) * 7).reshape(48, 64)
    png = cv2.imencode(".png", depth)[1].tobytes()
    np.testing.assert_array_equal(decode_depth("16UC1; compressedDepth", b"\0" * 12 + png), depth)
    np.testing.assert_array_equal(decode_depth("png", png), depth)
    with pytest.raises(ValueError):
        decode_depth("16UC1; compressedDepth rvl", b"\0" * 40)
    bgr = np.zeros((8, 8, 3), np.uint8)
    bgr[..., 2] = 255  # red in BGR
    rgb = decode_color("bgr8; png compressed", cv2.imencode(".png", bgr)[1].tobytes())
    assert tuple(rgb[0, 0]) == (255, 0, 0)


def test_joint_observation_uses_lerobot_keys_and_finger_travel_in_metres():
    names = [f"left_joint{i}" for i in range(1, 9)] + [f"right_joint{i}" for i in range(1, 7)]
    positions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.02, 0.02, 1, 2, 3, 4, 5, 6]
    left, reported = joint_observation(names, positions, "left")
    assert reported and left == {"left_joint1.pos": 0.1, "left_joint2.pos": 0.2, "left_joint3.pos": 0.3,
                                 "left_joint4.pos": 0.4, "left_joint5.pos": 0.5, "left_joint6.pos": 0.6,
                                 "left_gripper.pos": 0.02}
    right, reported = joint_observation(names, positions, "right")
    assert not reported and right["right_gripper.pos"] == 0.0
    assert joint_observation(names[:3], positions[:3], "left") is None


def test_lookup_transform_composes_and_inverts():
    a = np.eye(4)
    a[:3, 3] = [1, 0, 0]
    b = np.eye(4)
    b[:3, 3] = [0, 2, 0]
    edges = {("link", "color"): a, ("link", "depth"): b}
    np.testing.assert_allclose(lookup_transform(edges, "color", "depth")[:3, 3], [-1, 2, 0])
    assert lookup_transform(edges, "color", "nowhere") is None


# --- the integrated system, against a fake ROS hub ------------------------------------------------

LEFT = (0.3, 1.0, 1.2, 0.1, -0.2, 0.4)
RIGHT = (-0.2, 0.8, 1.0, 0.0, 0.3, -0.1)


def _arm(side, q, finger=0.02):
    obs = {**{f"{side}_joint{i}.pos": v for i, v in enumerate(q, 1)}, f"{side}_gripper.pos": finger}
    return ArmSample(obs, True, 1e12)


def _camera():
    depth = np.zeros((480, 640), np.uint16)
    depth[200:280, 280:360] = 300
    color = np.full((480, 640, 3), 90, np.uint8)
    return CameraFrame(color=color, depth=depth, K_color=K, K_depth=K, color_frame_id="c",
                       depth_frame_id="c", T_color_depth=np.eye(4), color_arrival=1e12,
                       depth_arrival=1e12, color_count=1, depth_count=1)


class FakeHub:
    """Stands in for RosHub: serves fixed arm samples and camera frames."""

    def __init__(self, arms, cameras):
        self.arms, self.cameras = arms, cameras
        self.joint_topics, self.watched = set(), []

    def watch_joints(self, topic):
        self.joint_topics.add(topic)

    def watch_camera(self, namespace, **topics):
        self.watched.append(namespace)

    def latest_arm(self, topic, side):
        return self.arms.get(side) if topic in self.joint_topics else None

    def latest_camera(self, namespace):
        return self.cameras.get(namespace)

    def close(self):
        pass


@pytest.fixture
def hub(monkeypatch):
    hub = FakeHub(
        {"left": _arm("left", LEFT), "right": _arm("right", RIGHT)},
        {ns: _camera() for ns in ("/camera/top", "/camera/left_wrist", "/camera/right_wrist")},
    )
    monkeypatch.setattr(ros_hub_module, "_hub", hub)
    return hub


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


TOP_X_WC = np.array([[1.0, 0, 0, 0.3], [0, -1.0, 0, -0.2], [0, 0, -1.0, 1.1], [0, 0, 0, 1]])


@pytest.fixture
def yam_config(tmp_path, monkeypatch):
    """Both arms plus an overhead camera and two wrist cameras, in a scratch working directory."""
    monkeypatch.chdir(tmp_path)
    extrinsics = tmp_path / "extrinsic_calibration.json"
    extrinsics.write_text(json.dumps({"camera_top": {"X_WC": TOP_X_WC.tolist()},
                                      "044322073544": {"X_WC": np.eye(4).tolist()}}))

    def make(**overrides):
        kwargs = dict(
            leaders=(), followers=(YamArmConfig("left"), YamArmConfig("right")), realsense_serials=(),
            robot_type="yam",
            ros_cameras=(RosCameraConfig("/camera/top"), RosCameraConfig("/camera/left_wrist", link="left_camera"),
                         RosCameraConfig("/camera/right_wrist", link="right_camera")),
            extrinsic_json=str(extrinsics), viser_port=_free_port(), segment_on_capture=False,
        )
        kwargs.update(overrides)
        return TeleopSystemConfig(**kwargs)

    return make


def _fk(viewer, side, link, q):
    state = viewer.robot_states[["left", "right"].index(side)]
    return state.get_link_transform(_arm(side, q).observation, link)


def test_yam_follower_is_an_observe_only_lerobot_robot(hub):
    from lerobot.robots.config import RobotConfig

    from lerobot_3d.robots.yam_follower import YamFollower, YamFollowerConfig

    assert RobotConfig.get_choice_class("yam_follower") is YamFollowerConfig
    robot = YamFollower(YamFollowerConfig(side="right", id="yam_right"))
    assert list(robot.observation_features) == [f"right_joint{i}.pos" for i in range(1, 7)] + ["right_gripper.pos"]
    robot.connect()
    assert robot.is_connected and robot.is_calibrated
    assert robot.get_observation() == hub.arms["right"].observation
    with pytest.raises(NotImplementedError, match="observe-only"):
        robot.send_action(hub.arms["right"].observation)

    del hub.arms["right"]
    with pytest.raises(ConnectionError, match="right_joint1..6"):
        YamFollower(YamFollowerConfig(side="right", id="yam_right", connect_timeout_s=0.1)).connect()


def test_viewer_observes_both_arms_and_places_fixed_and_wrist_cameras(hub, yam_config):
    from lerobot_3d.point_clouds.system_vis import SystemStateViewer

    viewer = SystemStateViewer(yam_config())
    try:
        datapoints, _, robot_pcds, _, snapshots = viewer.update()
        assert [s.joint_positions for s in snapshots] == [hub.arms["left"].observation, hub.arms["right"].observation]
        assert len(robot_pcds) == 2
        by_name = {dp.serial: dp for dp in datapoints}
        assert list(by_name) == ["camera_top", "camera_left_wrist", "camera_right_wrist"]
        np.testing.assert_allclose(by_name["camera_top"].X_WC, TOP_X_WC)  # fixed, from the extrinsics file
        # Wrist cameras sit exactly at their URDF camera links until calibrated.
        np.testing.assert_allclose(by_name["camera_left_wrist"].X_WC, _fk(viewer, "left", "left_camera", LEFT))
        np.testing.assert_allclose(by_name["camera_right_wrist"].X_WC, _fk(viewer, "right", "right_camera", RIGHT))
        dp = by_name["camera_top"]
        assert dp.depth_scale == 0.001 and dp.color_intrinsics.fx == 400.0
        assert dp.joint_positions == {**hub.arms["left"].observation, **hub.arms["right"].observation}
        with pytest.raises(ValueError, match="observe-only"):
            viewer.update(hub.arms["left"].observation, hub.arms["right"].observation)
    finally:
        viewer.close()


def test_fixed_ros_camera_needs_an_extrinsics_entry(hub, yam_config):
    from lerobot_3d.point_clouds.system_vis import SystemStateViewer

    with pytest.raises(ValueError, match="no 'camera_side' entry"):
        SystemStateViewer(yam_config(ros_cameras=(RosCameraConfig("/camera/side"),)))


def test_capture_writes_icp_inputs_and_calibrated_wrist_mount_follows_the_arm(hub, yam_config, tmp_path,
                                                                              monkeypatch):
    from lerobot_3d.point_clouds.system_vis import SystemStateViewer

    viewer = SystemStateViewer(yam_config())
    try:
        viewer.viewer.capture = True
        viewer.update()
        fk_left = _fk(viewer, "left", "left_camera", LEFT)
    finally:
        viewer.close()

    out = tmp_path / "calibration_files"
    assert sorted(p.name for p in out.iterdir()) == ["camera_left_wrist", "camera_right_wrist", "camera_top",
                                                    "robot_pcd.npz"]
    with np.load(out / "camera_left_wrist" / "depth.npz") as data:
        np.testing.assert_array_equal(data["depth"], _camera().depth)
    with np.load(out / "robot_pcd.npz") as data:
        pcd = data["pcd"]
    left_only = viewer.robot_states[0].get_mesh_points(hub.arms["left"].observation)
    assert len(pcd) == 2 * len(left_only)  # both arms are in the ICP target
    intrinsics = json.loads((tmp_path / "intrinsic_calibration.json").read_text())
    assert set(intrinsics) == {"camera_top", "camera_left_wrist", "camera_right_wrist"}
    extrinsics = json.loads((tmp_path / "extrinsic_calibration.json").read_text())
    np.testing.assert_allclose(extrinsics["camera_top"]["X_WC"], TOP_X_WC)  # fixed cameras untouched
    assert "044322073544" in extrinsics
    entry = extrinsics["camera_left_wrist"]
    assert entry["link"] == "left_camera"
    np.testing.assert_allclose(entry["X_WL"], fk_left)
    np.testing.assert_allclose(entry["X_WC"], fk_left)  # seeded at the URDF mount

    # icp.py refines X_WC; at new joint angles the camera keeps that offset from its link.
    offset = np.eye(4)
    offset[:3, 3] = [0.01, -0.02, 0.005]
    entry["X_WC"] = (fk_left @ offset).tolist()
    (tmp_path / "extrinsic_calibration.json").write_text(json.dumps(extrinsics))
    moved = (-0.4, 0.6, 1.0, 0.0, 0.3, -0.5)
    hub.arms["left"] = _arm("left", moved)
    monkeypatch.setattr(ros_hub_module, "_hub", hub)  # close() shut the shared hub down
    viewer = SystemStateViewer(yam_config())
    try:
        by_name = {dp.serial: dp for dp in viewer.update()[0]}
        np.testing.assert_allclose(by_name["camera_left_wrist"].X_WC,
                                   _fk(viewer, "left", "left_camera", moved) @ offset, atol=1e-9)
        np.testing.assert_allclose(by_name["camera_right_wrist"].X_WC, _fk(viewer, "right", "right_camera", RIGHT))
    finally:
        viewer.close()


def test_teleop_step_records_measured_joints_into_a_two_arm_dataset(hub, yam_config, tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from lerobot_3d.control.teleop import TeleopPointCloudSystem
    from lerobot_3d.recording.dataset_loader import frame_link_pcds, frame_to_datapoints, read_sidecar
    from lerobot_3d.recording.dataset_recorder import LeRobotDatasetRecorder, world_link_pcds

    system = TeleopPointCloudSystem(yam_config())
    recorder = LeRobotDatasetRecorder("local/yam_test", root=tmp_path / "ds", fps=15, robot_type="yam_follower")
    try:
        system.connect()
        recorder.start_episode()
        for _ in range(3):
            datapoints, _, _, _, snapshots = system.step()
            # No teleop: the measured joints are the action.
            assert system.last_actions == [hub.arms["left"].observation, hub.arms["right"].observation]
            recorder.add_frame(datapoints, snapshots, system.last_actions)
        expected_links = world_link_pcds(snapshots)
        recorder.stop_episode()
        recorder.save_episode()
    finally:
        recorder.finalize()
        system.close()

    ds = LeRobotDataset("local/yam_test", root=tmp_path / "ds")
    names = ds.meta.features["action"]["names"]
    assert names == list(hub.arms["left"].observation) + list(hub.arms["right"].observation)
    assert ds.meta.features["observation.state"]["names"] == names
    item = ds[0]
    np.testing.assert_allclose(item["action"].numpy(), [*LEFT, 0.02, *RIGHT, 0.02], rtol=1e-6)
    links = frame_link_pcds(item)
    assert any(n.startswith("left_") for n in links) and any(n.startswith("right_") for n in links)
    for name, pts in links.items():  # world frame: each arm's station pose applied
        np.testing.assert_allclose(pts, expected_links[name], atol=1e-5)
    serials = [dp.serial for dp in frame_to_datapoints(item)]
    assert serials == ["camera_top", "camera_left_wrist", "camera_right_wrist"]
    assert read_sidecar(tmp_path / "ds")["world_frame"].startswith("YAM station URDF world")


def test_yam_without_followers_previews_the_rest_pose_without_ros(monkeypatch, yam_config):
    from lerobot_3d.point_clouds.system_vis import SystemStateViewer

    monkeypatch.setattr(ros_hub_module, "_hub", None)
    monkeypatch.setattr(ros_hub_module, "RosHub", lambda: pytest.fail("ROS must not start"))
    viewer = SystemStateViewer(yam_config(followers=(), ros_cameras=()))
    try:
        assert not viewer.observe_only and len(viewer.robot_states) == 2
        _, _, _, _, snapshots = viewer.update(*viewer.rest_actions())
        assert [s.joint_positions for s in snapshots] == [s.rest_observation() for s in viewer.robot_states]
    finally:
        viewer.close()
