"""Live YAM viewer: RGB-D helpers, ROS message parsing and one viewer tick with a fake source."""

import os
import socket
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from lerobot_3d.point_clouds.rgbd import (
    colored_cloud,
    decode_color,
    decode_depth,
    depth_to_points,
    sample_colors,
)
from lerobot_3d.point_clouds.ros_yam_source import (
    ArmSample,
    CameraFrame,
    joint_observation,
    lookup_transform,
)

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


def test_joint_observation_uses_names_and_measured_finger_travel():
    names = [f"left_joint{i}" for i in range(1, 9)] + [f"right_joint{i}" for i in range(1, 7)]
    positions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.02, 0.02, 1, 2, 3, 4, 5, 6]
    left, reported = joint_observation(names, positions, "left", 0.04)
    assert reported and left["gripper_open"] == pytest.approx(0.5)
    np.testing.assert_allclose(left["position_rad"], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    right, reported = joint_observation(names, positions, "right", 0.04)
    assert not reported and right["gripper_open"] == 0.0
    assert joint_observation(names[:3], positions[:3], "left", 0.04) is None


def test_lookup_transform_composes_and_inverts():
    a = np.eye(4)
    a[:3, 3] = [1, 0, 0]
    b = np.eye(4)
    b[:3, 3] = [0, 2, 0]
    edges = {("link", "color"): a, ("link", "depth"): b}
    np.testing.assert_allclose(lookup_transform(edges, "color", "depth")[:3, 3], [-1, 2, 0])
    assert lookup_transform(edges, "color", "nowhere") is None


# --- viewer, against the real station URDF -------------------------------------------------------

@pytest.fixture
def urdf():
    pytest.importorskip("urchin")
    pytest.importorskip("viser")
    checkout = os.environ.get("I2RT_CHECKOUT")
    if not checkout:
        pytest.skip("Set I2RT_CHECKOUT to the pinned vendor checkout for real URDF tests")
    return (Path(checkout)
            / "i2rt/robot_models/station/yam_station_linear_4310_d405/yam_station_linear_4310_d405.urdf")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeSource:
    def __init__(self, arms, cams):
        self.arms, self.cams = arms, cams

    def latest(self):
        return self.arms, self.cams


def _camera():
    depth = np.zeros((480, 640), np.uint16)
    depth[200:280, 280:360] = 300
    color = np.full((480, 640, 3), 90, np.uint8)
    return CameraFrame(color=color, depth=depth, K_color=K, K_depth=K, color_frame_id="c",
                       depth_frame_id="c", T_color_depth=np.eye(4), color_arrival=1e12,
                       depth_arrival=1e12, color_count=1, depth_count=1)


def test_viewer_tick_places_camera_cloud_at_measured_fk(urdf):
    from lerobot_3d.point_clouds.yam_live_viewer import YamLiveViewer

    obs = {"position_rad": np.array([0.3, 1.0, 1.2, 0.1, -0.2, 0.4]), "gripper_open": 0.5}
    source = FakeSource({"left": ArmSample(obs, True, 1e12)}, {"/camera/left_wrist": _camera()})
    viewer = YamLiveViewer(urdf, source, {"/camera/left_wrist": "left_camera",
                                          "/camera/right_wrist": "right_camera"},
                           port=_free_port())
    try:
        viewer.tick()
        node = viewer._cam_nodes["/camera/left_wrist"]
        state = viewer.states["left"]
        expected = state.robot_urdf.link_fk(cfg=state.get_joint_positions(obs), use_names=True)["left_camera"]
        np.testing.assert_allclose(node["frame"].position, expected[:3, 3], atol=1e-6)
        assert node["frame"].visible and node["cloud"] is not None
        assert len(node["cloud"].points) == (80 // 4) * (80 // 4)
        status = viewer._status.content
        assert "0.50 open (from left_joint7)" in status and "waiting for `right_joint1..6`" in status
        assert "nominal URDF mount" in status
        # Joints outside the URDF limits are reported, not clipped or drawn.
        source.arms["left"] = ArmSample({**obs, "position_rad": np.full(6, 9.0)}, True, 1e12)
        viewer.tick()
        assert "left**: not drawn" in viewer._status.content
    finally:
        viewer.close()
