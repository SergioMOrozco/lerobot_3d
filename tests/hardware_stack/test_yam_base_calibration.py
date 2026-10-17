"""Calibrating the YAM right arm's base pose from one fixed camera calibrated against each arm alone."""

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("urchin")
pytest.importorskip("open3d")

from lerobot_3d.icp import record_per_arm_calibrations
from lerobot_3d.paths import YAM_URDF_PATH
from lerobot_3d.point_clouds.yam_kinematics import YamArmKinematics
from lerobot_3d.point_clouds.yam_robot_state import YamRobotState, yam_base_poses

pytestmark = pytest.mark.hardware_stack


def _pose(xyz, euler_deg):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", euler_deg, degrees=True).as_matrix()
    T[:3, 3] = xyz
    return T


URDF_RIGHT = _pose([0, 0.61, 0], [0, 0, 0])  # where the bundled URDF puts the right base
TRUE_RIGHT = _pose([0.012, 0.596, -0.004], [0.4, -0.3, 2.0])  # where it really is
TRUE_CAMERA = _pose([0.014, 0.29, 0.661], [-150, 0, -90])  # top camera, world = left base


def _capture(tmp_path, arms, linked=()):
    out = tmp_path / "calibration_files"
    out.mkdir(exist_ok=True)
    info = {"robot_type": "yam", "arms": {s: np.asarray(p).tolist() for s, p in arms.items()},
            "linked_cameras": list(linked)}
    (out / "capture.json").write_text(json.dumps(info))
    return str(out)


def test_two_single_arm_calibrations_give_the_right_base_pose(tmp_path):
    extrinsics = {"camera_top": {"X_WC": TRUE_CAMERA.tolist()}}

    # 1. Against the left arm alone: the left base is the world origin, so ICP finds the true pose.
    record_per_arm_calibrations(extrinsics, ["camera_top"], _capture(tmp_path, {"left": np.eye(4)}))
    assert set(extrinsics["camera_top"]["per_arm"]) == {"left"} and "yam_right_base" not in extrinsics

    # 2. Against the right arm alone, drawn at its URDF pose: ICP lines the camera up with the right arm
    #    as drawn, i.e. finds the camera's true pose relative to the right base, placed at URDF_RIGHT.
    X_RC = np.linalg.inv(TRUE_RIGHT) @ TRUE_CAMERA
    extrinsics["camera_top"]["X_WC"] = (URDF_RIGHT @ X_RC).tolist()
    record_per_arm_calibrations(extrinsics, ["camera_top"], _capture(tmp_path, {"right": URDF_RIGHT}))

    np.testing.assert_allclose(extrinsics["yam_right_base"]["X_W_base"], TRUE_RIGHT, atol=1e-12)
    assert extrinsics["yam_right_base"]["from_camera"] == "camera_top"
    np.testing.assert_allclose(extrinsics["camera_top"]["X_WC"], TRUE_CAMERA, atol=1e-12)  # live pose: left
    np.testing.assert_allclose(yam_base_poses(extrinsics)["right"], TRUE_RIGHT, atol=1e-12)

    # Recalibrating the right arm with the calibrated base in use gives the same answer.
    extrinsics["camera_top"]["X_WC"] = (TRUE_RIGHT @ X_RC).tolist()
    record_per_arm_calibrations(extrinsics, ["camera_top"], _capture(tmp_path, {"right": TRUE_RIGHT}))
    np.testing.assert_allclose(extrinsics["yam_right_base"]["X_W_base"], TRUE_RIGHT, atol=1e-12)


def test_both_arm_and_so101_captures_and_wrist_cameras_keep_no_per_arm_pose(tmp_path):
    extrinsics = {"camera_top": {"X_WC": TRUE_CAMERA.tolist()}, "camera_left_wrist": {"X_WC": np.eye(4).tolist()}}
    record_per_arm_calibrations(extrinsics, ["camera_top"], str(tmp_path))  # no capture.json: SO101
    record_per_arm_calibrations(extrinsics, ["camera_top"],
                                _capture(tmp_path, {"left": np.eye(4), "right": URDF_RIGHT}))
    assert "per_arm" not in extrinsics["camera_top"]
    record_per_arm_calibrations(extrinsics, ["camera_top", "camera_left_wrist"],
                                _capture(tmp_path, {"left": np.eye(4)}, linked=["camera_left_wrist"]))
    assert "per_arm" in extrinsics["camera_top"] and "per_arm" not in extrinsics["camera_left_wrist"]


def test_calibrated_base_moves_the_drawn_arm_ik_and_wrist_camera_together():
    state = YamRobotState(YAM_URDF_PATH, "right", base_pose=TRUE_RIGHT)
    nominal = YamRobotState(YAM_URDF_PATH, "right")
    q = [0.3, 1.0, 1.2, 0.1, -0.2, 0.4]
    obs = {**dict(zip(state.keys, q)), "right_gripper.pos": 0.5}
    np.testing.assert_allclose(state.get_link_transform(obs, "right_base"), TRUE_RIGHT, atol=1e-12)
    snapshot = state.get_robot_snapshot(obs)
    np.testing.assert_allclose(snapshot.base_offset, TRUE_RIGHT[:3, 3])
    np.testing.assert_allclose(Rotation.from_quat(np.roll(snapshot.base_wxyz, -1)).as_matrix(), TRUE_RIGHT[:3, :3],
                               atol=1e-9)
    correction = TRUE_RIGHT @ np.linalg.inv(URDF_RIGHT)
    np.testing.assert_allclose(state.get_link_transform(obs, "right_camera"),
                               correction @ nominal.get_link_transform(obs, "right_camera"), atol=1e-12)
    ik = YamArmKinematics(YAM_URDF_PATH, "right", base_pose=TRUE_RIGHT)
    np.testing.assert_allclose(ik.fk(np.array(q)), state.get_link_transform(obs, "right_gripper"), atol=1e-9)
    left = YamRobotState(YAM_URDF_PATH, "left")  # no pose for the left arm: it stays the URDF's (origin)
    np.testing.assert_allclose(left.get_link_transform(left.rest_observation(), "left_base"), np.eye(4))


def test_realsense_extrinsics_skip_non_camera_entries(tmp_path):
    from lerobot_3d.point_clouds.camera_stream import MultiRealSenseStream

    path = tmp_path / "extrinsic_calibration.json"
    path.write_text(json.dumps({"044322073544": {"X_WC": np.eye(4).tolist()},
                                "yam_right_base": {"X_W_base": TRUE_RIGHT.tolist()}}))
    stream = object.__new__(MultiRealSenseStream)
    stream.extrinsics_file = str(path)
    stream.get_camera_extrinsics()
    assert list(stream.extrinsics) == ["044322073544"]
