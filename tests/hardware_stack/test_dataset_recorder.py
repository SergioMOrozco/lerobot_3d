"""End-to-end LeRobotDataset recording with synthetic frames (no cameras or robot needed):
record -> finalize -> reload -> rebuild Datapoints -> fuse, and append to an existing dataset.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("open3d")
pytest.importorskip("lerobot")
pytest.importorskip("cv2")

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lerobot_3d.common.types import Datapoint, RobotSnapshot
from lerobot_3d.point_clouds.camera_stream import get_fused_point_cloud
from lerobot_3d.recording.dataset_loader import (
    frame_joint_positions,
    frame_link_pcds,
    frame_to_datapoints,
    read_fixed_extrinsics,
    read_sidecar,
)
from lerobot_3d.recording.dataset_recorder import EXTRINSICS_PREFIX, LeRobotDatasetRecorder
from lerobot_3d.table_plane import table_plane_from_extrinsics

pytestmark = pytest.mark.hardware_stack

H, W = 48, 64
SERIALS = ["111", "222"]
JOINTS = ["shoulder_pan.pos", "gripper.pos"]
REPO_ID = "local/test_recorder"


def _step_data(rng, t):
    datapoints = []
    for i, serial in enumerate(SERIALS):
        X_WC = np.eye(4)
        X_WC[:3, 3] = [0.1 * i, 0.2, 0.5]
        datapoints.append(
            Datapoint(
                serial=serial,
                color=rng.integers(0, 256, (H, W, 3), dtype=np.uint8),
                depth=rng.integers(200, 3000, (H, W), dtype=np.uint16),
                depth_scale=0.001,
                max_depth=10.0,
                X_WC=X_WC,
                color_intrinsics=SimpleNamespace(fx=60.0 + i, fy=61.0, ppx=32.0, ppy=24.0),
            )
        )
    snapshot = RobotSnapshot(
        index=0,
        joint_positions={"shoulder_pan.pos": float(t), "gripper.pos": 50.0},
        joint_radians={},
        pcd=np.zeros((0, 3)),
        link_pcds={
            "base": rng.random((100, 3)),
            "gripper_link": rng.random((200, 3)),
        },
        link_poses={},
        base_offset=np.zeros(3),
    )
    action = {"shoulder_pan.pos": float(t) + 0.5, "gripper.pos": 40.0}
    return datapoints, snapshot, action


def _record(recorder, rng, n_frames, keep, log):
    recorder.start_episode()
    for t in range(n_frames):
        step = _step_data(rng, t)
        recorder.add_frame(*step)
        if keep:
            log.append(step)
    recorder.stop_episode()
    if keep:
        recorder.save_episode()
    else:
        recorder.discard_episode()


@pytest.fixture
def recorded(tmp_path):
    root = tmp_path / "ds"
    rng = np.random.default_rng(0)
    kept = []
    recorder = LeRobotDatasetRecorder(REPO_ID, root=root, fps=10, task="push")
    _record(recorder, rng, 4, keep=True, log=kept)
    _record(recorder, rng, 3, keep=False, log=kept)
    _record(recorder, rng, 5, keep=True, log=kept)
    recorder.finalize()
    return root, kept


def test_discarded_episode_is_not_saved(recorded):
    root, kept = recorded

    ds = LeRobotDataset(REPO_ID, root=root)

    assert ds.meta.total_episodes == 2
    assert len(ds) == len(kept) == 9


def test_frames_roundtrip(recorded):
    root, kept = recorded
    ds = LeRobotDataset(REPO_ID, root=root)
    joint_names = ds.meta.features["observation.state"]["names"]
    extrinsics = read_fixed_extrinsics(root)

    for i, (datapoints, snapshot, action) in enumerate(kept):
        item = ds[i]
        restored = frame_to_datapoints(item, joint_names=joint_names, extrinsics=extrinsics)

        assert [dp.serial for dp in restored] == SERIALS
        for orig, back in zip(datapoints, restored):
            assert np.array_equal(back.depth, orig.depth)  # lossless
            assert back.depth_scale == pytest.approx(orig.depth_scale)
            assert np.allclose(back.X_WC, orig.X_WC)
            assert back.color_intrinsics.fx == pytest.approx(orig.color_intrinsics.fx)
            assert back.color_intrinsics.ppy == pytest.approx(orig.color_intrinsics.ppy)
            assert back.color.shape == orig.color.shape and back.color.dtype == np.uint8
            assert back.joint_positions == snapshot.joint_positions

        links = frame_link_pcds(item)
        assert set(links) == set(snapshot.link_pcds)
        for name, pts in snapshot.link_pcds.items():
            assert np.allclose(links[name], pts, atol=1e-6)

        assert np.allclose(item["action"].numpy(), [action[k] for k in JOINTS])
        assert frame_joint_positions(item, joint_names) == snapshot.joint_positions
        assert item["task"] == "push"


def test_fused_cloud_from_dataset_matches_live(recorded):
    root, kept = recorded
    ds = LeRobotDataset(REPO_ID, root=root)
    datapoints, _, _ = kept[0]

    live, _ = get_fused_point_cloud(datapoints)
    replay, _ = get_fused_point_cloud(frame_to_datapoints(ds[0], extrinsics=read_fixed_extrinsics(root)))

    # float32 storage of depth_scale: sub-micron differences only.
    assert np.allclose(np.asarray(live.points), np.asarray(replay.points), atol=1e-6)


def test_sidecar_describes_dataset(recorded):
    root, _ = recorded

    sidecar = read_sidecar(root)

    assert sidecar["depth_encoding"] == "uint16_packed_rg8"
    assert sidecar["camera_serials"] == SERIALS
    assert sidecar["robot_links"] == ["base", "gripper_link"]
    assert sidecar["joint_names"] == JOINTS


def test_fixed_extrinsics_are_stored_once_at_full_precision(recorded):
    root, kept = recorded
    datapoints, _, _ = kept[0]

    ds = LeRobotDataset(REPO_ID, root=root)
    fixed = read_fixed_extrinsics(root)

    assert not [key for key in ds.meta.features if key.startswith(EXTRINSICS_PREFIX)]
    assert list(fixed) == SERIALS
    for dp in datapoints:
        assert np.array_equal(fixed[dp.serial], dp.X_WC)


def test_loading_fixed_cameras_without_extrinsics_raises(recorded):
    root, _ = recorded
    ds = LeRobotDataset(REPO_ID, root=root)

    with pytest.raises(KeyError, match="read_fixed_extrinsics"):
        frame_to_datapoints(ds[0])


def test_moving_camera_keeps_per_frame_extrinsics(tmp_path):
    root = tmp_path / "ds"
    rng = np.random.default_rng(0)
    recorder = LeRobotDatasetRecorder(REPO_ID, root=root, fps=10, task="push")
    recorder.start_episode()
    poses = []
    for t in range(3):
        datapoints, snapshot, action = _step_data(rng, t)
        datapoints[1].moving = True
        datapoints[1].X_WC = datapoints[1].X_WC.copy()
        datapoints[1].X_WC[2, 3] += 0.01 * t
        poses.append(datapoints[1].X_WC)
        recorder.add_frame(datapoints, snapshot, action)
    recorder.stop_episode()
    recorder.save_episode()
    recorder.finalize()

    ds = LeRobotDataset(REPO_ID, root=root)
    extrinsics = read_fixed_extrinsics(root)

    assert list(extrinsics) == [SERIALS[0]]
    for i, X_WC in enumerate(poses):
        fixed_cam, moving_cam = frame_to_datapoints(ds[i], extrinsics=extrinsics)
        assert not fixed_cam.moving and moving_cam.moving
        assert np.allclose(moving_cam.X_WC, X_WC)


def test_fixed_camera_that_moves_raises(tmp_path):
    rng = np.random.default_rng(0)
    recorder = LeRobotDatasetRecorder(REPO_ID, root=tmp_path / "ds", fps=10)
    recorder.start_episode()
    recorder.add_frame(*_step_data(rng, 0))
    datapoints, snapshot, action = _step_data(rng, 1)
    datapoints[0].X_WC[0, 3] += 0.01

    with pytest.raises(ValueError, match="extrinsics"):
        recorder.add_frame(datapoints, snapshot, action)
    recorder.finalize()


def test_reopening_after_recalibration_raises(recorded):
    root, _ = recorded
    recorder = LeRobotDatasetRecorder(REPO_ID, root=root, fps=10)
    datapoints, snapshot, action = _step_data(np.random.default_rng(2), 0)
    datapoints[0].X_WC[0, 3] += 0.01

    recorder.start_episode()
    with pytest.raises(ValueError, match="extrinsics"):
        recorder.add_frame(datapoints, snapshot, action)
    recorder.finalize()


def test_appending_to_dataset_with_per_frame_extrinsics_keeps_them(tmp_path):
    root = tmp_path / "ds"
    rng = np.random.default_rng(0)
    recorder = LeRobotDatasetRecorder(REPO_ID, root=root, fps=10, task="push")
    recorder.start_episode()
    for t in range(2):
        datapoints, snapshot, action = _step_data(rng, t)
        for dp in datapoints:
            dp.moving = True  # the layout of a dataset recorded before fixed extrinsics existed
        recorder.add_frame(datapoints, snapshot, action)
    recorder.stop_episode()
    recorder.save_episode()
    recorder.finalize()

    recorder = LeRobotDatasetRecorder(REPO_ID, root=root, fps=10, task="push")
    _record(recorder, rng, 2, keep=True, log=[])
    recorder.finalize()

    ds = LeRobotDataset(REPO_ID, root=root)
    assert len(ds) == 4
    assert all(dp.moving for dp in frame_to_datapoints(ds[3]))


@pytest.fixture
def calibration_files(tmp_path):
    extrinsic_json = tmp_path / "extrinsic_calibration.json"
    intrinsic_json = tmp_path / "intrinsic_calibration.json"
    extrinsic_json.write_text(json.dumps({
        "111": {"X_WC": np.eye(4).tolist()},
        "camera_top": {"X_WC": np.eye(4).tolist(), "per_arm": {"left": np.eye(4).tolist()}},
        "yam_right_base": {"X_W_base": np.eye(4).tolist(), "from_camera": "camera_top"},
        "table_plane": {
            "X_W_table": np.eye(4).tolist(), "normal": [0.0, 0.0, 1.0], "offset": 0.1,
            "extent": [0.5, 0.4], "rms": 0.002, "inliers": 1000,
        },
    }))
    intrinsic_json.write_text(json.dumps({
        "111": {"fl_x": 1.0, "fl_y": 1.0, "cx": 1.0, "cy": 1.0, "w": 2, "h": 2},  # stale
        "unused_cam": {"fl_x": 5.0, "fl_y": 5.0, "cx": 2.0, "cy": 2.0, "w": 4, "h": 4},
    }))
    return extrinsic_json, intrinsic_json


def _record_with_calibration(root, extrinsic_json, intrinsic_json):
    recorder = LeRobotDatasetRecorder(
        REPO_ID, root=root, fps=10, task="push", extrinsic_json=extrinsic_json, intrinsic_json=intrinsic_json
    )
    _record(recorder, np.random.default_rng(0), 2, keep=True, log=[])
    recorder.finalize()


def test_sidecar_keeps_the_calibration_files(tmp_path, calibration_files):
    extrinsic_json, intrinsic_json = calibration_files
    root = tmp_path / "ds"

    _record_with_calibration(root, extrinsic_json, intrinsic_json)

    sidecar = read_sidecar(root)
    assert sidecar["extrinsic_calibration"] == json.loads(extrinsic_json.read_text())
    assert table_plane_from_extrinsics(sidecar["extrinsic_calibration"]).rms == pytest.approx(0.002)
    intrinsics = sidecar["intrinsic_calibration"]
    assert intrinsics["111"] == {"fl_x": 60.0, "fl_y": 61.0, "cx": 32.0, "cy": 24.0, "w": W, "h": H}
    assert intrinsics["222"]["fl_x"] == 61.0  # not in the file: live values
    assert intrinsics["unused_cam"]["fl_x"] == 5.0


def test_reopening_with_same_calibration_appends(tmp_path, calibration_files):
    extrinsic_json, intrinsic_json = calibration_files
    root = tmp_path / "ds"
    _record_with_calibration(root, extrinsic_json, intrinsic_json)

    _record_with_calibration(root, extrinsic_json, intrinsic_json)

    assert LeRobotDataset(REPO_ID, root=root).meta.total_episodes == 2


def test_reopening_after_table_plane_refit_raises(tmp_path, calibration_files):
    extrinsic_json, intrinsic_json = calibration_files
    root = tmp_path / "ds"
    _record_with_calibration(root, extrinsic_json, intrinsic_json)
    extrinsics = json.loads(extrinsic_json.read_text())
    extrinsics["table_plane"]["offset"] = 0.2
    extrinsic_json.write_text(json.dumps(extrinsics))

    with pytest.raises(ValueError, match="has changed"):
        LeRobotDatasetRecorder(REPO_ID, root=root, fps=10, extrinsic_json=extrinsic_json)


def test_reopening_appends(recorded):
    root, kept = recorded
    rng = np.random.default_rng(1)
    extra = []

    recorder = LeRobotDatasetRecorder(REPO_ID, root=root, fps=10, task="push")
    assert recorder.num_saved_episodes == 2
    _record(recorder, rng, 2, keep=True, log=extra)
    recorder.finalize()

    ds = LeRobotDataset(REPO_ID, root=root)
    assert ds.meta.total_episodes == 3
    assert len(ds) == len(kept) + 2
    last = frame_to_datapoints(ds[len(ds) - 1], extrinsics=read_fixed_extrinsics(root))
    assert np.array_equal(last[0].depth, extra[-1][0][0].depth)


def test_reopening_with_different_fps_raises(recorded):
    root, _ = recorded

    with pytest.raises(ValueError, match="fps"):
        LeRobotDatasetRecorder(REPO_ID, root=root, fps=30)


def test_mismatched_frame_shape_raises(recorded):
    root, _ = recorded
    recorder = LeRobotDatasetRecorder(REPO_ID, root=root, fps=10)
    datapoints, snapshot, action = _step_data(np.random.default_rng(2), 0)
    snapshot.link_pcds["base"] = np.zeros((7, 3))

    recorder.start_episode()
    with pytest.raises(ValueError, match="doesn't match"):
        recorder.add_frame(datapoints, snapshot, action)
    recorder.finalize()
