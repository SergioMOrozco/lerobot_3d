"""``lerobot-trim``: range parsing/validation, and a trimmed copy that keeps exactly the chosen
frames (depth bit-exact, numbers equal, video close) while leaving the source untouched."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot_3d.recording.trim import parse_ranges, resolve_ranges

pytestmark = pytest.mark.hardware_stack


def test_parse_ranges():
    assert parse_ranges(["0:2:5", "3:0:0"], [1]) == {0: (2, 5), 3: (0, 0), 1: None}
    assert parse_ranges(None, None) == {}
    for bad in (["0:2"], ["a:1:2"], ["0:1:2:3"]):
        with pytest.raises(ValueError, match="EPISODE:FIRST:LAST"):
            parse_ranges(bad, None)
    with pytest.raises(ValueError, match="more than once"):
        parse_ranges(["0:1:2", "0:3:4"], None)
    with pytest.raises(ValueError, match="both kept and dropped"):
        parse_ranges(["0:1:2"], [0])


def test_resolve_ranges_fills_whole_episodes_and_validates():
    lengths = {0: 10, 1: 5}
    assert resolve_ranges(lengths, {0: (2, 4)}) == {0: (2, 4), 1: (0, 4)}
    assert resolve_ranges(lengths, {1: None}) == {0: (0, 9), 1: None}
    with pytest.raises(ValueError, match="not in the dataset"):
        resolve_ranges(lengths, {7: (0, 1)})
    for span in ((3, 2), (-1, 2), (0, 10)):
        with pytest.raises(ValueError, match="first <= last"):
            resolve_ranges(lengths, {0: span})
    with pytest.raises(ValueError, match="nothing to write"):
        resolve_ranges(lengths, {0: None, 1: None})


pytest.importorskip("lerobot")
pytest.importorskip("cv2")

H, W = 48, 64
SERIALS = ["cam_a", "cam_b"]


def _record(root, lengths, rng):
    from lerobot_3d.common.types import Datapoint, RobotSnapshot
    from lerobot_3d.recording.dataset_recorder import LeRobotDatasetRecorder

    recorder = LeRobotDatasetRecorder(repo_id="local/trim_src", root=root, fps=10, task="push")
    for ep, n in enumerate(lengths):
        recorder.start_episode()
        for t in range(n):
            datapoints = []
            for i, serial in enumerate(SERIALS):
                X_WC = np.eye(4)
                X_WC[:3, 3] = [0.1 * i, 0.0, 1.0]
                datapoints.append(Datapoint(
                    serial=serial, color=np.full((H, W, 3), 20 * t + 10 * i, np.uint8),
                    depth=rng.integers(200, 3000, (H, W), dtype=np.uint16), depth_scale=0.001, max_depth=10.0,
                    X_WC=X_WC, color_intrinsics=SimpleNamespace(fx=60.0, fy=60.0, ppx=32.0, ppy=24.0),
                ))
            snapshot = RobotSnapshot(index=0, joint_positions={"j.pos": 100.0 * ep + t}, joint_radians={},
                                     pcd=np.zeros((0, 3)), link_pcds={"base": rng.random((10, 3))},
                                     link_poses={}, base_offset=np.zeros(3))
            recorder.add_frame(datapoints, snapshot, {"j.pos": 100.0 * ep + t + 0.5})
        recorder.stop_episode()
        recorder.save_episode()
    recorder.finalize()


def test_trim_writes_chosen_frames_only(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from lerobot_3d.recording import trim
    from lerobot_3d.recording.dataset_loader import frame_to_datapoints, read_fixed_extrinsics, read_sidecar

    source, target = tmp_path / "src", tmp_path / "dst"
    _record(source, [8, 4, 6], np.random.default_rng(0))
    before = sorted(p.relative_to(source) for p in source.rglob("*"))
    trim.main(["--repo-id", "local/trim_src", "--root", str(source), "--output-repo-id", "local/trim_dst",
               "--output-root", str(target), "--keep", "0:2:5", "--drop", "1"])

    src = LeRobotDataset("local/trim_src", root=source)
    dst = LeRobotDataset("local/trim_dst", root=target)
    assert dst.meta.total_episodes == 2 and len(dst) == 4 + 6
    assert list(dst.meta.episodes["length"]) == [4, 6]
    # New episode 0 = source episode 0 frames 2..5; new episode 1 = all of source episode 2.
    expected = [(0, f) for f in range(2, 6)] + [(2, f) for f in range(6)]
    src_start = dict(zip(src.meta.episodes["episode_index"], src.meta.episodes["dataset_from_index"]))
    for row, (ep, f) in enumerate(expected):
        a, b = src[src_start[ep] + f], dst[row]
        assert np.allclose(b["action"].numpy(), a["action"].numpy())
        assert np.allclose(b["observation.state"].numpy(), a["observation.state"].numpy())
        assert int(b["frame_index"]) == row - (0 if row < 4 else 4)
        before_dps = frame_to_datapoints(a, extrinsics=read_fixed_extrinsics(source))
        after_dps = frame_to_datapoints(b, extrinsics=read_fixed_extrinsics(target))
        for x, y in zip(before_dps, after_dps):
            assert np.array_equal(x.depth, y.depth)  # bit-exact
            assert np.allclose(x.X_WC, y.X_WC)
            assert np.abs(x.color.astype(int) - y.color.astype(int)).mean() < 3  # one more video encode
        assert b["task"] == "push"

    sidecar = read_sidecar(target)
    assert sidecar["camera_serials"] == SERIALS
    assert sidecar["trim"]["episodes"] == [
        {"episode_index": 0, "source_episode": 0, "source_first_frame": 2, "source_last_frame": 5},
        {"episode_index": 1, "source_episode": 2, "source_first_frame": 0, "source_last_frame": 5},
    ]
    assert not (target / "images").exists() and not (tmp_path / "dst.partial").exists()
    assert sorted(p.relative_to(source) for p in source.rglob("*")) == before  # source untouched
    assert json.loads((target / "meta" / "info.json").read_text())["total_frames"] == 10

    with pytest.raises(FileExistsError):
        trim.main(["--repo-id", "local/trim_src", "--root", str(source), "--output-repo-id", "x",
                   "--output-root", str(target), "--keep", "0:0:1"])
    with pytest.raises(ValueError, match="in place"):
        trim.main(["--repo-id", "local/trim_src", "--root", str(source), "--output-repo-id", "x",
                   "--output-root", str(source), "--keep", "0:0:1"])
