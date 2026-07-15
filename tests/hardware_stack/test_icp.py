"""icp.py imports open3d/pyrealsense2/viser at module scope (directly and via
camera_stream.py/alignment_viewer.py), so reaching even its pure filesystem logic
requires all three installed. No physical hardware needed.
"""
import pytest

pytest.importorskip("open3d")
pytest.importorskip("pyrealsense2")
pytest.importorskip("viser")

from lerobot_3d.icp import discover_calibration_serials

pytestmark = pytest.mark.hardware_stack


def test_discover_calibration_serials_finds_complete_dirs(tmp_path):
    complete = tmp_path / "111111"
    complete.mkdir()
    (complete / "depth.npz").write_bytes(b"")
    (complete / "mask.png").write_bytes(b"")

    incomplete = tmp_path / "222222"
    incomplete.mkdir()
    (incomplete / "depth.npz").write_bytes(b"")
    # no mask.png

    (tmp_path / "not_a_dir.txt").write_bytes(b"")

    result = discover_calibration_serials(str(tmp_path))

    assert result == ["111111"]


def test_discover_calibration_serials_empty_dir(tmp_path):
    assert discover_calibration_serials(str(tmp_path)) == []


def test_discover_calibration_serials_sorted(tmp_path):
    for name in ["b_serial", "a_serial"]:
        d = tmp_path / name
        d.mkdir()
        (d / "depth.npz").write_bytes(b"")
        (d / "mask.png").write_bytes(b"")

    assert discover_calibration_serials(str(tmp_path)) == ["a_serial", "b_serial"]
