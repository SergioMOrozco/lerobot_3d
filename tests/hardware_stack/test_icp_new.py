"""icp_new.py does `import open3d` at module scope, so reaching even its pure math
functions requires open3d installed. No physical hardware needed.
"""
import numpy as np
import pytest

pytest.importorskip("open3d")

from lerobot_3d.icp_new import discover_calibration_serials, matrix_to_rotvec, params_to_transforms
from scipy.spatial.transform import Rotation as R

pytestmark = pytest.mark.hardware_stack


def test_params_to_transforms_identity():
    p = np.zeros(6)

    (T,) = params_to_transforms(p, n_cams=1)

    assert np.allclose(T, np.eye(4))


def test_params_to_transforms_translation_only():
    p = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0])

    (T,) = params_to_transforms(p, n_cams=1)

    assert np.allclose(T[:3, :3], np.eye(3))
    assert np.allclose(T[:3, 3], [1.0, 2.0, 3.0])


def test_params_to_transforms_multiple_cameras():
    p = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0])

    transforms = params_to_transforms(p, n_cams=2)

    assert len(transforms) == 2
    assert np.allclose(transforms[0][:3, 3], [1.0, 0.0, 0.0])
    assert np.allclose(transforms[1][:3, 3], [0.0, 2.0, 0.0])


def test_matrix_to_rotvec_identity():
    assert np.allclose(matrix_to_rotvec(np.eye(4)), [0.0, 0.0, 0.0])


def test_matrix_to_rotvec_round_trips_with_from_rotvec():
    rotvec = np.array([0.1, -0.2, 0.3])
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(rotvec).as_matrix()

    result = matrix_to_rotvec(T)

    assert np.allclose(result, rotvec)


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
