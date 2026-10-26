"""table_plane: RANSAC fit, table frame/extent, JSON round trip, masked back-projection."""
import json

import numpy as np
import pytest

pytest.importorskip("cv2")
from scipy.spatial.transform import Rotation

from lerobot_3d.table_plane import (
    TABLE_PLANE_KEY,
    TablePlane,
    fit_table_plane,
    load_table_plane,
    masked_world_points,
    table_plane_from_extrinsics,
)

pytestmark = pytest.mark.hardware_stack


def _table_scene(rng, X_W_table, size=(1.2, 0.6), noise=0.001, n=20000):
    """Noisy table-top points in the world, plus a box on the table and stray outliers."""
    top = np.column_stack([
        rng.uniform(-size[0] / 2, size[0] / 2, n),
        rng.uniform(-size[1] / 2, size[1] / 2, n),
        rng.normal(0.0, noise, n),
    ])
    box = np.column_stack([rng.uniform(0.1, 0.2, 3000), rng.uniform(0.0, 0.1, 3000), rng.uniform(0.0, 0.1, 3000)])
    outliers = rng.uniform(-1.0, 1.0, (1500, 3))
    local = np.concatenate([top, box, outliers])
    return local @ X_W_table[:3, :3].T + X_W_table[:3, 3]


def _pose(rotvec, translation):
    X = np.eye(4)
    X[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    X[:3, 3] = translation
    return X


@pytest.mark.parametrize("rotvec", [(0.0, 0.0, 0.3), (0.05, -0.03, 1.2), (0.0, 0.0, 0.0)])
def test_fit_recovers_tilted_table(rotvec):
    X_W_table = _pose(rotvec, (0.3, -0.1, 0.75))
    points = _table_scene(np.random.default_rng(0), X_W_table)

    plane = fit_table_plane(points)

    assert np.degrees(np.arccos(np.clip(plane.normal @ X_W_table[:3, 2], -1, 1))) < 0.2
    assert abs(plane.signed_distance(X_W_table[:3, 3][None])[0]) < 0.001
    assert np.allclose(plane.X_W_table[:3, 3], X_W_table[:3, 3], atol=0.01)
    assert np.allclose(plane.extent, (1.2, 0.6), atol=0.01)
    # +x along the long edge (either direction), frame right-handed with z = normal.
    assert abs(plane.X_W_table[:3, 0] @ X_W_table[:3, 0]) > 0.999
    assert np.allclose(plane.X_W_table[:3, :3] @ plane.X_W_table[:3, :3].T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(plane.X_W_table[:3, :3]), 1.0)
    assert np.allclose(plane.X_W_table[:3, 2], plane.normal)
    assert plane.rms < 0.002
    # Corners sit on the plane, at the table's corners.
    assert np.allclose(plane.signed_distance(plane.corners()), 0.0, atol=1e-9)
    corners_local = (plane.corners() - X_W_table[:3, 3]) @ X_W_table[:3, :3]
    assert np.allclose(np.sort(np.abs(corners_local[:, :2]), axis=0), [[0.6, 0.3]] * 4, atol=0.01)


def test_normal_points_up_and_heights_are_positive_above():
    X_W_table = _pose((np.pi, 0.0, 0.0), (0.0, 0.0, 0.5))  # table frame upside down in the world
    plane = fit_table_plane(_table_scene(np.random.default_rng(1), X_W_table))

    assert plane.normal[2] > 0.99
    assert plane.signed_distance(np.array([[0.0, 0.0, 0.6]]))[0] == pytest.approx(0.1, abs=0.002)


def test_too_few_points_raises():
    with pytest.raises(ValueError, match="at least 3"):
        fit_table_plane(np.zeros((2, 3)))


def test_json_round_trip_through_extrinsics_file(tmp_path, monkeypatch):
    plane = fit_table_plane(_table_scene(np.random.default_rng(2), _pose((0.0, 0.0, 0.4), (0.1, 0.2, 0.0))))
    path = tmp_path / "extrinsic_calibration.json"
    path.write_text(json.dumps({"cam": {"X_WC": np.eye(4).tolist()}, TABLE_PLANE_KEY: plane.to_json()}))
    monkeypatch.chdir(tmp_path)

    loaded = load_table_plane(path.name)

    assert isinstance(loaded, TablePlane)
    for field in ("normal", "X_W_table", "extent"):
        assert np.allclose(getattr(loaded, field), getattr(plane, field))
    assert (loaded.offset, loaded.rms, loaded.inliers) == (plane.offset, plane.rms, plane.inliers)
    assert table_plane_from_extrinsics({}) is None


def test_masked_world_points_keeps_eroded_mask_in_world_frame():
    depth = np.full((40, 60), 1000, dtype=np.uint16)  # 1 m everywhere
    mask = np.zeros((40, 60), dtype=bool)
    mask[10:30, 20:50] = True
    K = np.array([[50.0, 0.0, 30.0], [0.0, 50.0, 20.0], [0.0, 0.0, 1.0]])
    X_WC = _pose((np.pi, 0.0, 0.0), (0.0, 0.0, 2.0))  # looking straight down from 2 m

    points = masked_world_points(depth, mask, K, 0.001, X_WC, erode_px=2, stride=1)

    assert len(points) == (20 - 4) * (30 - 4)
    assert np.allclose(points[:, 2], 1.0)
    u = points[:, 0] * 50.0 + 30.0  # camera x = world x under this pose
    assert u.min() == pytest.approx(22) and u.max() == pytest.approx(47)
