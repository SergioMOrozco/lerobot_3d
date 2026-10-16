"""Needs viser installed (module-level `import viser`). No physical hardware or a
running viser server needed -- ``_as_viser_colors`` is pure dtype/clipping logic.
"""
import numpy as np
import pytest

pytest.importorskip("viser")

from lerobot_3d.point_clouds.viser_viewer import _as_viser_colors, grid_offsets

pytestmark = pytest.mark.hardware_stack


def test_none_colors_fill_mid_gray():
    points = np.zeros((4, 3))

    colors = _as_viser_colors(points, None)

    assert colors.shape == (4, 3)
    assert colors.dtype == np.uint8
    assert (colors == 178).all()


def test_uint8_colors_pass_through_unchanged():
    points = np.zeros((2, 3))
    original = np.array([[0, 128, 255], [10, 20, 30]], dtype=np.uint8)

    colors = _as_viser_colors(points, original)

    assert colors.dtype == np.uint8
    assert np.array_equal(colors, original)


def test_float_colors_scaled_to_uint8():
    points = np.zeros((1, 3))
    colors = _as_viser_colors(points, np.array([[0.0, 0.5, 1.0]]))

    assert colors.dtype == np.uint8
    assert colors[0, 0] == 0
    assert colors[0, 2] == 255


def test_out_of_range_float_colors_are_clipped():
    points = np.zeros((1, 3))

    colors = _as_viser_colors(points, np.array([[-1.0, 2.0, 0.5]]))

    assert colors[0, 0] == 0
    assert colors[0, 1] == 255


def test_grid_offsets_single_robot_at_origin():
    offsets = grid_offsets(1, spacing=0.5)

    assert len(offsets) == 1
    assert np.array_equal(offsets[0], [0.0, 0.0, 0.0])


def test_grid_offsets_four_robots_form_2x2():
    offsets = grid_offsets(4, spacing=0.5)

    assert [tuple(o) for o in offsets] == [
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (0.0, 0.5, 0.0),
        (0.5, 0.5, 0.0),
    ]


def test_grid_offsets_five_robots_use_three_columns():
    offsets = grid_offsets(5, spacing=1.0)

    assert np.array_equal(offsets[0], [0.0, 0.0, 0.0])
    assert np.array_equal(offsets[2], [2.0, 0.0, 0.0])
    assert np.array_equal(offsets[3], [0.0, 1.0, 0.0])


# ---------------------------------------------------------------------------
# Cameras: live images in the scene, one "look through" button each
# ---------------------------------------------------------------------------


class _FakeCamera:
    position = look_at = up_direction = fov = None


class _FakeClient:
    def __init__(self):
        self.camera = _FakeCamera()

    def atomic(self):
        import contextlib

        return contextlib.nullcontext()


@pytest.fixture
def camera_viewer(monkeypatch):
    import socket

    from lerobot_3d.point_clouds.viser_viewer import ViserSceneViewer

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    viewer = ViserSceneViewer(port=port, controls=False)
    clients = {0: _FakeClient(), 1: _FakeClient()}
    monkeypatch.setattr(viewer.server, "get_clients", lambda: clients)
    yield viewer, clients
    viewer.close()


def _views(top_position=(0.1, 0.3, 0.9), image_value=200):
    from scipy.spatial.transform import Rotation

    from lerobot_3d.point_clouds.viser_viewer import CameraView

    top = np.eye(4)
    top[:3, :3] = Rotation.from_euler("x", 150, degrees=True).as_matrix()
    top[:3, 3] = top_position
    image = np.full((480, 640, 3), image_value, np.uint8)
    return {"camera_top": CameraView(top, 0.8, 640 / 480, image),
            "camera_left_wrist": CameraView(np.eye(4), 1.0, 640 / 480, image)}


def test_each_camera_is_drawn_with_its_live_image_and_gets_a_button(camera_viewer):
    viewer, _ = camera_viewer
    viewer.camera_image_period_s = 0.0
    views = _views()
    viewer.update_cameras(views)
    assert list(viewer._camera_buttons) == ["camera_top", "camera_left_wrist"]  # N cameras -> N buttons
    frustum = viewer._camera_frustums["camera_top"]
    np.testing.assert_allclose(frustum.position, views["camera_top"].X_WC[:3, 3])
    assert frustum.image.shape == (360, 480, 3)  # shrunk to camera_image_width
    assert frustum.fov == pytest.approx(0.8) and frustum.aspect == pytest.approx(640 / 480)

    viewer.update_cameras(_views(top_position=(0.5, 0.0, 1.0), image_value=10))  # it moves; new image
    np.testing.assert_allclose(frustum.position, [0.5, 0.0, 1.0])
    assert frustum.image.max() == 10

    viewer.update_cameras({"camera_top": _views()["camera_top"]})  # the wrist camera stops reporting
    assert not viewer._camera_frustums["camera_left_wrist"].visible
    assert viewer._camera_buttons["camera_left_wrist"].disabled and not viewer._camera_buttons["camera_top"].disabled


def test_images_are_rate_limited(camera_viewer):
    viewer, _ = camera_viewer
    viewer.camera_image_period_s = 60.0
    viewer.update_cameras(_views(image_value=200))
    viewer.update_cameras(_views(image_value=10))
    assert viewer._camera_frustums["camera_top"].image.max() == 200  # not re-sent yet


def test_look_through_moves_the_view_once_without_locking_it(camera_viewer):
    viewer, clients = camera_viewer
    views = _views()
    viewer.update_cameras(views)
    assert all(c.camera.position is None for c in clients.values())  # nothing moves until clicked

    viewer.look_through("camera_top", [clients[0]])  # what camera_top's button does for the clicking browser
    top = views["camera_top"].X_WC
    camera = clients[0].camera
    np.testing.assert_allclose(camera.position, top[:3, 3])
    np.testing.assert_allclose(camera.look_at - top[:3, 3], 0.5 * top[:3, 2])  # +z forward
    np.testing.assert_allclose(camera.up_direction, -top[:3, 1])  # -y up
    assert camera.fov == 0.8
    assert clients[1].camera.position is None  # other browsers keep their own view

    # Not locked: later updates (even with the camera moving) leave the view where the user put it.
    camera.position = np.array([2.0, 2.0, 2.0])
    viewer.update_cameras(_views(top_position=(0.5, 0.0, 1.0)))
    np.testing.assert_allclose(camera.position, [2.0, 2.0, 2.0])
