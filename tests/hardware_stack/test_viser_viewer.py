"""Needs viser installed (module-level `import viser`). No physical hardware or a
running viser server needed -- ``_as_viser_colors`` is pure dtype/clipping logic.
"""
import numpy as np
import pytest

pytest.importorskip("viser")

from lerobot_3d.point_clouds.viser_viewer import _as_viser_colors

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
