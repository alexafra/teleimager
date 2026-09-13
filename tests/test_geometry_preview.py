import numpy as np

from teleimager.geometry_preview import encode_depth_gray_rgb, encode_surface_normals_rgb


def test_depth_gray_fixed_metric_mapping_and_invalid_sentinel():
    depth = np.array([[0, 250, 625, 1000, 1250]], dtype=np.uint16)
    encoded = encode_depth_gray_rgb(depth, scale_m_per_unit=0.001)

    assert encoded.dtype == np.uint8
    assert encoded.shape == (1, 5, 3)
    assert encoded[:, :, 0].tolist() == [[0, 1, 128, 255, 255]]
    np.testing.assert_array_equal(encoded[:, :, 0], encoded[:, :, 1])
    np.testing.assert_array_equal(encoded[:, :, 1], encoded[:, :, 2])


def test_flat_plane_surface_normal_encoding():
    depth = np.full((480, 640), 600, dtype=np.uint16)
    encoded = encode_surface_normals_rgb(depth, scale_m_per_unit=0.001)

    assert encoded.dtype == np.uint8
    assert encoded.flags.c_contiguous
    expected = np.broadcast_to([128, 128, 1], encoded[1:-1, 1:-1].shape)
    np.testing.assert_array_equal(encoded[1:-1, 1:-1], expected)
    assert not encoded[0].any()
    assert not encoded[-1].any()
    assert not encoded[:, 0].any()
    assert not encoded[:, -1].any()
