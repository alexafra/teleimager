import hashlib
import json

import numpy as np
import pytest

from teleimager.geometry_preview import (
    color_intrinsics_from_calibration,
    encode_depth_gray_rgb,
    encode_surface_normals_rgb,
)


COLOR_INTRINSICS = {
    "width": 640,
    "height": 480,
    "fx": 609.3858642578125,
    "fy": 609.4705200195312,
    "cx": 325.95001220703125,
    "cy": 247.26507568359375,
    "distortion": "distortion.inverse_brown_conrady",
    "coeffs": [0.0] * 5,
    "format": "bgr8",
    "fps": 30,
}


def _calibration():
    calibration = {
        "schema": "realsense_rgbd_calibration.v1",
        "camera": {
            "model": "Intel RealSense D435I",
            "serial": "254322071415",
            "product_id": "0B3A",
            "firmware": "5.15.1.55",
        },
        "color": {
            **COLOR_INTRINSICS,
            "coeffs": list(COLOR_INTRINSICS["coeffs"]),
        },
        "depth": {
            "width": 640,
            "height": 480,
            "fx": 397.5912170410156,
            "fy": 397.5912170410156,
            "cx": 315.6465148925781,
            "cy": 244.2028350830078,
            "distortion": "distortion.brown_conrady",
            "coeffs": [0.0] * 5,
            "format": "z16",
            "fps": 30,
        },
        "depth_to_color": {
            "rotation": [
                0.9999486207962036,
                0.0033009203616529703,
                0.009585974738001823,
                -0.0033925294410437346,
                0.9999485611915588,
                0.009556086733937263,
                -0.009553938172757626,
                -0.009588115848600864,
                0.9999083876609802,
            ],
            "translation_m": [
                0.014800711534917355,
                0.0008831368759274483,
                0.0007359444862231612,
            ],
        },
    }
    canonical_json = json.dumps(
        calibration,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    calibration["fingerprint"] = (
        f"sha256:{hashlib.sha256(canonical_json).hexdigest()}"
    )
    return calibration


def test_live_calibration_supplies_replacement_camera_color_intrinsics():
    extracted = color_intrinsics_from_calibration(_calibration())

    assert extracted == {
        "width": 640,
        "height": 480,
        "fx": 609.3858642578125,
        "fy": 609.4705200195312,
        "cx": 325.95001220703125,
        "cy": 247.26507568359375,
    }


def test_live_calibration_rejects_tampered_intrinsics():
    calibration = _calibration()
    calibration["color"]["fx"] += 1.0

    with pytest.raises(ValueError, match="fingerprint"):
        color_intrinsics_from_calibration(calibration)


def test_depth_gray_fixed_metric_mapping_and_invalid_sentinel():
    depth = np.array([[0, 250, 625, 1000, 1250]], dtype=np.uint16)
    encoded = encode_depth_gray_rgb(
        depth,
        scale_m_per_unit=0.0010000000474974513,
    )

    assert encoded.dtype == np.uint8
    assert encoded.shape == (1, 5, 3)
    assert encoded[:, :, 0].tolist() == [[0, 1, 128, 255, 255]]
    np.testing.assert_array_equal(encoded[:, :, 0], encoded[:, :, 1])
    np.testing.assert_array_equal(encoded[:, :, 1], encoded[:, :, 2])


def test_flat_plane_surface_normal_encoding():
    depth = np.full((480, 640), 600, dtype=np.uint16)
    encoded = encode_surface_normals_rgb(
        depth,
        scale_m_per_unit=0.0010000000474974513,
        color_intrinsics=COLOR_INTRINSICS,
    )

    assert encoded.dtype == np.uint8
    assert encoded.flags.c_contiguous
    expected = np.broadcast_to([128, 128, 1], encoded[1:-1, 1:-1].shape)
    np.testing.assert_array_equal(encoded[1:-1, 1:-1], expected)
    assert not encoded[0].any()
    assert not encoded[-1].any()
    assert not encoded[:, 0].any()
    assert not encoded[:, -1].any()


def test_surface_normal_invalid_depth_uses_zero_rgb_sentinel():
    depth = np.full((480, 640), 600, dtype=np.uint16)
    depth[240, 320] = 0

    encoded = encode_surface_normals_rgb(
        depth,
        scale_m_per_unit=0.001,
        color_intrinsics=COLOR_INTRINSICS,
    )

    assert encoded[240, 320].tolist() == [0, 0, 0]


@pytest.mark.parametrize("scale", [0.0005, 0.0011, float("nan")])
def test_geometry_preview_rejects_noncanonical_scale(scale):
    depth = np.full((3, 3), 600, dtype=np.uint16)
    with pytest.raises(ValueError, match="canonical|finite"):
        encode_depth_gray_rgb(depth, scale_m_per_unit=scale)
