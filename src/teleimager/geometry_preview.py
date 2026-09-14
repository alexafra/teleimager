"""GR00T-compatible previews derived from aligned RealSense depth."""

import hashlib
import json

import numpy as np


DEPTH_NEAR_M = 0.25
DEPTH_FAR_M = 1.0
NORMAL_MAX_DEPTH_DELTA_M = 0.05
CANONICAL_DEPTH_SCALE_M_PER_UNIT = 0.001
REALSENSE_CALIBRATION_SCHEMA = "realsense_rgbd_calibration.v1"


def _canonical_depth_scale(scale_m_per_unit):
    try:
        scale = float(scale_m_per_unit)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(
            "scale_m_per_unit must be a finite number"
        ) from error
    if not np.isfinite(scale):
        raise ValueError("scale_m_per_unit must be a finite number")
    if np.float32(scale) != np.float32(CANONICAL_DEPTH_SCALE_M_PER_UNIT):
        raise ValueError(
            "scale_m_per_unit must equal the canonical "
            f"{CANONICAL_DEPTH_SCALE_M_PER_UNIT} m/unit at float32 precision; "
            f"got {scale!r}"
        )
    return CANONICAL_DEPTH_SCALE_M_PER_UNIT


def _validated_color_intrinsics(color):
    if not isinstance(color, dict):
        raise ValueError("Camera calibration is missing color intrinsics")

    required = ("width", "height", "fx", "fy", "cx", "cy")
    missing = [key for key in required if key not in color]
    if missing:
        raise ValueError(
            "Camera color intrinsics are missing " + ", ".join(missing)
        )
    if (
        isinstance(color["width"], bool)
        or not isinstance(color["width"], (int, np.integer))
        or isinstance(color["height"], bool)
        or not isinstance(color["height"], (int, np.integer))
    ):
        raise ValueError("Camera color width and height must be integers")

    width = int(color["width"])
    height = int(color["height"])
    try:
        fx, fy, cx, cy = (
            float(color[key]) for key in ("fx", "fy", "cx", "cy")
        )
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("Camera color intrinsics must be finite numbers") from error
    if not np.isfinite((fx, fy, cx, cy)).all():
        raise ValueError("Camera color intrinsics must be finite numbers")
    if width < 3 or height < 3 or fx <= 0.0 or fy <= 0.0:
        raise ValueError("Camera color intrinsics have invalid dimensions or focal lengths")
    if not 0.0 <= cx < width or not 0.0 <= cy < height:
        raise ValueError("Camera color principal point lies outside the image")
    return {
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
    }


def color_intrinsics_from_calibration(calibration):
    """Verify live calibration provenance and return its color pinhole model."""

    expected_keys = {
        "schema",
        "camera",
        "color",
        "depth",
        "depth_to_color",
        "fingerprint",
    }
    if not isinstance(calibration, dict) or set(calibration) != expected_keys:
        raise ValueError(
            "Camera config is missing complete depth calibration metadata"
        )
    if calibration["schema"] != REALSENSE_CALIBRATION_SCHEMA:
        raise ValueError(
            "Camera calibration schema must be "
            f"{REALSENSE_CALIBRATION_SCHEMA!r}"
        )
    fingerprint_payload = {
        key: value
        for key, value in calibration.items()
        if key != "fingerprint"
    }
    try:
        canonical_json = json.dumps(
            fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Camera calibration contains non-canonical JSON values"
        ) from error
    expected_fingerprint = (
        f"sha256:{hashlib.sha256(canonical_json).hexdigest()}"
    )
    if calibration["fingerprint"] != expected_fingerprint:
        raise ValueError(
            "Camera calibration fingerprint does not match its payload"
        )
    return _validated_color_intrinsics(calibration["color"])


def encode_depth_gray_rgb(depth_u16, *, scale_m_per_unit):
    """Match the fixed-metric grayscale depth supplied to GR00T."""

    depth = np.asarray(depth_u16)
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError(
            f"Expected an HxW uint16 depth image; got shape={depth.shape}, dtype={depth.dtype}"
        )
    scale = _canonical_depth_scale(scale_m_per_unit)

    depth_m = depth.astype(np.float32) * scale
    valid = depth != 0
    normalized = np.clip((depth_m - DEPTH_NEAR_M) / (DEPTH_FAR_M - DEPTH_NEAR_M), 0.0, 1.0)
    gray = np.zeros_like(depth, dtype=np.uint8)
    gray[valid] = 1 + np.round(254 * normalized[valid]).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


def encode_surface_normals_rgb(
    depth_u16,
    *,
    scale_m_per_unit,
    color_intrinsics,
):
    """Match the camera-XYZ surface-normal image supplied to GR00T."""

    depth = np.asarray(depth_u16)
    camera = _validated_color_intrinsics(color_intrinsics)
    expected_shape = (camera["height"], camera["width"])
    if depth.dtype != np.uint16 or depth.shape != expected_shape:
        raise ValueError(
            f"Expected a {camera['height']}x{camera['width']} uint16 "
            "aligned-depth image; "
            f"got shape={depth.shape}, dtype={depth.dtype}"
        )
    scale = _canonical_depth_scale(scale_m_per_unit)

    depth_m = depth.astype(np.float32) * np.float32(scale)
    x_scale = (
        np.arange(camera["width"], dtype=np.float32)
        - np.float32(camera["cx"])
    ) / np.float32(camera["fx"])
    y_scale = (
        np.arange(camera["height"], dtype=np.float32)
        - np.float32(camera["cy"])
    ) / np.float32(camera["fy"])
    point_x = depth_m * x_scale[None, :]
    point_y = depth_m * y_scale[:, None]

    center_depth = depth_m[1:-1, 1:-1]
    left_depth = depth_m[1:-1, :-2]
    right_depth = depth_m[1:-1, 2:]
    up_depth = depth_m[:-2, 1:-1]
    down_depth = depth_m[2:, 1:-1]

    tangent_x_x = point_x[1:-1, 2:] - point_x[1:-1, :-2]
    tangent_x_y = point_y[1:-1, 2:] - point_y[1:-1, :-2]
    tangent_x_z = right_depth - left_depth
    tangent_y_x = point_x[2:, 1:-1] - point_x[:-2, 1:-1]
    tangent_y_y = point_y[2:, 1:-1] - point_y[:-2, 1:-1]
    tangent_y_z = down_depth - up_depth

    normal_x = tangent_x_y * tangent_y_z - tangent_x_z * tangent_y_y
    normal_y = tangent_x_z * tangent_y_x - tangent_x_x * tangent_y_z
    normal_z = tangent_x_x * tangent_y_y - tangent_x_y * tangent_y_x
    norm = np.sqrt(normal_x * normal_x + normal_y * normal_y + normal_z * normal_z)

    source_valid = (
        (depth[1:-1, 1:-1] != 0)
        & (depth[1:-1, :-2] != 0)
        & (depth[1:-1, 2:] != 0)
        & (depth[:-2, 1:-1] != 0)
        & (depth[2:, 1:-1] != 0)
    )
    locally_continuous = (
        (np.abs(left_depth - center_depth) <= NORMAL_MAX_DEPTH_DELTA_M)
        & (np.abs(right_depth - center_depth) <= NORMAL_MAX_DEPTH_DELTA_M)
        & (np.abs(up_depth - center_depth) <= NORMAL_MAX_DEPTH_DELTA_M)
        & (np.abs(down_depth - center_depth) <= NORMAL_MAX_DEPTH_DELTA_M)
    )
    valid = source_valid & locally_continuous & np.isfinite(norm) & (norm > np.float32(1e-12))

    center_x = point_x[1:-1, 1:-1]
    center_y = point_y[1:-1, 1:-1]
    dot_normal_point = normal_x * center_x + normal_y * center_y + normal_z * center_depth
    orientation_sign = np.where(dot_normal_point > 0, np.float32(-1.0), np.float32(1.0))
    safe_norm = np.where(valid, norm, np.float32(1.0))
    normals = np.stack(
        (
            orientation_sign * normal_x / safe_norm,
            orientation_sign * normal_y / safe_norm,
            orientation_sign * normal_z / safe_norm,
        ),
        axis=-1,
    )
    encoded_inner = 1 + np.rint(
        np.float32(127.0) * (np.clip(normals, -1.0, 1.0) + 1.0)
    ).astype(np.uint8)
    encoded = np.zeros(
        (camera["height"], camera["width"], 3),
        dtype=np.uint8,
    )
    encoded[1:-1, 1:-1][valid] = encoded_inner[valid]
    return np.ascontiguousarray(encoded)
