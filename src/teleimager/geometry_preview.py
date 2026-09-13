"""GR00T-compatible previews derived from aligned RealSense depth."""

import numpy as np


DEPTH_NEAR_M = 0.25
DEPTH_FAR_M = 1.0
NORMAL_MAX_DEPTH_DELTA_M = 0.05

# D435I 242322076480 colour intrinsics at 640x480. Aligned depth uses these
# colour-camera intrinsics. Keep synchronized with unitree_lerobot's encoders.
WIDTH = 640
HEIGHT = 480
FX = 605.421508789062
FY = 605.590515136719
CX = 321.856811523438
CY = 242.249740600586


def encode_depth_gray_rgb(depth_u16, *, scale_m_per_unit):
    """Match the fixed-metric grayscale depth supplied to GR00T."""

    depth = np.asarray(depth_u16)
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError(
            f"Expected an HxW uint16 depth image; got shape={depth.shape}, dtype={depth.dtype}"
        )
    scale = float(scale_m_per_unit)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"scale_m_per_unit must be positive and finite, got {scale}")

    depth_m = depth.astype(np.float32) * scale
    valid = depth != 0
    normalized = np.clip((depth_m - DEPTH_NEAR_M) / (DEPTH_FAR_M - DEPTH_NEAR_M), 0.0, 1.0)
    gray = np.zeros_like(depth, dtype=np.uint8)
    gray[valid] = 1 + np.round(254 * normalized[valid]).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


def encode_surface_normals_rgb(depth_u16, *, scale_m_per_unit):
    """Match the camera-XYZ surface-normal image supplied to GR00T."""

    depth = np.asarray(depth_u16)
    if depth.dtype != np.uint16 or depth.shape != (HEIGHT, WIDTH):
        raise ValueError(
            f"Expected a {HEIGHT}x{WIDTH} uint16 aligned-depth image; "
            f"got shape={depth.shape}, dtype={depth.dtype}"
        )
    scale = float(scale_m_per_unit)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"scale_m_per_unit must be positive and finite, got {scale}")

    depth_m = depth.astype(np.float32) * np.float32(scale)
    x_scale = (np.arange(WIDTH, dtype=np.float32) - np.float32(CX)) / np.float32(FX)
    y_scale = (np.arange(HEIGHT, dtype=np.float32) - np.float32(CY)) / np.float32(FY)
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
    encoded = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    encoded[1:-1, 1:-1][valid] = encoded_inner[valid]
    return np.ascontiguousarray(encoded)
