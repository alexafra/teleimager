#!/usr/bin/env python3
"""
Minimal head-camera publisher for the Unitree G1 D435I, standing in for
teleimager's image_server.py.

Why this exists: image_server.py unconditionally imports aiohttp/aiortc/av
at module level (needed for its WebRTC signalling path), which is a heavy
compiled dependency chain to get onto an offline aarch64 robot. This script
only needs what image_client.py already provides (ZMQ_PublisherManager,
ZMQ_Responser) — both installed already via `pip install -e . --no-build-isolation`
in ~/teleimager — plus pyrealsense2, which is already present at
/usr/lib/python3/dist-packages on this robot.

Wire-compatible with the stock `teleimager-client`: publishes JPEG-encoded
frames on the head_camera zmq_port from cam_config_server.yaml (5555), and
serves a colour-only copy of that config over a ZMQ_Responser on port 60000 so
the client's config auto-discovery (ImageClient.__init__ ->
ZMQ_Requester.request()) works without any client-side changes.

Usage (on the robot):
    python3 realsense_zmq_publisher.py
Ctrl+C to stop.
"""

import copy
import time

import cv2
import numpy as np
import yaml

from teleimager.image_client import ZMQ_PublisherManager, ZMQ_Responser

CONFIG_PATH = "/home/unitree/teleimager/cam_config_server.yaml"
JPEG_QUALITY = 80


def as_color_only_config(cam_config):
    """Return a config that advertises only streams this fallback publishes."""

    cam_config = copy.deepcopy(cam_config)
    head_cfg = cam_config["head_camera"]
    head_cfg["enable_depth"] = False
    for unsupported_key in (
        "depth_zmq_port",
        "raw_depth_zmq_port",
        "rgbd_zmq_port",
        "rgbd_protocol",
        "depth_scale_m_per_unit",
        "depth_scale_reported_m_per_unit",
        "calibration",
    ):
        head_cfg.pop(unsupported_key, None)
    return cam_config


def main():
    import pyrealsense2 as rs

    with open(CONFIG_PATH) as f:
        cam_config = as_color_only_config(yaml.safe_load(f))

    head_cfg = cam_config["head_camera"]
    height, width = head_cfg["image_shape"]  # yaml stores [rows, cols] i.e. [H, W]
    fps = head_cfg["fps"]
    serial = head_cfg["serial_number"]
    zmq_port = head_cfg["zmq_port"]

    # Serve cam_config over REQ/REP on 60000 so ImageClient can auto-discover
    # the port instead of needing it hardcoded on the client side.
    responser = ZMQ_Responser(cam_config, host="0.0.0.0", port=60000)
    publisher = ZMQ_PublisherManager.get_instance()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    pipeline.start(config)

    print(f"[realsense_zmq_publisher] Publishing {width}x{height}@{fps} "
          f"on port {zmq_port}, serving config on 60000. Ctrl+C to stop.")

    frame_count = 0
    t0 = time.monotonic()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            img = np.asanyarray(color_frame.get_data())
            ok, jpg = cv2.imencode(
                ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )
            if not ok:
                continue

            publisher.publish(jpg.tobytes(), port=zmq_port, host="0.0.0.0")

            frame_count += 1
            if frame_count % 60 == 0:
                dt = time.monotonic() - t0
                print(f"[realsense_zmq_publisher] ~{frame_count / dt:.1f} fps")
                frame_count = 0
                t0 = time.monotonic()

    except KeyboardInterrupt:
        print("\n[realsense_zmq_publisher] Stopping.")
    finally:
        pipeline.stop()
        responser.stop()
        publisher.close()


if __name__ == "__main__":
    main()
