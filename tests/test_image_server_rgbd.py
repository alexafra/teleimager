import sys
import threading
import types

import cv2
import logging_mp
import numpy as np
import pytest


# image_server imports its optional WebRTC stack at module import time. These
# tests exercise only RealSense/ZMQ behavior, so provide the smallest interfaces
# needed on machines without the server extras installed.
aiortc = types.ModuleType("aiortc")
aiortc_rtp = types.ModuleType("aiortc.rtcrtpsender")
aiortc_contrib = types.ModuleType("aiortc.contrib")
aiortc_media = types.ModuleType("aiortc.contrib.media")
aiortc_codecs = types.ModuleType("aiortc.codecs")
aiortc_h264 = types.ModuleType("aiortc.codecs.h264")
av = types.ModuleType("av")


class _DummyMediaStreamTrack:
    kind = "video"


class _DummyH264Encoder:
    pass


class _DummyVideoFrame:
    pass


class _DummyRtpSender:
    @staticmethod
    def getCapabilities(_kind):
        return types.SimpleNamespace(codecs=[])


aiortc.RTCPeerConnection = object
aiortc.RTCSessionDescription = object
aiortc.MediaStreamTrack = _DummyMediaStreamTrack
aiortc_rtp.RTCRtpSender = _DummyRtpSender
aiortc_media.MediaRelay = object
aiortc_h264.H264Encoder = _DummyH264Encoder
aiortc_codecs.h264 = aiortc_h264
av.VideoFrame = _DummyVideoFrame
sys.modules.setdefault("aiortc", aiortc)
sys.modules.setdefault("aiortc.rtcrtpsender", aiortc_rtp)
sys.modules.setdefault("aiortc.contrib", aiortc_contrib)
sys.modules.setdefault("aiortc.contrib.media", aiortc_media)
sys.modules.setdefault("aiortc.codecs", aiortc_codecs)
sys.modules.setdefault("aiortc.codecs.h264", aiortc_h264)
sys.modules.setdefault("av", av)

logging_mp.basic_config = lambda **_kwargs: None
if not hasattr(logging_mp, "get_logger"):
    logging_mp.get_logger = logging_mp.getLogger

from teleimager.image_client import TripleRingBuffer  # noqa: E402
from teleimager import image_server as image_server_module  # noqa: E402
from teleimager.image_server import (  # noqa: E402
    ImageServer,
    RealSenseCamera,
    _configured_transport_ports,
    _validated_rgbd_zmq_port,
)
from teleimager.rgbd_protocol import RGBD_PROTOCOL, unpack_rgbd_packet  # noqa: E402


class _Frame:
    def __init__(self, data):
        self._data = data

    def get_data(self):
        return self._data

    def __bool__(self):
        return self._data is not None


class _Frames:
    def __init__(self, color, aligned_depth, raw_depth=None):
        self.color = _Frame(color)
        self.aligned_depth = _Frame(aligned_depth)
        self.raw_depth = _Frame(raw_depth)

    def get_depth_frame(self):
        return self.raw_depth


class _AlignedFrames:
    def __init__(self, frames):
        self._frames = frames

    def get_color_frame(self):
        return self._frames.color

    def get_depth_frame(self):
        return self._frames.aligned_depth


class _Pipeline:
    def __init__(self, frames):
        self._frames = frames

    def wait_for_frames(self):
        return self._frames


class _Align:
    def process(self, frames):
        return _AlignedFrames(frames)


def _fake_camera(color, aligned_depth, *, legacy_buffers=False):
    camera = object.__new__(RealSenseCamera)
    camera.pipeline = _Pipeline(_Frames(color, aligned_depth))
    camera.align = _Align()
    camera._capture_sequence = 0
    camera._enable_depth = True
    camera._latest_depth = None
    camera._latest_raw_depth = None
    camera._depth_zmq_buffer = TripleRingBuffer() if legacy_buffers else None
    camera._raw_depth_zmq_buffer = None
    camera._rgbd_zmq_buffer = TripleRingBuffer()
    camera._enable_webrtc = False
    camera._webrtc_buffer = None
    camera._enable_zmq = legacy_buffers
    camera._zmq_buffer = TripleRingBuffer() if legacy_buffers else None
    camera._ready = threading.Event()
    camera._cam_topic = "head_camera"
    return camera


def test_update_frame_emits_one_same_frameset_packet_and_reuses_encodes():
    color = np.zeros((4, 5, 3), dtype=np.uint8)
    color[..., 2] = 200
    aligned_depth = np.arange(20, dtype=np.uint16).reshape(4, 5) * 23
    camera = _fake_camera(color, aligned_depth, legacy_buffers=True)

    camera._update_frame()

    sequence, packet = camera.get_rgbd_packet()
    frame = unpack_rgbd_packet(packet)
    decoded_color = cv2.imdecode(
        np.frombuffer(frame.color_jpeg, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    decoded_depth = cv2.imdecode(
        np.frombuffer(frame.aligned_depth_png, dtype=np.uint8),
        cv2.IMREAD_UNCHANGED,
    )
    assert sequence == 1
    assert frame.sequence == sequence
    assert frame.server_capture_monotonic_ns > 0
    assert decoded_color.shape == color.shape
    np.testing.assert_array_equal(decoded_depth, aligned_depth)
    assert frame.color_jpeg == camera._zmq_buffer.read()
    assert frame.aligned_depth_png == camera._depth_zmq_buffer.read()


@pytest.mark.parametrize(
    ("color", "aligned_depth"),
    [
        (np.zeros((4, 5, 3), dtype=np.uint8), None),
        (None, np.zeros((4, 5), dtype=np.uint16)),
    ],
)
def test_update_frame_does_not_emit_incomplete_pair(color, aligned_depth):
    camera = _fake_camera(color, aligned_depth)

    camera._update_frame()

    assert camera.get_rgbd_packet() is None


def test_packet_construction_failure_disables_only_atomic_path(monkeypatch):
    color = np.zeros((4, 5, 3), dtype=np.uint8)
    aligned_depth = np.arange(20, dtype=np.uint16).reshape(4, 5)
    camera = _fake_camera(color, aligned_depth, legacy_buffers=True)

    def _fail_pack(*_args, **_kwargs):
        raise RuntimeError("pack failed")

    monkeypatch.setattr(image_server_module, "pack_rgbd_packet", _fail_pack)

    camera._update_frame()

    assert camera._rgbd_zmq_buffer is None
    assert camera._zmq_buffer.read() is not None
    assert camera._depth_zmq_buffer.read() is not None
    assert camera._ready.is_set()


def test_invalid_optional_rgbd_config_is_removed_without_touching_legacy_streams():
    config = {
        "enable_zmq": True,
        "zmq_port": 5555,
        "enable_depth": True,
        "depth_zmq_port": 5558,
        "rgbd_zmq_port": 5560,
        "rgbd_protocol": "teleimager-rgbd-v999",
    }

    assert _validated_rgbd_zmq_port("head_camera", config, "realsense") is None

    assert "rgbd_zmq_port" not in config
    assert "rgbd_protocol" not in config
    assert config["enable_zmq"] is True
    assert config["zmq_port"] == 5555
    assert config["depth_zmq_port"] == 5558


@pytest.mark.parametrize(
    ("config_update", "cam_type"),
    [
        ({"rgbd_zmq_port": 0, "rgbd_protocol": RGBD_PROTOCOL}, "realsense"),
        ({"rgbd_zmq_port": "5560", "rgbd_protocol": RGBD_PROTOCOL}, "realsense"),
        ({"rgbd_zmq_port": 5560, "rgbd_protocol": RGBD_PROTOCOL}, "uvc"),
        ({"rgbd_protocol": RGBD_PROTOCOL}, "realsense"),
    ],
)
def test_malformed_or_unsupported_rgbd_config_degrades_to_legacy(config_update, cam_type):
    config = {"enable_zmq": True, "enable_depth": True, **config_update}

    assert _validated_rgbd_zmq_port("head_camera", config, cam_type) is None
    assert "rgbd_zmq_port" not in config
    assert "rgbd_protocol" not in config
    assert config["enable_zmq"] is True


def test_valid_rgbd_config_is_kept():
    config = {
        "enable_zmq": True,
        "enable_depth": True,
        "rgbd_zmq_port": 5560,
        "rgbd_protocol": RGBD_PROTOCOL,
    }

    assert _validated_rgbd_zmq_port("head_camera", config, "realsense") == 5560
    assert config["rgbd_zmq_port"] == 5560
    assert config["rgbd_protocol"] == RGBD_PROTOCOL


@pytest.mark.parametrize(
    "colliding_key",
    ["zmq_port", "depth_zmq_port", "raw_depth_zmq_port", "webrtc_port"],
)
def test_rgbd_port_collision_with_same_camera_stream_degrades_to_legacy(colliding_key):
    config = {
        "enable_zmq": True,
        "enable_depth": True,
        colliding_key: 5560,
        "rgbd_zmq_port": 5560,
        "rgbd_protocol": RGBD_PROTOCOL,
    }

    assert _validated_rgbd_zmq_port("head_camera", config, "realsense") is None
    assert "rgbd_zmq_port" not in config
    assert "rgbd_protocol" not in config
    assert config[colliding_key] == 5560


def test_rgbd_port_collision_with_other_camera_or_responder_degrades_to_legacy():
    head_config = {
        "enable_zmq": True,
        "enable_depth": True,
        "rgbd_zmq_port": 5560,
        "rgbd_protocol": RGBD_PROTOCOL,
    }
    all_config = {
        "head_camera": head_config,
        "left_wrist_camera": {"enable_zmq": True, "zmq_port": 5560},
    }

    configured_ports = _configured_transport_ports(all_config)
    assert _validated_rgbd_zmq_port(
        "head_camera",
        head_config,
        "realsense",
        configured_ports=configured_ports,
    ) is None
    assert "rgbd_zmq_port" not in head_config
    assert all_config["left_wrist_camera"]["zmq_port"] == 5560

    responder_collision = {
        "enable_depth": True,
        "rgbd_zmq_port": 60000,
        "rgbd_protocol": RGBD_PROTOCOL,
    }
    assert _validated_rgbd_zmq_port(
        "head_camera",
        responder_collision,
        "realsense",
    ) is None


class _FailingAtomicCamera:
    def get_fps(self):
        return 30

    def get_rgbd_zmq_port(self):
        return 5560

    def get_rgbd_packet(self):
        raise RuntimeError("optional stream failed")


class _RepeatedThenFailingAtomicCamera(_FailingAtomicCamera):
    def __init__(self):
        self.calls = 0

    def get_rgbd_packet(self):
        self.calls += 1
        if self.calls <= 2:
            return 7, b"same-capture"
        raise RuntimeError("end test loop")


class _RecordingPublisher:
    def __init__(self):
        self.calls = []

    def publish(self, packet, port):
        self.calls.append((packet, port))


def test_atomic_publisher_failure_does_not_stop_legacy_server():
    server = object.__new__(ImageServer)
    server._stop_event = threading.Event()
    server._zmq_publisher_manager = object()

    server._rgbd_zmq_pub("head_camera", _FailingAtomicCamera())

    assert not server._stop_event.is_set()


def test_atomic_publisher_does_not_republish_a_stale_capture():
    server = object.__new__(ImageServer)
    server._stop_event = threading.Event()
    server._zmq_publisher_manager = _RecordingPublisher()

    server._rgbd_zmq_pub("head_camera", _RepeatedThenFailingAtomicCamera())

    assert server._zmq_publisher_manager.calls == [(b"same-capture", 5560)]
    assert not server._stop_event.is_set()
