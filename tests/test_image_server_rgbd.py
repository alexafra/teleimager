import hashlib
import json
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
    CANONICAL_DEPTH_SCALE_M_PER_UNIT,
    ImageServer,
    REALSENSE_CALIBRATION_SCHEMA,
    RealSenseCamera,
    _advertise_realsense_depth_metadata,
    _configured_transport_ports,
    _realsense_calibration,
    _validated_realsense_depth_scale,
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


class _VideoProfile:
    def __init__(self, intrinsics, format_name, fps, extrinsics=None):
        self._intrinsics = intrinsics
        self._format_name = format_name
        self._fps = fps
        self._extrinsics = extrinsics

    def get_intrinsics(self):
        return self._intrinsics

    def format(self):
        return self._format_name

    def fps(self):
        return self._fps

    def get_extrinsics_to(self, _other):
        return self._extrinsics


class _Device:
    def __init__(self, info):
        self._info = info

    def supports(self, key):
        return key in self._info

    def get_info(self, key):
        return self._info[key]


def test_sdk_scale_is_float32_validated_but_processing_scale_is_exact():
    processing, reported = _validated_realsense_depth_scale(
        0.0010000000474974513
    )

    assert processing == CANONICAL_DEPTH_SCALE_M_PER_UNIT == 0.001
    assert reported == 0.0010000000474974513


@pytest.mark.parametrize("reported", [0.0005, 0.0011, float("nan"), float("inf")])
def test_incompatible_sdk_depth_scale_is_rejected(reported):
    with pytest.raises(ValueError, match="depth scale"):
        _validated_realsense_depth_scale(reported)


def test_active_realsense_calibration_has_canonical_schema_and_fingerprint():
    rs = types.SimpleNamespace(
        camera_info=types.SimpleNamespace(
            name="name",
            serial_number="serial",
            product_id="product_id",
            firmware_version="firmware",
        )
    )
    device = _Device(
        {
            "name": "Intel RealSense D435I",
            "serial": "254322071415",
            "product_id": "0B3A",
            "firmware": "5.15.1.55",
        }
    )
    color_intrinsics = types.SimpleNamespace(
        width=640,
        height=480,
        fx=609.3858642578125,
        fy=609.4705200195312,
        ppx=325.95001220703125,
        ppy=247.26507568359375,
        model="distortion.inverse_brown_conrady",
        coeffs=[0.0] * 5,
    )
    depth_intrinsics = types.SimpleNamespace(
        width=640,
        height=480,
        fx=397.5912170410156,
        fy=397.5912170410156,
        ppx=315.6465148925781,
        ppy=244.2028350830078,
        model="distortion.brown_conrady",
        coeffs=[0.0] * 5,
    )
    extrinsics = types.SimpleNamespace(
        rotation=[
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
        translation=[
            0.014800711534917355,
            0.0008831368759274483,
            0.0007359444862231612,
        ],
    )
    color_profile = _VideoProfile(color_intrinsics, "format.bgr8", 30)
    depth_profile = _VideoProfile(
        depth_intrinsics,
        "format.z16",
        30,
        extrinsics=extrinsics,
    )

    calibration = _realsense_calibration(
        rs,
        device,
        color_profile,
        depth_profile,
    )

    assert calibration["schema"] == REALSENSE_CALIBRATION_SCHEMA
    assert calibration["camera"] == {
        "model": "Intel RealSense D435I",
        "serial": "254322071415",
        "product_id": "0B3A",
        "firmware": "5.15.1.55",
    }
    assert calibration["color"]["format"] == "bgr8"
    assert calibration["color"]["cx"] == 325.95001220703125
    assert calibration["depth"]["format"] == "z16"
    assert calibration["depth_to_color"]["rotation"] == extrinsics.rotation
    assert calibration["fingerprint"] == (
        "sha256:f7860e3e2be34af131e214c74d417217c889eff3a069dbec543be3fba15b027b"
    )
    fingerprint = calibration.pop("fingerprint")
    canonical_json = json.dumps(
        calibration,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    assert fingerprint == f"sha256:{hashlib.sha256(canonical_json).hexdigest()}"


def test_advertised_depth_metadata_uses_canonical_scale_and_is_copied():
    camera = types.SimpleNamespace(
        depth_scale_reported_m_per_unit=0.0010000000474974513,
        calibration={
            "schema": REALSENSE_CALIBRATION_SCHEMA,
            "fingerprint": "sha256:abc",
        },
    )
    config = {}

    _advertise_realsense_depth_metadata(config, camera)
    camera.calibration["schema"] = "mutated"

    assert config["depth_scale_m_per_unit"] == 0.001
    assert (
        config["depth_scale_reported_m_per_unit"]
        == 0.0010000000474974513
    )
    assert config["calibration"]["schema"] == REALSENSE_CALIBRATION_SCHEMA


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
