from unittest import mock

import cv2
import numpy as np
import pytest

from teleimager import image_client as image_client_module
from teleimager.image_client import (
    ImageClient,
    TeleImage,
    ZMQ_PublisherThread,
    ZMQ_SubscriberThread,
    decode_rgbd_frame,
)
from teleimager.rgbd_protocol import RGBD_PROTOCOL, pack_rgbd_packet, unpack_rgbd_packet


def _encoded_rgbd(sequence=7):
    color = np.zeros((4, 5, 3), dtype=np.uint8)
    color[..., 1] = 150
    depth = np.arange(20, dtype=np.uint16).reshape(4, 5) * 37
    color_ok, color_jpeg = cv2.imencode(".jpg", color)
    depth_ok, depth_png = cv2.imencode(".png", depth)
    assert color_ok and depth_ok
    packet = pack_rgbd_packet(
        sequence,
        1234,
        color_jpeg.tobytes(),
        depth_png.tobytes(),
    )
    return packet, color, depth


def test_publisher_application_queue_keeps_newest_value():
    publisher = ZMQ_PublisherThread(port=1, context=object())

    publisher.send(b"oldest")
    publisher.send(b"middle")
    publisher.send(b"newest")

    assert publisher._queue.qsize() == 1
    assert publisher._queue.get_nowait() == b"newest"
    publisher._running = False


def test_teleimage_iteration_remains_legacy_compatible():
    bgr = np.zeros((1, 1, 3), dtype=np.uint8)
    image = TeleImage(
        fps=30.0,
        jpg=b"jpeg",
        bgr=bgr,
        received_monotonic_ns=123,
    )

    fps, jpeg, unpacked_bgr = image

    assert fps == 30.0
    assert jpeg == b"jpeg"
    assert unpacked_bgr is bgr
    assert image.received_monotonic_ns == 123


def test_subscriber_returns_bytes_and_receive_time_atomically():
    subscriber = ZMQ_SubscriberThread("127.0.0.1", 1, request_bgr=False)
    subscriber._jpg_3ring_buffer.write((b"payload", 456))

    image = subscriber.recv()

    assert image.jpg == b"payload"
    assert image.received_monotonic_ns == 456
    subscriber._running = False


def test_decode_preserves_uint16_depth_and_pair_shape():
    packet, _, expected_depth = _encoded_rgbd()
    frame = unpack_rgbd_packet(packet, received_monotonic_ns=88)

    color_bgr, depth = decode_rgbd_frame(frame)

    assert color_bgr.shape == (4, 5, 3)
    assert depth.dtype == np.uint16
    np.testing.assert_array_equal(depth, expected_depth)


def test_decode_rejects_non_uint16_depth():
    color = np.zeros((2, 2, 3), dtype=np.uint8)
    depth = np.zeros((2, 2), dtype=np.uint8)
    _, color_jpeg = cv2.imencode(".jpg", color)
    _, depth_png = cv2.imencode(".png", depth)
    frame = unpack_rgbd_packet(
        pack_rgbd_packet(1, 2, color_jpeg.tobytes(), depth_png.tobytes())
    )

    with pytest.raises(ValueError, match="uint16"):
        decode_rgbd_frame(frame)


def test_decode_rejects_mismatched_image_shapes():
    color = np.zeros((2, 3, 3), dtype=np.uint8)
    depth = np.zeros((4, 5), dtype=np.uint16)
    _, color_jpeg = cv2.imencode(".jpg", color)
    _, depth_png = cv2.imencode(".png", depth)
    frame = unpack_rgbd_packet(
        pack_rgbd_packet(1, 2, color_jpeg.tobytes(), depth_png.tobytes())
    )

    with pytest.raises(ValueError, match="shapes"):
        decode_rgbd_frame(frame)


class _FakeSubscriberManager:
    def __init__(self, image, fail_subscribe=False):
        self.image = image
        self.fail_subscribe = fail_subscribe
        self.calls = []
        self.closed = False

    def subscribe(self, host, port, request_bgr=False):
        self.calls.append((host, port, request_bgr))
        if self.fail_subscribe:
            raise RuntimeError("subscription failed")
        return self.image

    def close(self):
        self.closed = True


def _client_with_rgbd_message(image, protocol=RGBD_PROTOCOL):
    client = ImageClient.__new__(ImageClient)
    client._host = "camera-host"
    client._cam_config = {
        "head_camera": {
            "enable_depth": True,
            "rgbd_zmq_port": 5560,
            "rgbd_protocol": protocol,
        }
    }
    client._subscriber_manager = _FakeSubscriberManager(image)
    return client


def test_get_head_rgbd_frame_parses_sequence_and_receive_time():
    packet, _, _ = _encoded_rgbd(sequence=42)
    client = _client_with_rgbd_message(
        TeleImage(fps=30.0, jpg=packet, received_monotonic_ns=999)
    )

    frame = client.get_head_rgbd_frame()

    assert frame.sequence == 42
    assert frame.received_monotonic_ns == 999
    assert client._subscriber_manager.calls == [("camera-host", 5560, False)]


def test_get_head_rgbd_frame_rejects_unknown_protocol_without_subscribing():
    packet, _, _ = _encoded_rgbd()
    client = _client_with_rgbd_message(
        TeleImage(fps=30.0, jpg=packet),
        protocol="teleimager-rgbd-v2",
    )

    assert client.get_head_rgbd_frame() is None
    assert client._subscriber_manager.calls == []


def test_eager_flags_preserve_lazy_getters_without_legacy_depth_bandwidth():
    config = {
        "head_camera": {
            "enable_zmq": True,
            "enable_webrtc": False,
            "zmq_port": 5555,
            "enable_depth": True,
            "depth_zmq_port": 5558,
            "raw_depth_zmq_port": 5559,
            "rgbd_zmq_port": 5560,
            "rgbd_protocol": RGBD_PROTOCOL,
        },
        "left_wrist_camera": {"enable_zmq": False},
        "right_wrist_camera": {"enable_zmq": False},
    }
    manager = _FakeSubscriberManager(TeleImage(fps=0.0, jpg=None))
    requester = mock.Mock()
    requester.request.return_value = config

    with mock.patch.object(
        image_client_module.ZMQ_SubscriberManager,
        "get_instance",
        return_value=manager,
    ), mock.patch.object(image_client_module, "ZMQ_Requester", return_value=requester):
        client = ImageClient(
            host="camera-host",
            eager_head_color=False,
            eager_aligned_depth=False,
            eager_raw_depth=False,
        )

    assert manager.calls == []
    assert client.get_head_depth_frame() is None
    assert manager.calls == [("camera-host", 5558, False)]
    requester.close.assert_called_once_with()


def test_constructor_cleans_up_after_subscription_failure():
    config = {
        "head_camera": {
            "enable_zmq": True,
            "enable_webrtc": False,
            "zmq_port": 5555,
            "enable_depth": False,
        },
        "left_wrist_camera": {"enable_zmq": False},
        "right_wrist_camera": {"enable_zmq": False},
    }
    manager = _FakeSubscriberManager(None, fail_subscribe=True)
    requester = mock.Mock()
    requester.request.return_value = config

    with mock.patch.object(
        image_client_module.ZMQ_SubscriberManager,
        "get_instance",
        return_value=manager,
    ), mock.patch.object(image_client_module, "ZMQ_Requester", return_value=requester):
        with pytest.raises(RuntimeError, match="subscription failed"):
            ImageClient(host="camera-host")

    requester.close.assert_called_once_with()
    assert manager.closed


def test_missing_paired_config_leaves_legacy_client_compatible():
    client = ImageClient.__new__(ImageClient)
    client._host = "camera-host"
    client._cam_config = {"head_camera": {"enable_depth": True}}
    client._subscriber_manager = _FakeSubscriberManager(None)

    assert client.get_head_rgbd_frame() is None
    assert client._subscriber_manager.calls == []
