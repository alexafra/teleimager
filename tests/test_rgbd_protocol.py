import pytest

from teleimager.rgbd_protocol import (
    RGBD_MAGIC,
    RGBD_PROTOCOL,
    RGBD_VERSION,
    pack_rgbd_packet,
    unpack_rgbd_packet,
)


def test_round_trip_preserves_metadata_and_encoded_images():
    packet = pack_rgbd_packet(17, 987654321, b"jpeg-data", b"png-data")

    frame = unpack_rgbd_packet(packet, received_monotonic_ns=123456789)

    assert RGBD_MAGIC == b"TELRGBD\0"
    assert RGBD_VERSION == 1
    assert RGBD_PROTOCOL == "teleimager-rgbd-v1"
    assert frame.sequence == 17
    assert frame.server_capture_monotonic_ns == 987654321
    assert frame.received_monotonic_ns == 123456789
    assert frame.color_jpeg == b"jpeg-data"
    assert frame.aligned_depth_png == b"png-data"


def test_unpack_rejects_wrong_magic_and_version():
    packet = bytearray(pack_rgbd_packet(1, 2, b"jpeg", b"png"))
    packet[0] ^= 0xFF
    with pytest.raises(ValueError, match="magic"):
        unpack_rgbd_packet(packet)

    packet = bytearray(pack_rgbd_packet(1, 2, b"jpeg", b"png"))
    packet[8] = RGBD_VERSION + 1
    with pytest.raises(ValueError, match="version"):
        unpack_rgbd_packet(packet)


def test_unpack_rejects_truncated_and_trailing_data():
    packet = pack_rgbd_packet(1, 2, b"jpeg", b"png")
    with pytest.raises(ValueError, match="length"):
        unpack_rgbd_packet(packet[:-1])
    with pytest.raises(ValueError, match="length"):
        unpack_rgbd_packet(packet + b"unexpected")


def test_pack_rejects_invalid_fields():
    with pytest.raises(TypeError):
        pack_rgbd_packet(True, 2, b"jpeg", b"png")
    with pytest.raises(ValueError):
        pack_rgbd_packet(-1, 2, b"jpeg", b"png")
    with pytest.raises(ValueError):
        pack_rgbd_packet(1, 2, b"", b"png")
    with pytest.raises(TypeError):
        pack_rgbd_packet(1, 2, "jpeg", b"png")


def test_received_timestamp_is_local_metadata_not_wire_data():
    packet = pack_rgbd_packet(3, 4, b"jpeg", b"png")
    assert unpack_rgbd_packet(packet).received_monotonic_ns is None
    assert unpack_rgbd_packet(packet, received_monotonic_ns=99).received_monotonic_ns == 99
