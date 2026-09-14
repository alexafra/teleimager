from realsense_zmq_publisher import as_color_only_config


def test_fallback_does_not_advertise_unserved_depth_streams():
    original = {
        "head_camera": {
            "enable_zmq": True,
            "zmq_port": 5555,
            "enable_depth": True,
            "depth_zmq_port": 5558,
            "raw_depth_zmq_port": 5559,
            "rgbd_zmq_port": 5560,
            "rgbd_protocol": "teleimager-rgbd-v1",
            "depth_scale_m_per_unit": 0.001,
            "depth_scale_reported_m_per_unit": 0.0010000000474974513,
            "calibration": {"schema": "realsense_rgbd_calibration.v1"},
        }
    }

    advertised = as_color_only_config(original)

    assert original["head_camera"]["enable_depth"] is True
    assert advertised["head_camera"]["enable_depth"] is False
    for key in (
        "depth_zmq_port",
        "raw_depth_zmq_port",
        "rgbd_zmq_port",
        "rgbd_protocol",
        "depth_scale_m_per_unit",
        "depth_scale_reported_m_per_unit",
        "calibration",
    ):
        assert key not in advertised["head_camera"]
