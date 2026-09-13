from pathlib import Path

import yaml


def test_repository_default_config_does_not_advertise_atomic_rgbd():
    config_path = Path(__file__).resolve().parents[1] / "cam_config_server.yaml"
    config = yaml.safe_load(config_path.read_text())
    head_config = config["head_camera"]

    assert "rgbd_zmq_port" not in head_config
    assert "rgbd_protocol" not in head_config
