import numpy as np
import pytest

from scripts.set_robotiq_gripper_width import (
    command_gripper_width,
    write_last_normalized_record,
)


class _Response:
    def raise_for_status(self):
        pass

    def json(self):
        return {"message": "Gripper target queued"}


def test_converts_millimeters_to_server_meter_payload(monkeypatch):
    request = {}

    def fake_post(url, json, timeout):
        request.update(url=url, json=json, timeout=timeout)
        return _Response()

    monkeypatch.setattr("scripts.set_robotiq_gripper_width.requests.post", fake_post)

    result = command_gripper_width(44, velocity=0.05, force=20, timeout=3)

    assert request == {
        "url": "http://127.0.0.1:8092/move_gripper/left",
        "json": {"width": 0.044, "velocity": 0.05, "force_limit": 20.0},
        "timeout": 3.0,
    }
    assert result == {"message": "Gripper target queued"}


@pytest.mark.parametrize("width_mm", [-1, 86])
def test_rejects_width_outside_default_robotiq_stroke(width_mm):
    with pytest.raises(ValueError, match="width_mm must be within"):
        command_gripper_width(width_mm, dry_run=True)


def test_writes_final_normalized_frame_as_text(tmp_path):
    output = write_last_normalized_record(
        tmp_path / "last.txt",
        timestamp=123.5,
        elapsed=4.25,
        phase="after_command",
        sample_count=99,
        normalized_xyz=np.array(
            [[1.0, -2.5, 3.25], [4.0, 5.0, -6.0]], dtype="float32"
        ),
    )

    assert output.read_text() == (
        "Final normalized magnet change\n"
        "timestamp=123.500000000\n"
        "elapsed=4.250000 s\n"
        "phase=after_command\n"
        "magnet_sample_count=99\n"
        "S1: x=1.000000 y=-2.500000 z=3.250000\n"
        "S2: x=4.000000 y=5.000000 z=-6.000000\n"
    )
