import pytest

from scripts.open_close_robotiq_gripper import run_open_close


class FakeGripper:
    def __init__(self, **kwargs):
        self.calls = [("init", kwargs)]

    def reset(self):
        self.calls.append(("reset",))

    def activate(self, **kwargs):
        self.calls.append(("activate", kwargs))

    def close(self, **kwargs):
        self.calls.append(("close", kwargs))

    def open(self, **kwargs):
        self.calls.append(("open", kwargs))

    def calibrate_bit(self, **kwargs):
        self.calls.append(("calibrate_bit", kwargs))

    def calibrate_mm(self, **kwargs):
        self.calls.append(("calibrate_mm", kwargs))

    def move_mm(self, width_mm, **kwargs):
        self.calls.append(("move_mm", width_mm, kwargs))

    def disconnect(self):
        self.calls.append(("disconnect",))


def test_runs_close_open_cycles_without_robot_arm_dependency():
    created = []
    sleeps = []

    def factory(**kwargs):
        gripper = FakeGripper(**kwargs)
        created.append(gripper)
        return gripper

    run_open_close(
        port="/dev/ttyUSB1",
        device_id=9,
        cycles=2,
        hold_sec=0.5,
        start_delay_sec=3.0,
        reset_wait_sec=1.0,
        speed=200,
        force=80,
        gripper_factory=factory,
        sleep_fn=sleeps.append,
    )

    assert created[0].calls == [
        ("init", {"com_port": "/dev/ttyUSB1", "device_id": 9}),
        ("reset",),
        ("activate", {"reset": False}),
        ("close", {"speed": 200, "force": 80, "wait": True}),
        ("open", {"speed": 200, "force": 80, "wait": True}),
        ("close", {"speed": 200, "force": 80, "wait": True}),
        ("open", {"speed": 200, "force": 80, "wait": True}),
        ("disconnect",),
    ]
    assert sleeps == [1.0, 3.0, 0.5, 0.5, 0.5, 0.5]


def test_skip_reset_activates_without_reset():
    gripper = FakeGripper()

    run_open_close(
        cycles=1,
        hold_sec=0,
        start_delay_sec=0,
        skip_reset=True,
        gripper_factory=lambda **kwargs: gripper,
        sleep_fn=lambda _: None,
    )

    assert ("reset",) not in gripper.calls
    assert ("activate", {"reset": False}) in gripper.calls
    assert gripper.calls[-1] == ("disconnect",)


def test_opens_to_requested_millimeter_width_without_extra_calibration_motion():
    gripper = FakeGripper()

    run_open_close(
        cycles=1,
        hold_sec=0,
        start_delay_sec=0,
        open_width_mm=44,
        max_width_mm=85,
        speed=180,
        force=70,
        gripper_factory=lambda **kwargs: gripper,
        sleep_fn=lambda _: None,
    )

    assert ("calibrate_bit", {"openbit": 0, "closebit": 255}) in gripper.calls
    assert ("calibrate_mm", {"closemm": 0.0, "openmm": 85.0}) in gripper.calls
    assert (
        "move_mm",
        44.0,
        {"speed": 180, "force": 70, "wait": True},
    ) in gripper.calls
    assert not any(call[0] == "open" for call in gripper.calls)


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"cycles": 0}, "cycles must be positive"),
        ({"hold_sec": -1}, "hold_sec must be non-negative"),
        ({"speed": 256}, "speed must be within"),
        ({"force": -1}, "force must be within"),
        ({"open_width_mm": -1}, "open_width_mm must be within"),
        ({"open_width_mm": 86}, "open_width_mm must be within"),
        ({"max_width_mm": 0}, "max_width_mm must be positive"),
    ],
)
def test_rejects_invalid_motion_settings(kwargs, error):
    with pytest.raises(ValueError, match=error):
        run_open_close(gripper_factory=FakeGripper, **kwargs)
