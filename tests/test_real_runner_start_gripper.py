from unittest.mock import Mock

from reactive_diffusion_policy.env_runner.real_runner import RealRunner


def test_explicit_start_width_takes_precedence_over_full_open():
    runner = RealRunner.__new__(RealRunner)
    runner.start_gripper_width_mm = 52.0
    runner.open_gripper_on_start = True
    runner.env = Mock()

    runner._set_startup_gripper_state(settle_seconds=0.0)

    runner.env.send_gripper_width_m_direct.assert_called_once_with(0.052)
    runner.env.send_gripper_command_direct.assert_not_called()


def test_full_open_remains_the_default_startup_command():
    runner = RealRunner.__new__(RealRunner)
    runner.start_gripper_width_mm = None
    runner.open_gripper_on_start = True
    runner.env = Mock()
    runner.env.max_gripper_width = 0.15

    runner._set_startup_gripper_state(settle_seconds=0.0)

    runner.env.send_gripper_command_direct.assert_called_once_with(0.15, 0.15)
