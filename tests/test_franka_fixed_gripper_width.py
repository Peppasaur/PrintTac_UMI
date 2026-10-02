from unittest.mock import Mock

from reactive_diffusion_policy.env.franka_polymetis.franka_polymetis_env import FrankaPolymetisEnv


def _make_env():
    env = FrankaPolymetisEnv.__new__(FrankaPolymetisEnv)
    env.ignore_gripper_commands = False
    env.ignore_policy_gripper_commands = False
    env.fixed_gripper_width_m = 0.055
    env.last_fixed_gripper_width_command = None
    env.last_gripper_width_target = [0.0, 0.0]
    env.gripper_stroke = 0.085
    env.gripper_velocity = 0.08
    env.grasp_force = 20.0
    env.gripper_http_timeout = 5.0
    env.send_command = Mock(return_value={})
    env._physical_gripper_width_to_nominal_width = Mock(return_value=0.049)
    env._width_to_raw_gripper = Mock(return_value=0.11)
    return env


def test_fixed_width_overrides_direct_startup_command_and_policy_output():
    env = _make_env()

    env.send_gripper_command_direct(1.0, 1.0)
    env.send_gripper_command(0.0, 0.0)

    env.send_command.assert_called_once_with(
        "/move_gripper/left",
        {"width": 0.055, "velocity": 0.08, "force_limit": 20.0},
        timeout=5.0,
    )
    assert env.last_gripper_width_target[0] == 0.11


def test_fixed_width_still_executes_when_only_policy_commands_are_ignored():
    env = _make_env()
    env.ignore_policy_gripper_commands = True

    env.send_gripper_command(0.0, 0.0)

    env.send_command.assert_called_once()


def test_physical_start_width_is_sent_once_without_policy_mapping():
    env = _make_env()
    env.fixed_gripper_width_m = None

    env.send_gripper_width_m_direct(0.052)

    env.send_command.assert_called_once_with(
        "/move_gripper/left",
        {"width": 0.052, "velocity": 0.08, "force_limit": 20.0},
        timeout=5.0,
    )
    env._physical_gripper_width_to_nominal_width.assert_called_once_with(0.052)


def test_physical_start_width_rejects_values_beyond_stroke():
    env = _make_env()
    env.fixed_gripper_width_m = None
    env.gripper_stroke = 0.085

    try:
        env.send_gripper_width_m_direct(0.090)
    except ValueError as exc:
        assert "within the gripper stroke" in str(exc)
    else:
        raise AssertionError("Expected an out-of-range startup width to fail")
