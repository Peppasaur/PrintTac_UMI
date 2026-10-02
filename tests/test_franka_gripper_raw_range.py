from unittest.mock import Mock, patch

import numpy as np

import reactive_diffusion_policy.env.franka_polymetis.franka_polymetis_env as env_module
from reactive_diffusion_policy.env.franka_polymetis.franka_polymetis_env import FrankaPolymetisEnv


def test_trajectory_zero_to_max_uses_action_max_and_zero_closed_endpoint():
    root = {
        "data": {
            "action": np.array(
                [
                    [0.0, 0.11],
                    [0.0, 0.15],
                    [0.0, 0.13],
                ],
                dtype=np.float64,
            ),
            # A different observation range must not affect action-label calibration.
            "left_robot_gripper_width": np.array([[0.07], [0.20]], dtype=np.float64),
        }
    }
    store = Mock()

    with patch.object(env_module, "_dataset_candidates", return_value=["fixture.zarr"]), \
            patch.object(env_module.os.path, "exists", return_value=True), \
            patch.object(env_module, "_open_zarr", return_value=(root, store)):
        result = env_module._infer_gripper_raw_range(
            "fixture.zarr",
            range_mode="trajectory_zero_to_max",
        )

    assert result[:3] == (0.0, 0.15, 0.15)
    store.close.assert_called_once()


def test_calibrated_marker_range_uses_zero_calibrated_gap_as_raw_minimum():
    root = {
        "data": {
            "action": np.array([[0.0, 0.11], [0.0, 0.15]], dtype=np.float64),
        }
    }

    with patch.object(env_module, "_dataset_candidates", return_value=["fixture.zarr"]), \
            patch.object(env_module.os.path, "exists", return_value=True), \
            patch.object(env_module, "_open_zarr", return_value=(root, Mock())):
        result = env_module._infer_gripper_raw_range(
            "fixture.zarr",
            range_mode="calibrated_marker_trajectory",
            calibration_scale=0.8,
            calibration_offset=-0.04,
        )

    np.testing.assert_allclose(result[:3], (0.05, 0.15, 0.15), atol=1e-12)


def test_calibrated_marker_width_range_uses_physical_stroke_endpoint():
    root = {
        "data": {
            "action": np.array([[0.0, 0.053], [0.0, 0.112]], dtype=np.float64),
        }
    }

    with patch.object(env_module, "_dataset_candidates", return_value=["fixture.zarr"]), \
            patch.object(env_module.os.path, "exists", return_value=True), \
            patch.object(env_module, "_open_zarr", return_value=(root, Mock())):
        result = env_module._infer_gripper_raw_range(
            "fixture.zarr",
            range_mode="calibrated_marker_width",
            calibration_scale=0.8,
            calibration_offset=-0.04,
            gripper_stroke=0.08,
        )

    np.testing.assert_allclose(result[:3], (0.05, 0.15, 0.112), atol=1e-12)


def test_zero_to_max_range_maps_raw_zero_and_action_max_to_physical_endpoints():
    env = FrankaPolymetisEnv.__new__(FrankaPolymetisEnv)
    env.gripper_raw_range_mode = "trajectory_zero_to_max"
    env.gripper_raw_min = 0.0
    env.gripper_raw_max = 0.15
    env.gripper_stroke = 0.085
    env.last_gripper_width_target = [0.0]

    assert env._raw_gripper_to_width(0.0) == 0.0
    assert env._raw_gripper_to_width(0.15) == 0.085
    assert env._raw_gripper_to_width(0.075) == 0.0425
    assert env._width_to_raw_gripper(0.0) == 0.0
    assert env._width_to_raw_gripper(0.085) == 0.15


def test_teleop_raw_open_ratio_maps_directly_to_physical_width():
    env = FrankaPolymetisEnv.__new__(FrankaPolymetisEnv)
    env.gripper_raw_range_mode = "trajectory_zero_to_max"
    env.gripper_raw_min = 0.0
    env.gripper_raw_max = 1.0
    env.gripper_stroke = 0.085
    env.last_gripper_width_target = [0.0]

    np.testing.assert_allclose(env._raw_gripper_to_width(0.0), 0.0, atol=1e-12)
    np.testing.assert_allclose(env._raw_gripper_to_width(0.44), 0.0374, atol=1e-12)
    np.testing.assert_allclose(env._raw_gripper_to_width(1.0), 0.085, atol=1e-12)
    np.testing.assert_allclose(env._width_to_raw_gripper(0.0374), 0.44, atol=1e-12)


def test_physical_width_action_round_trips_through_commanded_observation():
    env = FrankaPolymetisEnv.__new__(FrankaPolymetisEnv)
    env.gripper_raw_range_mode = "calibrated_marker_width"
    env.gripper_raw_calibration_scale = 1.0
    env.gripper_raw_calibration_offset = 0.0
    env.gripper_raw_min = 0.0
    env.gripper_raw_max = 0.085
    env.gripper_stroke = 0.085
    env.gripper_command_width_offset = 0.0
    env.gripper_obs_raw_offset = 0.0
    env.gripper_velocity = 0.08
    env.grasp_force = 20.0
    env.gripper_http_timeout = 5.0
    env.last_gripper_width_target = [0.08]
    env.send_command = Mock(return_value={})

    for policy_width in (0.037141006, 0.05, 0.08):
        env._send_gripper_raw(policy_width)
        endpoint, payload = env.send_command.call_args.args

        assert endpoint == "/move_gripper/left"
        np.testing.assert_allclose(payload["width"], policy_width, atol=1e-12)
        np.testing.assert_allclose(
            env._commanded_gripper_obs_raw(),
            policy_width,
            atol=1e-12,
        )


def test_calibrated_marker_trajectory_maps_zero_gap_to_closed_and_max_gap_to_open():
    env = FrankaPolymetisEnv.__new__(FrankaPolymetisEnv)
    env.gripper_raw_range_mode = "calibrated_marker_trajectory"
    env.gripper_raw_calibration_scale = 0.8
    env.gripper_raw_calibration_offset = -0.04
    env.gripper_raw_min = 0.05
    env.gripper_raw_max = 0.15
    env.gripper_calibrated_full_open = 0.08
    env.gripper_stroke = 0.085
    env.last_gripper_width_target = [0.05]

    np.testing.assert_allclose(env._raw_gripper_to_width(0.05), 0.0, atol=1e-12)
    np.testing.assert_allclose(env._raw_gripper_to_width(0.10), 0.0425, atol=1e-12)
    np.testing.assert_allclose(env._raw_gripper_to_width(0.15), 0.085, atol=1e-12)
    np.testing.assert_allclose(env._width_to_raw_gripper(0.0), 0.05, atol=1e-12)
    np.testing.assert_allclose(env._width_to_raw_gripper(0.0425), 0.10, atol=1e-12)
    np.testing.assert_allclose(env._width_to_raw_gripper(0.085), 0.15, atol=1e-12)


def test_calibrated_marker_width_uses_calibrated_gap_without_dataset_rescaling():
    env = FrankaPolymetisEnv.__new__(FrankaPolymetisEnv)
    env.gripper_raw_range_mode = "calibrated_marker_width"
    env.gripper_raw_calibration_scale = 0.8413230600966157
    env.gripper_raw_calibration_offset = -0.03930162909704434
    env.gripper_raw_min = -env.gripper_raw_calibration_offset / env.gripper_raw_calibration_scale
    env.gripper_raw_max = (
        0.085 - env.gripper_raw_calibration_offset
    ) / env.gripper_raw_calibration_scale
    env.gripper_calibrated_full_open = None
    env.gripper_stroke = 0.085
    env.gripper_command_width_offset = 0.003
    env.gripper_obs_raw_offset = 0.0
    env.last_gripper_width_target = [env.gripper_raw_min]

    np.testing.assert_allclose(
        env._raw_gripper_to_width(0.11188576),
        0.054830440887391176,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        env._width_to_raw_gripper(0.054830440887391176),
        0.11188576,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        env._raw_gripper_to_command_width(0.11188576),
        0.05783044088739118,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        env._width_to_obs_raw_gripper(0.05783044088739118),
        0.11188576,
        atol=1e-12,
    )


def test_command_width_offset_is_added_after_raw_mapping_and_clipped_to_stroke():
    env = FrankaPolymetisEnv.__new__(FrankaPolymetisEnv)
    env.gripper_raw_range_mode = "calibrated_marker_trajectory"
    env.gripper_raw_calibration_scale = 0.8
    env.gripper_raw_calibration_offset = -0.04
    env.gripper_raw_min = 0.05
    env.gripper_raw_max = 0.15
    env.gripper_calibrated_full_open = 0.08
    env.gripper_stroke = 0.085
    env.gripper_command_width_offset = 0.006
    env.gripper_obs_raw_offset = 0.0
    env.last_gripper_width_target = [0.05]

    np.testing.assert_allclose(env._raw_gripper_to_width(0.10), 0.0425, atol=1e-12)
    np.testing.assert_allclose(
        env._raw_gripper_to_command_width(0.10),
        0.0485,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        env._raw_gripper_to_command_width(0.05),
        0.006,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        env._raw_gripper_to_command_width(0.15),
        0.085,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        env._physical_gripper_width_to_nominal_width(0.0485),
        0.0425,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        env._width_to_obs_raw_gripper(0.0485),
        0.10,
        atol=1e-12,
    )
