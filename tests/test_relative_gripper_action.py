import unittest

import numpy as np
from omegaconf import OmegaConf

from reactive_diffusion_policy.common.action_utils import (
    absolute_gripper_actions_to_relative_actions,
    relative_gripper_actions_to_absolute_actions,
)
from reactive_diffusion_policy.dataset.real_image_tactile_dataset import RealImageTactileDataset
from reactive_diffusion_policy.env_runner.real_runner import RealRunner
from eval_real_robot_flexiv import _sync_relative_gripper_action_from_checkpoint


class RelativeGripperActionUtilsTest(unittest.TestCase):
    def test_single_arm_round_trip_preserves_tcp_action(self):
        actions = np.arange(30, dtype=np.float32).reshape(3, 10) / 100.0
        base = np.array([0.12], dtype=np.float32)

        relative = absolute_gripper_actions_to_relative_actions(actions, base)

        np.testing.assert_array_equal(relative[:, :9], actions[:, :9])
        np.testing.assert_allclose(relative[:, 9], actions[:, 9] - base[0])
        np.testing.assert_allclose(
            relative_gripper_actions_to_absolute_actions(relative, base),
            actions,
        )

    def test_bimanual_round_trip_uses_one_base_per_gripper(self):
        actions = np.zeros((2, 20), dtype=np.float32)
        actions[:, 18:] = [[0.10, 0.11], [0.09, 0.13]]
        base = np.array([0.10, 0.12], dtype=np.float32)

        relative = absolute_gripper_actions_to_relative_actions(actions, base)

        np.testing.assert_allclose(
            relative[:, 18:],
            [[0.0, -0.01], [-0.01, 0.01]],
            atol=1e-7,
        )
        np.testing.assert_allclose(
            relative_gripper_actions_to_absolute_actions(relative, base),
            actions,
        )


class RelativeGripperActionDatasetTest(unittest.TestCase):
    def test_uses_latest_observed_width_as_horizon_base(self):
        dataset = RealImageTactileDataset.__new__(RealImageTactileDataset)
        dataset.relative_gripper_action = True
        action = np.zeros((3, 10), dtype=np.float32)
        action[:, 9] = [0.12, 0.115, 0.11]
        obs = {
            "left_robot_gripper_width": np.array([[0.125], [0.12]], dtype=np.float32)
        }

        result = dataset._maybe_make_gripper_action_relative(action, obs)

        np.testing.assert_allclose(result[:, 9], [0.0, -0.005, -0.01], atol=1e-7)

    def test_disabled_option_preserves_absolute_action(self):
        dataset = RealImageTactileDataset.__new__(RealImageTactileDataset)
        dataset.relative_gripper_action = False
        action = np.arange(20, dtype=np.float32).reshape(2, 10)

        result = dataset._maybe_make_gripper_action_relative(action, {})

        np.testing.assert_array_equal(result, action)


class RelativeGripperActionRunnerTest(unittest.TestCase):
    def test_eval_restores_absolute_width_from_latest_observation(self):
        runner = object.__new__(RealRunner)
        runner.use_relative_gripper_action = True
        runner.shape_meta = {"action": {"shape": [10]}}
        relative_action = np.zeros((3, 10), dtype=np.float32)
        relative_action[:, 9] = [0.0, -0.005, -0.01]
        obs = {
            "left_robot_gripper_width": np.array([[0.125], [0.12]], dtype=np.float32)
        }

        result = runner._maybe_make_gripper_action_absolute(relative_action, obs)

        np.testing.assert_allclose(result[:, 9], [0.12, 0.115, 0.11], atol=1e-7)

    def test_eval_uses_action_representation_saved_in_checkpoint(self):
        cfg = OmegaConf.create({
            "task": {
                "dataset": {"relative_gripper_action": False},
                "env_runner": {"use_relative_gripper_action": False},
            }
        })
        payload = {
            "cfg": OmegaConf.create({
                "task": {"dataset": {"relative_gripper_action": True}}
            })
        }

        _sync_relative_gripper_action_from_checkpoint(cfg, payload)

        self.assertTrue(cfg.task.dataset.relative_gripper_action)
        self.assertTrue(cfg.task.env_runner.use_relative_gripper_action)

    def test_old_checkpoint_defaults_to_absolute_gripper_action(self):
        cfg = OmegaConf.create({
            "task": {
                "dataset": {"relative_gripper_action": True},
                "env_runner": {"use_relative_gripper_action": True},
            }
        })

        _sync_relative_gripper_action_from_checkpoint(
            cfg,
            {"cfg": OmegaConf.create({"task": {"dataset": {}}})},
        )

        self.assertFalse(cfg.task.dataset.relative_gripper_action)
        self.assertFalse(cfg.task.env_runner.use_relative_gripper_action)


if __name__ == "__main__":
    unittest.main()
