import unittest
from unittest.mock import Mock

import numpy as np

from reactive_diffusion_policy.dataset.real_image_tactile_dataset import RealImageTactileDataset
from reactive_diffusion_policy.env.franka_polymetis.franka_polymetis_env import FrankaPolymetisEnv


class ZeroGripperActionDatasetTest(unittest.TestCase):
    def test_zeroes_only_single_arm_gripper_action(self):
        dataset = RealImageTactileDataset.__new__(RealImageTactileDataset)
        dataset.zero_gripper_action = True
        action = np.arange(30, dtype=np.float32).reshape(3, 10)
        expected_tcp_action = action[:, :9].copy()

        result = dataset._maybe_zero_gripper_action(action.copy())

        np.testing.assert_array_equal(result[:, :9], expected_tcp_action)
        np.testing.assert_array_equal(result[:, 9], np.zeros(3, dtype=np.float32))

    def test_disabled_ablation_preserves_action(self):
        dataset = RealImageTactileDataset.__new__(RealImageTactileDataset)
        dataset.zero_gripper_action = False
        action = np.arange(20, dtype=np.float32).reshape(2, 10)

        result = dataset._maybe_zero_gripper_action(action.copy())

        np.testing.assert_array_equal(result, action)


class IgnorePolicyGripperCommandTest(unittest.TestCase):
    def test_policy_commands_can_be_ignored_without_blocking_direct_commands(self):
        env = FrankaPolymetisEnv.__new__(FrankaPolymetisEnv)
        env.ignore_gripper_commands = False
        env.ignore_policy_gripper_commands = True
        env.gripper_action_mode = "continuous"
        env._send_gripper_raw = Mock()

        env.send_gripper_command(0.0, 0.0)
        env._send_gripper_raw.assert_not_called()

        env.send_gripper_command_direct(0.14, 0.14)
        env._send_gripper_raw.assert_called_once_with(0.14)


if __name__ == "__main__":
    unittest.main()
