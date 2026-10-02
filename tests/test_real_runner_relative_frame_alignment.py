import unittest

import numpy as np

from reactive_diffusion_policy.common.action_utils import (
    absolute_actions_to_relative_actions,
    relative_actions_to_absolute_actions,
)
from reactive_diffusion_policy.common.space_utils import (
    homo_matrix_to_pose_9d_batch,
    pose_3d_9d_to_homo_matrix_batch,
)
from reactive_diffusion_policy.env_runner.real_runner import RealRunner


def _pose9(matrix):
    return homo_matrix_to_pose_9d_batch(matrix[None])[0]


def _runner(dataset_start_pose):
    runner = object.__new__(RealRunner)
    runner.use_relative_action = True
    runner.use_relative_tcp_obs_for_relative_action = True
    runner.policy_tcp_pose_obs_mode = "dataset_tool_rotation_offset"
    runner.policy_relative_action_frame_mode = "dataset_tool_rotation_offset"
    runner._policy_tcp_pose_obs_offset_mat = None
    runner._policy_tcp_pose_obs_xyz_offset = None
    runner._policy_tcp_pose_obs_warned_missing = False
    runner._policy_relative_action_frame_offset_mat = None
    runner._policy_relative_action_frame_offset_inv = None
    runner._policy_relative_action_frame_logged = False
    runner._load_policy_tcp_pose_dataset_start = lambda: {
        "pose9": dataset_start_pose,
        "path": "test",
        "episode_idx": 0,
        "episode_count": 1,
        "row_start": 0,
        "row_end": 2,
        "source": "test",
    }
    return runner


class RotationOnlyFrameAlignmentTest(unittest.TestCase):
    def setUp(self):
        self.dataset_start = np.eye(4)
        self.dataset_start[:3, :3] = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        self.dataset_start[:3, 3] = [0.67, -0.22, 0.73]

        self.real_start = np.eye(4)
        self.real_start[:3, :3] = np.array([
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        self.real_start[:3, 3] = [0.58, 0.10, 0.42]

    def test_action_alignment_preserves_training_world_translation(self):
        dataset_target = self.dataset_start.copy()
        dataset_target[:3, 3] += [0.02, 0.04, -0.18]
        dataset_actions = np.stack([_pose9(dataset_target)])
        dataset_relative = absolute_actions_to_relative_actions(
            dataset_actions,
            base_absolute_action=_pose9(self.dataset_start),
        )

        runner = _runner(_pose9(self.dataset_start))
        aligned_relative = runner._maybe_transform_relative_action_frame(
            dataset_relative,
            _pose9(self.real_start),
        )
        real_target = relative_actions_to_absolute_actions(
            aligned_relative,
            _pose9(self.real_start),
        )[0]

        np.testing.assert_allclose(
            real_target[:3] - self.real_start[:3, 3],
            dataset_target[:3, 3] - self.dataset_start[:3, 3],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            runner._policy_relative_action_frame_offset_mat[:3, 3],
            0.0,
            atol=1e-12,
        )

    def test_policy_obs_alignment_is_inverse_of_action_alignment(self):
        runner = _runner(_pose9(self.dataset_start))
        runner._ensure_policy_relative_action_frame_alignment(_pose9(self.real_start))
        offset = runner._policy_relative_action_frame_offset_mat

        dataset_relative = np.eye(4)
        dataset_relative[:3, 3] = [0.01, -0.03, -0.08]
        real_relative = offset @ dataset_relative @ np.linalg.inv(offset)
        real_next = self.real_start @ real_relative
        real_sequence = np.stack([_pose9(self.real_start), _pose9(real_next)])

        policy_obs = {"left_robot_tcp_pose": real_sequence.copy()}
        real_obs = {"left_robot_tcp_pose": real_sequence.copy()}
        runner._maybe_adjust_policy_tcp_pose_obs(policy_obs, real_obs)
        adjusted_relative = absolute_actions_to_relative_actions(
            policy_obs["left_robot_tcp_pose"],
            base_absolute_action=policy_obs["left_robot_tcp_pose"][-1],
        )

        expected_fake = pose_3d_9d_to_homo_matrix_batch(real_sequence) @ offset[None]
        expected_pose = homo_matrix_to_pose_9d_batch(expected_fake)
        expected_relative = absolute_actions_to_relative_actions(
            expected_pose,
            base_absolute_action=expected_pose[-1],
        )
        np.testing.assert_allclose(adjusted_relative, expected_relative, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
