import numpy as np

from scripts.trim_dataset_after_tactile_contact import (
    build_trim_plan,
    detect_contact_frame,
    terminal_hold_actions,
    tactile_contact_score,
)


def test_tactile_contact_score_uses_largest_sensor_norm():
    tactile = np.zeros((2, 15), dtype=np.float32)
    tactile[0, :6] = [3, 4, 0, 0, 0, 6]
    tactile[1, 9:12] = [1, 2, 2]

    np.testing.assert_allclose(tactile_contact_score(tactile), [6, 3])


def test_detect_contact_requires_sustained_episode_relative_response():
    scores = np.array([10] * 20 + [55, 12, 55, 56, 57, 20], dtype=np.float32)

    trigger, baseline, threshold = detect_contact_frame(scores)

    assert trigger == 22
    assert baseline == 10
    assert threshold == 50


def test_build_trim_plan_keeps_trigger_and_preserves_no_contact_episode():
    tactile = np.zeros((16, 6), dtype=np.float32)
    tactile[:8, 0] = [1, 1, 1, 1, 9, 10, 11, 12]
    tactile[8:, 0] = 2
    episode_ends = np.array([8, 16], dtype=np.int64)

    indices, new_episode_ends, trims = build_trim_plan(
        tactile,
        episode_ends,
        sensor_dims=6,
        baseline_frames=4,
        absolute_threshold=8,
        baseline_delta=5,
        consecutive_frames=3,
        search_start_frame=4,
    )

    np.testing.assert_array_equal(indices, [0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15])
    np.testing.assert_array_equal(new_episode_ends, [5, 13])
    assert trims[0].trigger_frame == 4
    assert trims[1].trigger_frame is None


def test_terminal_hold_actions_use_current_pose_and_gripper():
    tcp_pose = np.arange(5 * 3, dtype=np.float32).reshape(5, 3)
    gripper = np.arange(5, dtype=np.float32).reshape(5, 1) / 10
    data = {
        "action": np.full((5, 4), -1, dtype=np.float32),
        "left_robot_tcp_pose": tcp_pose,
        "left_robot_gripper_width": gripper,
    }

    indices, holds = terminal_hold_actions(data, np.array([2, 5]))

    np.testing.assert_array_equal(indices, [1, 4])
    np.testing.assert_array_equal(
        holds,
        np.stack(
            [
                np.concatenate([tcp_pose[1], gripper[1]]),
                np.concatenate([tcp_pose[4], gripper[4]]),
            ]
        ),
    )
