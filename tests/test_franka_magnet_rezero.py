import threading
from collections import deque
from unittest.mock import patch

import numpy as np

from reactive_diffusion_policy.env.franka_polymetis.franka_polymetis_env import FrankaPolymetisEnv


def _make_env():
    env = FrankaPolymetisEnv.__new__(FrankaPolymetisEnv)
    env.magnet_reader = object()
    env.magnet_rezero_after_policy_start_sec = 1.0
    env.magnet_rezero_lock = threading.Lock()
    env.magnet_rezero_deadline = None
    env.magnet_rezero_pending = False
    env.magnet_rezero_baseline = None
    env.magnet_baseline_lock = threading.Lock()
    env.magnet_baseline = np.ones((4, 3), dtype=np.float32)
    env.lock = threading.Lock()
    env.obs_buffer = deque([{"stale": True}])
    return env


def _magnet_frame(sensor_values):
    frame = np.zeros((2, 4, 3), dtype=np.float32)
    for sensor_idx, value in enumerate(sensor_values):
        frame[:, sensor_idx, :] = value
    return frame


def test_rezero_happens_once_one_second_after_policy_start_for_all_sensors():
    env = _make_env()
    baseline_frame = _magnet_frame([10.0, 20.0, 30.0, 40.0])
    later_frame = _magnet_frame([11.0, 22.0, 33.0, 44.0])
    sample_count = np.asarray([2], dtype=np.int32)

    with patch(
            "reactive_diffusion_policy.env.franka_polymetis.franka_polymetis_env.time.monotonic",
            side_effect=[100.0, 100.5, 101.0, 102.0]):
        env.notify_policy_started()
        before = env._apply_scheduled_magnet_rezero(baseline_frame, sample_count)
        at_deadline = env._apply_scheduled_magnet_rezero(baseline_frame, sample_count)
        after = env._apply_scheduled_magnet_rezero(later_frame, sample_count)

    np.testing.assert_array_equal(before, np.zeros_like(baseline_frame))
    np.testing.assert_array_equal(at_deadline, np.zeros_like(baseline_frame))
    np.testing.assert_array_equal(
        after,
        _magnet_frame([1.0, 2.0, 3.0, 4.0]),
    )
    assert env.magnet_rezero_pending is False
    assert env.magnet_baseline is None
    assert len(env.obs_buffer) == 0
