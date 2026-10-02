import threading
import unittest

import numpy as np

from reactive_diffusion_policy.env.franka_polymetis.franka_polymetis_env import (
    FrankaPolymetisEnv,
    _magnet_to_tactile_embedding,
    _parse_magnet_sensor_order,
    _remap_magnet_sensors,
)


class _Reader:
    def __init__(self, value):
        self.value = float(value)

    def get_recent_samples(self):
        return {
            "magnet_xyz": np.full((2, 4, 3), self.value, dtype=np.float32),
            "magnet_timestamp_ns": np.full(2, int(self.value), dtype=np.int64),
            "magnet_sample_count": np.asarray([2], dtype=np.int32),
        }


class MagnetSensorOrderTest(unittest.TestCase):
    @staticmethod
    def _dual_input_env():
        env = FrankaPolymetisEnv.__new__(FrankaPolymetisEnv)
        for input_index, value in enumerate((1.0, 2.0)):
            prefix = "magnet" if input_index == 0 else "magnet2"
            setattr(env, f"{prefix}_reader", _Reader(value))
            setattr(env, f"{prefix}_sensor_order", np.arange(4, dtype=np.int64))
            setattr(env, f"{prefix}_zero_channels", np.asarray([], dtype=np.int64))
            setattr(env, f"{prefix}_tactile_dim", 15)
            setattr(
                env,
                f"{prefix}_tactile_key",
                f"left_gripper{input_index + 1}_marker_offset_emb",
            )
            setattr(env, f"{prefix}_filter_abnormal_readings", False)
            setattr(env, f"{prefix}_abnormal_abs_threshold", 5000.0)
            setattr(env, f"{prefix}_rezero_lock", threading.Lock())
            setattr(env, f"{prefix}_rezero_pending", False)
            setattr(env, f"{prefix}_rezero_deadline", None)
            setattr(env, f"{prefix}_rezero_baseline", None)
            setattr(env, f"{prefix}_normalize_to_first_frame", False)
            setattr(env, f"{prefix}_baseline_lock", threading.Lock())
            setattr(env, f"{prefix}_baseline", None)
        return env

    def test_samples_two_independent_magnetometer_inputs(self):
        env = self._dual_input_env()

        first = env._sample_magnet_input(0)
        second = env._sample_magnet_input(1)

        self.assertEqual(first["tactile_key"], "left_gripper1_marker_offset_emb")
        self.assertEqual(second["tactile_key"], "left_gripper2_marker_offset_emb")
        np.testing.assert_array_equal(first["tactile_emb"][:12], 1.0)
        np.testing.assert_array_equal(second["tactile_emb"][:12], 2.0)
        np.testing.assert_array_equal(first["tactile_emb"][12:], 0.0)
        np.testing.assert_array_equal(second["tactile_emb"][12:], 0.0)
        self.assertEqual(first["sample_count_key"], "magnet_sample_count")
        self.assertEqual(second["sample_count_key"], "magnet2_sample_count")
        self.assertEqual(first["timestamp_key"], "magnet_timestamp_ns")
        self.assertEqual(second["timestamp_key"], "magnet2_timestamp_ns")
        self.assertEqual(env._magnet_input_indices(), [0, 1])

    def test_remaps_live_magnet_sensors_to_policy_order(self):
        magnet_xyz = np.zeros((2, 4, 3), dtype=np.float32)
        for sensor_idx in range(4):
            magnet_xyz[:, sensor_idx, :] = sensor_idx + 1

        sensor_order = _parse_magnet_sensor_order(
            [2, 1, 4, 3],
            used_sensor_count=4,
        )
        remapped = _remap_magnet_sensors(magnet_xyz, sensor_order)

        np.testing.assert_array_equal(
            remapped[:, :, 0],
            np.asarray([[2, 1, 4, 3], [2, 1, 4, 3]], dtype=np.float32),
        )
        np.testing.assert_array_equal(magnet_xyz[0, :, 0], [1, 2, 3, 4])
        tactile_emb = _magnet_to_tactile_embedding(
            remapped[None, ...],
            sample_count=np.asarray([2], dtype=np.int32),
            output_dim=15,
        )[0]
        np.testing.assert_array_equal(
            tactile_emb,
            [2, 2, 2, 1, 1, 1, 4, 4, 4, 3, 3, 3, 0, 0, 0],
        )

    def test_rejects_invalid_magnet_sensor_order(self):
        invalid_orders = (
            [2, 1, 3],
            [2, 1, 4, 4],
            [2, 1, 4, 5],
        )
        for sensor_order in invalid_orders:
            with self.subTest(sensor_order=sensor_order):
                with self.assertRaisesRegex(ValueError, "each used sensor exactly once"):
                    _parse_magnet_sensor_order(sensor_order, used_sensor_count=4)


if __name__ == "__main__":
    unittest.main()
