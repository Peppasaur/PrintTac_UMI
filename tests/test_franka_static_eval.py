import csv
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from reactive_diffusion_policy.env.franka_polymetis.franka_polymetis_env import (
    FrankaPolymetisEnv,
)


class StaticEvalTest(unittest.TestCase):
    def test_generic_static_policy_eval_disables_every_motion_source(self):
        script = (Path(__file__).parents[1] / "eval_static_policy.sh").read_text()

        self.assertIn("export OPEN_GRIPPER_ON_START=False", script)
        self.assertIn('export START_GRIPPER_WIDTH_MM=""', script)
        self.assertIn('export FIXED_GRIPPER_WIDTH_MM=""', script)
        self.assertIn("export MOVE_TO_START=False", script)
        self.assertIn("export IGNORE_GRIPPER_COMMANDS=True", script)
        self.assertIn("export IGNORE_POLICY_TCP_COMMANDS=True", script)
        self.assertIn("export IGNORE_POLICY_GRIPPER_COMMANDS=True", script)
        self.assertIn('exec bash "${SCRIPT_DIR}/eval.sh"', script)

    def test_static_eval_script_disables_all_robot_motion(self):
        script = (Path(__file__).parents[1] / "eval_static_10s.sh").read_text()

        self.assertIn("export OPEN_GRIPPER_ON_START=False", script)
        self.assertIn("export MOVE_TO_START=False", script)
        self.assertIn("export IGNORE_GRIPPER_COMMANDS=True", script)
        self.assertIn("export IGNORE_POLICY_TCP_COMMANDS=True", script)
        self.assertIn("export IGNORE_POLICY_GRIPPER_COMMANDS=True", script)

    def test_policy_action_is_recorded_without_sending_tcp_command(self):
        env = FrankaPolymetisEnv.__new__(FrankaPolymetisEnv)
        env.ignore_policy_tcp_commands = True
        env.send_gripper_command = Mock()
        env._post_tcp_pose = Mock()
        action = np.arange(8, dtype=np.float32)

        env.execute_action(action)

        np.testing.assert_array_equal(env.last_policy_action_command, action)
        env.send_gripper_command.assert_called_once_with(6.0, 6.0)
        env._post_tcp_pose.assert_not_called()

    def test_policy_recording_frame_renders_gripper_observation_panel(self):
        env = FrankaPolymetisEnv.__new__(FrankaPolymetisEnv)
        env.policy_recording_plot_window_sec = 10.0
        env.policy_recording_plot_width = 360
        env.policy_recording_image_width = None
        env.policy_recording_image_height = None
        env.gripper_stroke = 0.085
        env.gripper_raw_min = 0.0
        env.gripper_raw_max = 1.0
        env.gripper_raw_range_mode = "dataset_observed_range"
        env.magnet_used_sensor_count = 4
        env.magnet_tactile_dim = 15

        image = np.full((240, 320, 3), 120, dtype=np.uint8)
        records = [
            {
                "elapsed": 0.0,
                "magnet_mean": np.zeros((4, 3), dtype=np.float32),
                "normalized_tactile_emb": np.zeros((15,), dtype=np.float32),
                "gripper_obs": np.asarray([1.0], dtype=np.float32),
            },
            {
                "elapsed": 0.5,
                "magnet_mean": np.ones((4, 3), dtype=np.float32),
                "normalized_tactile_emb": np.ones((15,), dtype=np.float32),
                "gripper_obs": np.asarray([0.5], dtype=np.float32),
            },
        ]

        frame = env._render_policy_recording_frame(image, records)

        self.assertEqual(frame.shape[0], 240)
        self.assertEqual(frame.shape[1], 320 + max(620, 360))
        self.assertEqual(frame.shape[2], 3)
        self.assertTrue(np.isfinite(frame).all())

    def test_dual_magnet_visualization_shows_both_raw_inputs_without_norm(self):
        env = FrankaPolymetisEnv.__new__(FrankaPolymetisEnv)
        env.policy_recording_plot_window_sec = 10.0
        env.policy_recording_plot_width = 620
        env.policy_recording_image_width = None
        env.policy_recording_image_height = None
        env.gripper_stroke = 0.085
        env.gripper_raw_min = 0.0
        env.gripper_raw_max = 1.0
        env.gripper_raw_range_mode = "dataset_observed_range"
        env.magnet_used_sensor_count = 4
        env.magnet2_used_sensor_count = 4
        env.magnet_tactile_dim = 15

        image = np.full((240, 320, 3), 120, dtype=np.uint8)
        records = [
            {
                "elapsed": float(step),
                "magnet_mean": np.full((4, 3), step, dtype=np.float32),
                "magnet2_mean": np.full((4, 3), step * 10, dtype=np.float32),
                "normalized_tactile_emb": np.full((15,), 999, dtype=np.float32),
                "predicted_normalized_tactile_emb": np.full((2, 15), 999, dtype=np.float32),
                "gripper_obs": np.asarray([0.5], dtype=np.float32),
            }
            for step in range(2)
        ]

        original_put_text = cv2.putText
        with patch.object(cv2, "putText", wraps=original_put_text) as put_text:
            frame = env._render_policy_recording_frame(image, records)

        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertIn("Magnet 1 delta", labels)
        self.assertIn("Magnet 2 delta", labels)
        self.assertNotIn("Policy norm", labels)
        self.assertFalse(any(label.startswith("nX=") for label in labels))
        self.assertFalse(any(label.startswith("pX=") for label in labels))
        self.assertEqual(frame.shape, (240, 940, 3))
        self.assertTrue(np.isfinite(frame).all())

    def test_dual_magnet_trace_saves_both_raw_inputs(self):
        env = FrankaPolymetisEnv.__new__(FrankaPolymetisEnv)
        env.magnet_used_sensor_count = 4
        env.magnet2_used_sensor_count = 4
        env.magnet_tactile_dim = 15
        env.magnet2_tactile_dim = 15
        records = []
        for step in range(2):
            magnet_mean = np.full((4, 3), step + 1, dtype=np.float32)
            magnet2_mean = np.full((4, 3), (step + 1) * 10, dtype=np.float32)
            records.append(
                {
                    "timestamp": float(step),
                    "elapsed": float(step),
                    "magnet_sample_count": 8,
                    "magnet_mean": magnet_mean,
                    "magnet_magnitude": np.linalg.norm(magnet_mean, axis=1),
                    "tactile_emb": np.zeros((15,), dtype=np.float32),
                    "normalized_tactile_emb": np.zeros((15,), dtype=np.float32),
                    "magnet2_sample_count": 8,
                    "magnet2_mean": magnet2_mean,
                    "magnet2_magnitude": np.linalg.norm(magnet2_mean, axis=1),
                    "tactile2_emb": np.ones((15,), dtype=np.float32),
                    "tcp_pose_obs": np.zeros((9,), dtype=np.float32),
                    "gripper_obs": np.zeros((1,), dtype=np.float32),
                    "last_action_command": np.zeros((16,), dtype=np.float32),
                }
            )

        with tempfile.TemporaryDirectory() as directory:
            npz_path = Path(directory) / "trace.npz"
            csv_path = Path(directory) / "trace.csv"
            env._save_policy_recording_trace(records, npz_path, csv_path)

            with np.load(npz_path) as trace:
                self.assertIn("magnet_mean", trace.files)
                self.assertIn("magnet2_mean", trace.files)
                np.testing.assert_array_equal(trace["magnet_mean"][-1], 2.0)
                np.testing.assert_array_equal(trace["magnet2_mean"][-1], 20.0)
            with csv_path.open(newline="") as csv_file:
                header = next(csv.reader(csv_file))
            self.assertIn("magnet_s1_x", header)
            self.assertIn("magnet2_s1_x", header)
            self.assertIn("tactile2_emb_0", header)

    def test_policy_recording_frame_records_both_magnet_inputs(self):
        env = FrankaPolymetisEnv.__new__(FrankaPolymetisEnv)
        env.policy_recording_lock = threading.Lock()
        env.policy_recording_active = True
        env.policy_recording_started_at = 0.0
        env.policy_recording_records = []
        env.policy_recording_writer = Mock()
        env.policy_recording_magnet_normalizer = None
        env.magnet_tactile_key = "left_gripper1_marker_offset_emb"
        env.magnet2_tactile_key = "left_gripper2_marker_offset_emb"
        env.magnet_tactile_dim = 15
        env.magnet2_tactile_dim = 15
        env.magnet_used_sensor_count = 4
        env.magnet2_used_sensor_count = 4
        env.predicted_magnet_lock = threading.Lock()
        env.latest_predicted_tactile_emb = None
        env.latest_predicted_normalized_tactile_emb = None
        env.last_policy_action_command = np.zeros((16,), dtype=np.float32)
        env._render_policy_recording_frame = Mock(
            return_value=np.zeros((8, 8, 3), dtype=np.uint8)
        )
        obs = {
            "timestamp": np.asarray([1.0], dtype=np.float64),
            "left_gripper1_marker_offset_emb": np.ones((15,), dtype=np.float32),
            "left_gripper2_marker_offset_emb": np.full((15,), 2, dtype=np.float32),
            "magnet_sample_count": np.asarray([8], dtype=np.int32),
            "magnet2_sample_count": np.asarray([7], dtype=np.int32),
            "left_robot_tcp_pose": np.zeros((9,), dtype=np.float32),
            "left_robot_gripper_width": np.asarray([0.5], dtype=np.float32),
        }

        env._record_policy_frame(
            np.zeros((8, 8, 3), dtype=np.uint8),
            obs,
            {
                0: np.ones((4, 3), dtype=np.float32),
                1: np.full((4, 3), 2, dtype=np.float32),
            },
        )

        self.assertEqual(len(env.policy_recording_records), 1)
        record = env.policy_recording_records[0]
        self.assertEqual(record["magnet_sample_count"], 8)
        self.assertEqual(record["magnet2_sample_count"], 7)
        np.testing.assert_array_equal(record["magnet_mean"], 1.0)
        np.testing.assert_array_equal(record["magnet2_mean"], 2.0)
        np.testing.assert_array_equal(record["tactile_emb"], 1.0)
        np.testing.assert_array_equal(record["tactile2_emb"], 2.0)
        env.policy_recording_writer.write.assert_called_once()


if __name__ == "__main__":
    unittest.main()
