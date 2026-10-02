import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from scripts.pack_gello_polymetis_raw_to_zarr import (
    add_optional_secondary_magnet_data,
    draw_magnet_inputs_panel,
    save_episode_video,
    validate_secondary_magnet_episode_consistency,
)


def _frame(value):
    return {
        "magnet2_xyz": np.full((2, 4, 3), value, dtype=np.float32),
        "magnet2_timestamp_ns": np.full(2, value, dtype=np.int64),
        "magnet2_sample_count": np.asarray([2], dtype=np.int32),
    }


class PackDualMagnetTest(unittest.TestCase):
    def test_dual_magnet_panel_shows_both_raw_inputs_without_norm(self):
        magnet1 = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
        magnet2 = magnet1 * 10

        original_put_text = cv2.putText
        with patch.object(cv2, "putText", wraps=original_put_text) as put_text:
            panel = draw_magnet_inputs_panel(
                magnet_frame=magnet1,
                panel_height=240,
                panel_width=620,
                value_limit=25.0,
                magnet2_frame=magnet2,
                magnet2_value_limit=250.0,
            )

        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertIn("Magnet 1 delta", labels)
        self.assertIn("Magnet 2 delta", labels)
        self.assertFalse(any("norm" in label.lower() for label in labels))
        self.assertEqual(panel.shape, (240, 620, 3))
        self.assertTrue(np.isfinite(panel).all())

    def test_episode_video_automatically_receives_second_magnet_input(self):
        magnet2_xyz = np.zeros((2, 2, 4, 3), dtype=np.float32)
        episode_data = {
            "camera0_rgb": np.zeros((2, 224, 224, 3), dtype=np.uint8),
            "magnet_xyz": np.zeros((2, 2, 4, 3), dtype=np.float32),
            "magnet2_xyz": magnet2_xyz,
            "timestamp": np.asarray([0.0, 0.04], dtype=np.float64),
        }

        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "scripts.pack_gello_polymetis_raw_to_zarr.save_rgb_magnet_video"
            ) as save_video:
                save_episode_video(
                    episode_data=episode_data,
                    episode_dir=Path(directory) / "episode_000",
                    video_output_dir=Path(directory) / "videos",
                    fallback_fps=25.0,
                    panel_width=420,
                )

        self.assertIs(save_video.call_args.kwargs["magnet2_xyz"], magnet2_xyz)

    def test_stacks_secondary_magnet_fields(self):
        data = {}
        add_optional_secondary_magnet_data(data, [_frame(1), _frame(2)])

        self.assertEqual(data["magnet2_xyz"].shape, (2, 2, 4, 3))
        self.assertEqual(data["magnet2_timestamp_ns"].shape, (2, 2))
        self.assertEqual(data["magnet2_sample_count"].shape, (2, 1))
        np.testing.assert_array_equal(data["magnet2_xyz"][0], 1.0)
        np.testing.assert_array_equal(data["magnet2_xyz"][1], 2.0)

    def test_rejects_incomplete_secondary_magnet_fields(self):
        frames = [_frame(1), _frame(2)]
        del frames[1]["magnet2_timestamp_ns"]

        with self.assertRaisesRegex(KeyError, "missing from frame"):
            add_optional_secondary_magnet_data({}, frames)

    def test_rejects_mixed_single_and_dual_magnet_episodes(self):
        dual_episode = {key: np.zeros((2, 1)) for key in (
            "magnet2_xyz",
            "magnet2_timestamp_ns",
            "magnet2_sample_count",
        )}

        with self.assertRaisesRegex(ValueError, "Cannot mix"):
            validate_secondary_magnet_episode_consistency([{}, dual_episode])


if __name__ == "__main__":
    unittest.main()
