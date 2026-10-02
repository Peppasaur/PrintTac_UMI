import numpy as np

from scripts.visualize_rdp_dataset_magnet import (
    annotate_rgb_frames,
    episode_bounds,
    parse_episode_selection,
    rolling_magnet_windows,
)


def test_episode_selection_and_bounds():
    assert parse_episode_selection("0,2-3", 5) == [0, 2, 3]
    assert episode_bounds(np.array([4, 9, 15]), 0) == (0, 4)
    assert episode_bounds(np.array([4, 9, 15]), 2) == (9, 15)


def test_annotation_preserves_video_shape_and_marks_data():
    frames = np.full((3, 64, 80, 3), 180, dtype=np.uint8)
    annotated = annotate_rgb_frames(
        frames,
        np.array([0.14, 0.14, 0.11], dtype=np.float32),
        contact_frame=2,
        episode=7,
    )

    assert annotated.shape == frames.shape
    assert not np.array_equal(annotated, frames)


def test_rolling_magnet_windows_keep_recent_history():
    magnet = np.arange(3 * 2 * 3, dtype=np.float32).reshape(3, 2, 3)
    windows = rolling_magnet_windows(magnet, window_frames=2)

    assert windows.shape == (3, 2, 2, 3)
    np.testing.assert_array_equal(windows[0], [magnet[0], magnet[0]])
    np.testing.assert_array_equal(windows[2], [magnet[1], magnet[2]])
