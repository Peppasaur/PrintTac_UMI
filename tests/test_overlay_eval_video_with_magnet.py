from unittest.mock import patch

import cv2
import numpy as np
import pytest

from scripts.overlay_eval_video_with_magnet import (
    load_magnet_offset_file,
    render_full_trace_plot,
    save_full_trace_plot,
)


def test_render_full_trace_plot_combines_all_sensors():
    elapsed = np.linspace(0.0, 5.0, 11, dtype=np.float32)
    magnet = np.zeros((11, 4, 3), dtype=np.float32)
    for sensor_idx in range(4):
        magnet[:, sensor_idx, 0] = elapsed * (sensor_idx + 1)
        magnet[:, sensor_idx, 1] = -elapsed * (sensor_idx + 1)
        magnet[:, sensor_idx, 2] = np.sin(elapsed) * (sensor_idx + 1)

    image = render_full_trace_plot(elapsed, magnet, raw_limit=25.0)

    assert image.shape == (418, 1450, 3)
    assert image.dtype == np.uint8
    assert np.any(image != 255)
    # Axes and the S1-S4 legend should render non-white pixels.
    assert np.any(image[20:110, 1050:] != 255)
    assert np.any(image[100:350, :1400] != 255)


def test_save_full_trace_plot_creates_parent_and_image(tmp_path):
    elapsed = np.array([0.0, 1.0], dtype=np.float32)
    magnet = np.array(
        [
            [[[0.0, 0.0, 0.0]]],
            [[[1.0, -1.0, 2.0]]],
        ],
        dtype=np.float32,
    ).reshape(2, 1, 3)
    output = tmp_path / "nested" / "magnet_overview.png"

    result = save_full_trace_plot(output, elapsed, magnet, raw_limit=2.0)

    assert result == output
    saved = cv2.imread(str(output))
    assert saved is not None
    assert saved.shape == (418, 1450, 3)


def test_render_full_trace_plot_rejects_empty_sensor_axis():
    with pytest.raises(ValueError, match="at least one sensor"):
        render_full_trace_plot(
            np.array([0.0], dtype=np.float32),
            np.empty((1, 0, 3), dtype=np.float32),
            raw_limit=1.0,
        )


def test_render_full_trace_plot_accepts_manual_y_range():
    elapsed = np.array([0.0, 1.0], dtype=np.float32)
    magnet = np.array(
        [
            [[0.0, 0.0, 0.0]],
            [[100.0, 200.0, 300.0]],
        ],
        dtype=np.float32,
    )

    image = render_full_trace_plot(
        elapsed,
        magnet,
        raw_limit=1.0,
        magnet_y_range=(-250.0, 750.0),
    )

    assert image.shape == (418, 1450, 3)


def test_render_full_trace_plot_uses_manual_y_tick_interval():
    import matplotlib.ticker

    elapsed = np.array([0.0, 1.0], dtype=np.float32)
    magnet = np.zeros((2, 1, 3), dtype=np.float32)
    original_locator = matplotlib.ticker.MultipleLocator

    with patch.object(
        matplotlib.ticker,
        "MultipleLocator",
        wraps=original_locator,
    ) as locator:
        image = render_full_trace_plot(
            elapsed,
            magnet,
            raw_limit=500.0,
            magnet_y_range=(-500.0, 500.0),
            magnet_y_tick=125.0,
        )

    locator.assert_called_once_with(125.0)
    assert image.shape == (418, 1450, 3)


@pytest.mark.parametrize("magnet_y_tick", [0.0, -100.0, float("nan"), float("inf")])
def test_render_full_trace_plot_rejects_invalid_y_tick_interval(magnet_y_tick):
    with pytest.raises(ValueError, match="must be positive and finite"):
        render_full_trace_plot(
            np.array([0.0], dtype=np.float32),
            np.zeros((1, 1, 3), dtype=np.float32),
            raw_limit=1.0,
            magnet_y_tick=magnet_y_tick,
        )


def test_render_full_trace_plot_accepts_manual_x_range():
    elapsed = np.array([0.0, 5.0, 10.0], dtype=np.float32)
    magnet = np.zeros((3, 1, 3), dtype=np.float32)

    image = render_full_trace_plot(
        elapsed,
        magnet,
        raw_limit=1.0,
        magnet_x_range=(2.0, 8.0),
    )

    assert image.shape == (418, 1450, 3)


@pytest.mark.parametrize(
    "magnet_y_range, error",
    [
        ((1.0, 1.0), "Y_MIN < Y_MAX"),
        ((2.0, -1.0), "Y_MIN < Y_MAX"),
        ((float("nan"), 1.0), "must be finite"),
    ],
)
def test_render_full_trace_plot_rejects_invalid_manual_y_range(
    magnet_y_range,
    error,
):
    with pytest.raises(ValueError, match=error):
        render_full_trace_plot(
            np.array([0.0], dtype=np.float32),
            np.zeros((1, 1, 3), dtype=np.float32),
            raw_limit=1.0,
            magnet_y_range=magnet_y_range,
        )


@pytest.mark.parametrize(
    "magnet_x_range, error",
    [
        ((1.0, 1.0), "X_MIN < X_MAX"),
        ((2.0, -1.0), "X_MIN < X_MAX"),
        ((0.0, float("inf")), "must be finite"),
    ],
)
def test_render_full_trace_plot_rejects_invalid_manual_x_range(
    magnet_x_range,
    error,
):
    with pytest.raises(ValueError, match=error):
        render_full_trace_plot(
            np.array([0.0], dtype=np.float32),
            np.zeros((1, 1, 3), dtype=np.float32),
            raw_limit=1.0,
            magnet_x_range=magnet_x_range,
        )


def test_loads_normalized_offset_text_file(tmp_path):
    offset_file = tmp_path / "last_normalized.txt"
    offset_file.write_text(
        "Final normalized magnet change\n"
        "S1: x=1.5 y=-2 z=3\n"
        "S2: x=4 y=5.25 z=-6\n"
    )

    offset = load_magnet_offset_file(offset_file, sensor_count=2)

    np.testing.assert_allclose(offset, [[1.5, -2.0, 3.0], [4.0, 5.25, -6.0]])


def test_loads_normalized_offset_from_trace_csv(tmp_path):
    offset_file = tmp_path / "offset.csv"
    offset_file.write_text(
        "normalized_s1_x,normalized_s1_y,normalized_s1_z,"
        "normalized_s2_x,normalized_s2_y,normalized_s2_z\n"
        "1,2,3,4,5,6\n"
    )

    offset = load_magnet_offset_file(offset_file, sensor_count=2)

    np.testing.assert_allclose(offset, [[1, 2, 3], [4, 5, 6]])


def test_full_trace_plot_applies_normalized_offset_after_baseline():
    elapsed = np.array([0.0, 1.0], dtype=np.float32)
    magnet = np.zeros((2, 1, 3), dtype=np.float32)
    magnet[1, 0] = [1.0, 2.0, -3.0]
    image = render_full_trace_plot(
        elapsed,
        magnet,
        raw_limit=10.0,
        magnet_offset=np.array([[5.0, 6.0, 7.0]], dtype=np.float32),
    )

    assert image.shape == (418, 1450, 3)
