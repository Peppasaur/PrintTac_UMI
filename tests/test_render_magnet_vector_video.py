import csv

import cv2
import numpy as np
import pytest

from scripts.render_magnet_vector_video import (
    first_finite_baseline,
    load_magnet_csv,
    render_vector_frame,
    render_vector_video,
)


def _write_trace(path, prefix="magnet", sensor_count=5):
    fields = ["elapsed"]
    for sensor in range(1, sensor_count + 1):
        fields.extend(f"{prefix}_s{sensor}_{axis}" for axis in ("x", "y", "z"))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for frame in range(2):
            row = {"elapsed": frame * 0.1}
            for sensor in range(1, sensor_count + 1):
                row.update(
                    {
                        f"{prefix}_s{sensor}_x": frame + sensor,
                        f"{prefix}_s{sensor}_y": -frame - sensor,
                        f"{prefix}_s{sensor}_z": 2 * sensor,
                    }
                )
            writer.writerow(row)


def test_loads_five_sensor_magnet_trace(tmp_path):
    path = tmp_path / "trace.csv"
    _write_trace(path)

    elapsed, values, prefix = load_magnet_csv(path, sensor_count=5)

    assert prefix == "magnet"
    np.testing.assert_allclose(elapsed, [0.0, 0.1])
    assert values.shape == (2, 5, 3)
    np.testing.assert_allclose(values[1, 4], [6.0, -6.0, 10.0])


def test_auto_prefers_normalized_columns(tmp_path):
    path = tmp_path / "trace.csv"
    _write_trace(path, prefix="normalized", sensor_count=1)

    elapsed, values, prefix = load_magnet_csv(path, sensor_count=1)

    assert prefix == "normalized"
    np.testing.assert_allclose(values[0, 0], [1.0, -1.0, 2.0])


def test_loads_arpose_streamer_zero_based_sensor_columns(tmp_path):
    path = tmp_path / "magnetic_right.csv"
    fields = ["sequence", "relative_time"]
    for sensor in range(5):
        fields.extend(f"s{sensor}_{axis}" for axis in ("t", "x", "y", "z"))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        row = {"sequence": 1, "relative_time": 0.25}
        for sensor in range(5):
            row.update(
                {
                    f"s{sensor}_t": 10,
                    f"s{sensor}_x": sensor + 1,
                    f"s{sensor}_y": sensor + 2,
                    f"s{sensor}_z": sensor + 3,
                }
            )
        writer.writerow(row)

    elapsed, values, prefix = load_magnet_csv(path, sensor_count=5)

    assert prefix == "s"
    np.testing.assert_allclose(elapsed, [0.25])
    np.testing.assert_allclose(values[0, 4], [5.0, 6.0, 7.0])


def test_first_finite_baseline_is_per_sensor_axis():
    values = np.array(
        [
            [[np.nan, 2.0, 3.0]],
            [[1.0, 4.0, 5.0]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(first_finite_baseline(values), [[1.0, 2.0, 3.0]])


def test_renders_vector_frame_for_five_sensors():
    values = np.arange(15, dtype=np.float32).reshape(1, 5, 3)
    frame = render_vector_frame(values, np.array([0.0]), 0, limit=20.0, width=800, height=480)

    assert frame.shape == (480, 800, 3)
    assert np.any(frame != 250)


def test_writes_vector_video(tmp_path):
    output = tmp_path / "vectors.mp4"
    elapsed = np.array([0.0, 0.1], dtype=np.float32)
    values = np.ones((2, 5, 3), dtype=np.float32)

    result, limit = render_vector_video(elapsed, values, output, fps=10.0, width=640, height=360)

    assert result == output
    assert limit >= 1.0
    capture = cv2.VideoCapture(str(output))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 2
    assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 640
    assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 360
    capture.release()


def test_rejects_missing_sensor_group(tmp_path):
    path = tmp_path / "trace.csv"
    _write_trace(path, sensor_count=4)

    with pytest.raises(KeyError, match="No complete"):
        load_magnet_csv(path, sensor_count=5)
