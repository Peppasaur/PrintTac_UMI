import csv

import numpy as np

from scripts.compare_iphone_robot_pose import (
    compare_pose_sequences,
    detect_movement_start,
    read_receiver_clock_offset_ms,
)


def test_pose_comparison_first_frame_alignment_and_relative_motion():
    iphone_pose = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.1],
            [0.2, 0.0, 0.0, 0.0, 0.0, 0.2],
        ]
    )
    robot_pose = iphone_pose.copy()
    robot_pose[:, :3] += [0.5, -0.2, 0.3]

    result = compare_pose_sequences(iphone_pose, robot_pose)

    np.testing.assert_allclose(result["position_error_mm"], 0.0, atol=1e-9)
    np.testing.assert_allclose(result["rotation_error_deg"], 0.0, atol=1e-9)
    np.testing.assert_allclose(
        result["relative_translation_error_mm"], 0.0, atol=1e-9
    )
    np.testing.assert_allclose(
        result["relative_rotation_error_deg"], 0.0, atol=1e-9
    )


def test_receiver_clock_offset_uses_video_median(tmp_path):
    path = tmp_path / "receiver_transport.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("kind", "clock_offset_ms"),
        )
        writer.writeheader()
        writer.writerows(
            [
                {"kind": "video", "clock_offset_ms": "310.0"},
                {"kind": "ultrawide_video", "clock_offset_ms": "900.0"},
                {"kind": "video", "clock_offset_ms": "312.0"},
            ]
        )

    result = read_receiver_clock_offset_ms(path)

    assert result["offset_ms"] == 311.0
    assert result["sample_count"] == 2


def test_movement_start_requires_sustained_departure_from_first_pose():
    timestamp_ns = np.arange(7, dtype=np.int64) * 10_000_000
    pose = np.zeros((7, 6), dtype=np.float64)
    pose[:, 0] = np.array([0.0, 0.0005, 0.003, 0.0002, 0.0021, 0.003, 0.004])

    result = detect_movement_start(
        timestamp_ns,
        pose,
        position_threshold_mm=2.0,
        rotation_threshold_deg=0.5,
        sustain_frames=3,
    )

    assert result["index"] == 4
    assert result["timestamp_ns"] == 40_000_000
