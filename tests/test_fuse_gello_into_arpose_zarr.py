import tempfile
from pathlib import Path

import numpy as np
import zarr

from scripts.fuse_gello_into_arpose_zarr import (
    EpisodeFusion,
    EpisodeMatch,
    align_causal_magnet_windows,
    build_compatible_magnetic_fields,
    calculate_trajectory_error,
    check_output_paths,
    flatten_magnet_samples,
    apply_clock_offset_ns,
    interpolate_gripper_values,
    interpolate_gripper_width,
    interpolate_pose6,
    match_episode_names,
    next_observation_indices,
    read_receiver_clock_offset_ms,
    summarize_trajectory_errors,
    validate_output,
    write_fused_zarr,
)


def test_episode_matching_uses_recording_time_not_list_position():
    matches, unmatched_arpose, unmatched_gello = match_episode_names(
        ["20260804-204139", "20260804-204247", "20260804-204321"],
        ["20260804_203950", "20260804_204247", "20260804_204321"],
        tolerance_sec=1.0,
    )

    assert [(match.arpose_name, match.gello_name) for match in matches] == [
        ("20260804-204247", "20260804_204247"),
        ("20260804-204321", "20260804_204321"),
    ]
    assert unmatched_arpose == ["20260804-204139"]
    assert unmatched_gello == ["20260804_203950"]


def test_receiver_clock_offset_uses_video_median_and_shifts_timestamps():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "receiver_transport.csv"
        path.write_text(
            "kind,clock_offset_ms\n"
            "video,250.0\n"
            "ultrawide_video,900.0\n"
            "video,252.0\n",
            encoding="utf-8",
        )

        result = read_receiver_clock_offset_ms(path)

    assert result["offset_ms"] == 251.0
    assert result["method"] == "clock_offset_ms"
    assert result["sample_count"] == 2
    np.testing.assert_array_equal(
        apply_clock_offset_ns(np.array([1_000_000_000]), result["offset_ms"]),
        [1_251_000_000],
    )


def test_receiver_clock_offset_recovers_legacy_latency_columns():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "receiver_transport.csv"
        path.write_text(
            "kind,raw_latency_ms,corrected_latency_ms,clock_offset_ms\n"
            "pose,57.0,5.0,\n"
            "pose,60.0,8.0,\n"
            "video,80.0,28.0,\n",
            encoding="utf-8",
        )

        result = read_receiver_clock_offset_ms(path)

    assert result["offset_ms"] == 52.0
    assert result["method"] == "raw_latency_ms_minus_corrected_latency_ms"
    assert result["sample_count"] == 3


def test_gripper_alignment_converts_raw_ratio_to_meter_width():
    width, report = interpolate_gripper_width(
        target_timestamp_ns=np.array([500_000_000, 1_500_000_000, 2_500_000_000]),
        source_timestamp_ns=np.array([1_000_000_000, 2_000_000_000]),
        source_gripper_raw=np.array([1.0, 0.5]),
        stroke_m=0.08,
    )

    np.testing.assert_allclose(width[:, 0], [0.08, 0.06, 0.04])
    assert report["frames_before_source"] == 1
    assert report["frames_after_source"] == 1


def test_gripper_alignment_can_preserve_raw_teleop_ratio():
    values, report = interpolate_gripper_values(
        target_timestamp_ns=np.array([1_000_000_000, 1_500_000_000, 2_000_000_000]),
        source_timestamp_ns=np.array([1_000_000_000, 2_000_000_000]),
        source_gripper_raw=np.array([1.0, 0.5]),
        value_mode="raw_ratio",
        stroke_m=0.08,
    )

    np.testing.assert_allclose(values[:, 0], [1.0, 0.75, 0.5])
    assert report["output_value_mode"] == "raw_ratio"
    assert report["output_unit"] == "open_ratio"


def test_pose_alignment_interpolates_position_and_rotation():
    pose, report = interpolate_pose6(
        target_timestamp_ns=np.array([-1, 1_000_000_000, 3_000_000_000]),
        source_timestamp_ns=np.array([0, 2_000_000_000]),
        source_pose6=np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [2.0, 4.0, 6.0, 0.0, 0.0, np.pi],
            ]
        ),
    )

    np.testing.assert_allclose(pose[:, :3], [[0, 0, 0], [1, 2, 3], [2, 4, 6]])
    np.testing.assert_allclose(pose[:, 3:6], [[0, 0, 0], [0, 0, np.pi / 2], [0, 0, np.pi]])
    assert report["frames_before_source"] == 1
    assert report["frames_after_source"] == 1


def test_trajectory_error_excludes_frames_outside_robot_time_range():
    iphone_pose = np.zeros((4, 6), dtype=np.float64)
    iphone_pose[:, 0] = [0.0, 0.1, 0.2, 0.3]
    robot_pose = iphone_pose.copy()
    robot_pose[:, :3] += [0.5, -0.2, 0.3]

    report, values = calculate_trajectory_error(
        iphone_pose,
        robot_pose,
        target_timestamp_ns=np.array([0, 10, 20, 30]),
        robot_timestamp_ns=np.array([10, 20, 30]),
    )

    assert report["overlap_frames"] == 3
    assert report["excluded_outside_robot_time_range"] == 1
    np.testing.assert_allclose(values["position_error_mm"], 0.0, atol=1e-9)
    np.testing.assert_allclose(
        values["relative_translation_error_mm"], 0.0, atol=1e-9
    )


def test_trajectory_error_summary_reports_equal_and_frame_weighted_means():
    metric_names = (
        "position_error_mm",
        "rotation_error_deg",
        "relative_translation_error_mm",
        "relative_rotation_error_deg",
    )

    def fusion(name, metric_values):
        return EpisodeFusion(
            match=EpisodeMatch(0, name, name.replace("-", "_"), 0.0),
            source_start=0,
            source_end=len(metric_values),
            gripper_value=np.zeros((len(metric_values), 1), dtype=np.float32),
            magnet_xyz=np.zeros((len(metric_values), 1, 1, 3), dtype=np.float32),
            magnet_timestamp_ns=np.zeros((len(metric_values), 1), dtype=np.int64),
            magnet_sample_count=np.zeros((len(metric_values), 1), dtype=np.int32),
            report={},
            trajectory_error_values={
                metric: np.asarray(metric_values, dtype=np.float64)
                for metric in metric_names
            },
        )

    summary = summarize_trajectory_errors(
        [
            fusion("20260804-100000", [1.0]),
            fusion("20260804-100100", [3.0, 3.0, 3.0]),
        ]
    )

    assert summary is not None
    for metric in metric_names:
        assert summary["episode_equal_mean"][metric]["mean"] == 2.0
        assert summary["frame_weighted"][metric]["mean"] == 2.5


def test_magnet_alignment_deduplicates_overlapping_frames_and_is_causal():
    magnet_xyz = np.array(
        [
            [[[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]], [[3.0, 0.0, 0.0]]],
            [[[2.0, 0.0, 0.0]], [[3.0, 0.0, 0.0]], [[4.0, 0.0, 0.0]]],
        ],
        dtype=np.float32,
    )
    magnet_timestamp_ns = np.array([[10, 20, 30], [20, 30, 40]], dtype=np.int64)
    source_time, source_values = flatten_magnet_samples(
        magnet_xyz,
        magnet_timestamp_ns,
        np.array([3, 3]),
    )
    np.testing.assert_array_equal(source_time, [10, 20, 30, 40])

    xyz, timestamps, counts, report = align_causal_magnet_windows(
        target_timestamp_ns=np.array([5, 25, 40]),
        source_timestamp_ns=source_time,
        source_magnet_xyz=source_values,
        samples_per_frame=3,
    )

    np.testing.assert_array_equal(counts[:, 0], [0, 2, 3])
    np.testing.assert_array_equal(timestamps[1], [0, 10, 20])
    np.testing.assert_array_equal(timestamps[2], [20, 30, 40])
    np.testing.assert_allclose(xyz[2, :, 0, 0], [2.0, 3.0, 4.0])
    assert np.all(timestamps <= np.array([5, 25, 40])[:, None])
    assert report["frames_without_past_sample"] == 1
    assert report["sample_count_min"] == 0
    assert report["sample_count_max"] == 3


def test_writer_selects_matched_episode_and_rebuilds_action_gripper():
    with tempfile.TemporaryDirectory() as directory:
        source_path = Path(directory) / "source.zarr"
        output_path = Path(directory) / "fused.zarr"
        source = zarr.group(store=zarr.DirectoryStore(str(source_path)), overwrite=True)
        source.attrs["source_directories"] = ["20260804-100000", "20260804-100100"]
        data = source.create_group("data")
        data.create_dataset(
            "camera0_rgb",
            data=np.zeros((5, 4, 4, 3), dtype=np.uint8),
            chunks=(1, 4, 4, 3),
        )
        data.create_dataset("timestamp", data=np.arange(5, dtype=np.float64), chunks=(5,))
        source_gripper = np.full((5, 1), 0.01, dtype=np.float32)
        data.create_dataset("robot0_gripper_width", data=source_gripper, chunks=(5, 1))
        action = np.zeros((5, 7), dtype=np.float32)
        action[:, 0] = np.arange(5)
        action[:, 6] = source_gripper[:, 0]
        data.create_dataset("action", data=action, chunks=(5, 7))
        data.create_dataset(
            "magnet_xyz",
            data=np.zeros((5, 2, 1, 3), dtype=np.float32),
            chunks=(5, 2, 1, 3),
        )
        data.create_dataset(
            "magnet_timestamp_ns",
            data=np.zeros((5, 2), dtype=np.int64),
            chunks=(5, 2),
        )
        data.create_dataset(
            "magnet_sample_count",
            data=np.zeros((5, 1), dtype=np.int32),
            chunks=(5, 1),
        )
        meta = source.create_group("meta")
        meta.create_dataset("episode_ends", data=np.array([2, 5]), chunks=(2,))

        gripper = np.array([[0.08], [0.06], [0.04]], dtype=np.float32)
        fusion = EpisodeFusion(
            match=EpisodeMatch(1, "20260804-100100", "20260804_100100", 0.0),
            source_start=2,
            source_end=5,
            gripper_value=gripper,
            magnet_xyz=np.ones((3, 3, 1, 3), dtype=np.float32),
            magnet_timestamp_ns=np.array(
                [[0, 0, 2_000_000_000], [0, 2_000_000_000, 3_000_000_000], [2_000_000_000, 3_000_000_000, 4_000_000_000]],
                dtype=np.int64,
            ),
            magnet_sample_count=np.array([[1], [2], [3]], dtype=np.int32),
            report={},
        )
        report = {
            "gripper_stroke_m": 0.08,
            "magnet_samples_per_frame": 3,
            "episodes": [{}],
            "unmatched_arpose": ["20260804-100000"],
        }
        write_fused_zarr(source, output_path, [fusion], report, overwrite=False)
        validate_output(source, output_path, [fusion])

        output = zarr.open(str(output_path), mode="r")
        np.testing.assert_array_equal(output["meta/episode_ends"][:], [3])
        np.testing.assert_array_equal(output["data/action"][:, 0], [2, 3, 4])
        np.testing.assert_allclose(output["data/action"][:, 6], [0.06, 0.04, 0.04])
        assert output["data/magnet_xyz"].shape == (3, 3, 1, 3)
        assert output.attrs["source_directories"] == ["20260804-100100"]


def test_writer_can_replace_phone_pose_and_action_with_gello_robot_data():
    with tempfile.TemporaryDirectory() as directory:
        source_path = Path(directory) / "source.zarr"
        output_path = Path(directory) / "fused.zarr"
        source = zarr.group(store=zarr.DirectoryStore(str(source_path)), overwrite=True)
        source.attrs.update(
            {
                "source_directories": ["20260804-100000"],
                "action_source": "next_obs",
                "eef_pose_source": "iphone",
            }
        )
        data = source.create_group("data")
        data.create_dataset(
            "camera0_rgb",
            data=np.zeros((3, 4, 4, 3), dtype=np.uint8),
            chunks=(1, 4, 4, 3),
        )
        data.create_dataset(
            "timestamp", data=np.arange(3, dtype=np.float64), chunks=(3,)
        )
        data.create_dataset(
            "robot0_eef_pos", data=np.full((3, 3), -1, dtype=np.float32), chunks=(3, 3)
        )
        data.create_dataset(
            "robot0_eef_rot_axis_angle",
            data=np.full((3, 3), -1, dtype=np.float32),
            chunks=(3, 3),
        )
        data.create_dataset(
            "robot0_demo_start_pose",
            data=np.full((3, 6), -1, dtype=np.float32),
            chunks=(3, 6),
        )
        data.create_dataset(
            "robot0_demo_end_pose",
            data=np.full((3, 6), -1, dtype=np.float32),
            chunks=(3, 6),
        )
        data.create_dataset(
            "robot0_gripper_width",
            data=np.zeros((3, 1), dtype=np.float32),
            chunks=(3, 1),
        )
        data.create_dataset(
            "action", data=np.full((3, 7), -1, dtype=np.float32), chunks=(3, 7)
        )
        data.create_dataset(
            "magnet_xyz",
            data=np.zeros((3, 2, 1, 3), dtype=np.float32),
            chunks=(3, 2, 1, 3),
        )
        data.create_dataset(
            "magnet_timestamp_ns",
            data=np.zeros((3, 2), dtype=np.int64),
            chunks=(3, 2),
        )
        data.create_dataset(
            "magnet_sample_count",
            data=np.zeros((3, 1), dtype=np.int32),
            chunks=(3, 1),
        )
        meta = source.create_group("meta")
        meta.create_dataset("episode_ends", data=np.array([3]), chunks=(1,))

        robot_pose = np.array(
            [
                [0.1, 0.2, 0.3, 0.0, 0.0, 0.1],
                [0.4, 0.5, 0.6, 0.0, 0.0, 0.2],
                [0.7, 0.8, 0.9, 0.0, 0.0, 0.3],
            ],
            dtype=np.float32,
        )
        gripper = np.array([[1.0], [0.75], [0.5]], dtype=np.float32)
        action = np.concatenate([robot_pose + 1.0, gripper], axis=1)
        fusion = EpisodeFusion(
            match=EpisodeMatch(0, "20260804-100000", "20260804_100000", 0.0),
            source_start=0,
            source_end=3,
            gripper_value=gripper,
            magnet_xyz=np.zeros((3, 2, 1, 3), dtype=np.float32),
            magnet_timestamp_ns=np.zeros((3, 2), dtype=np.int64),
            magnet_sample_count=np.zeros((3, 1), dtype=np.int32),
            report={},
            robot_pose=robot_pose,
            action=action,
        )
        report = {
            "robot_data_source": "gello",
            "gripper_value_mode": "raw_ratio",
            "gripper_stroke_m": 0.08,
            "magnet_samples_per_frame": 2,
            "episodes": [{}],
            "unmatched_arpose": [],
        }

        write_fused_zarr(source, output_path, [fusion], report, overwrite=False)
        validate_output(source, output_path, [fusion])

        output = zarr.open(str(output_path), mode="r")
        np.testing.assert_allclose(output["data/robot0_eef_pos"][:], robot_pose[:, :3])
        np.testing.assert_allclose(output["data/action"][:], action)
        np.testing.assert_allclose(
            output["data/robot0_demo_start_pose"][:],
            np.repeat(robot_pose[:1], 3, axis=0),
        )
        np.testing.assert_allclose(
            output["data/robot0_demo_end_pose"][:],
            np.repeat(robot_pose[-1:], 3, axis=0),
        )
        assert output.attrs["robot_data_source"] == "gello"
        assert output.attrs["gripper_value_mode"] == "raw_ratio"
        assert output.attrs["gripper_width_unit"] == "open_ratio"
        assert output.attrs["eef_pose_source"] == "gello_actual_tcp_timestamp_aligned"
        assert output.attrs["action_source"] == "gello_command_timestamp_aligned"


def test_next_observation_indices_stop_at_episode_boundaries():
    np.testing.assert_array_equal(
        next_observation_indices([2, 3]),
        np.array([1, 1, 3, 4, 4]),
    )


def test_compatible_magnetic_fields_use_latest_gello_sample():
    magnet = np.full((2, 3, 2, 3), np.nan, dtype=np.float32)
    magnet[0, -1] = [[1, 2, 3], [4, 5, 6]]
    fields = build_compatible_magnetic_fields(magnet, np.array([[1], [0]]))

    np.testing.assert_array_equal(
        fields["magnetic_txyz"][0, :2],
        [[0, 1, 2, 3], [0, 4, 5, 6]],
    )
    np.testing.assert_array_equal(fields["magnetic_valid"][0], [True, True, False, False, False])
    assert not np.any(fields["magnetic_valid"][1])
    assert not np.any(fields["magnetic_left_valid"])


def test_output_preflight_reports_all_existing_paths():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        output = root / "output.zarr"
        report = root / "report.json"
        video = root / "videos"
        output.mkdir()
        report.write_text("{}", encoding="utf-8")

        try:
            check_output_paths(output, report, video, write_video=False, overwrite=False)
        except FileExistsError as exc:
            assert str(output) in str(exc)
            assert str(report) in str(exc)
        else:
            raise AssertionError("Existing output paths should fail preflight")

        check_output_paths(output, report, video, write_video=True, overwrite=True)
