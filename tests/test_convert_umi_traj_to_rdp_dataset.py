import subprocess
import sys

import numpy as np
import pytest
import zarr

from scripts.convert_umi_traj_to_rdp_dataset import (
    action_from_pose_gripper,
    gripper_from_source_action,
    magnetic_time_mean,
    validate_magnet_output_keys,
)


def test_converter_exports_two_independent_magnet_inputs(tmp_path):
    source_path = tmp_path / "source.zarr"
    output_path = tmp_path / "output"
    root = zarr.open(str(source_path), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    n = 4
    data.create_dataset("camera0_rgb", data=np.zeros((n, 8, 8, 3), dtype=np.uint8))
    data.create_dataset("robot0_eef_pos", data=np.zeros((n, 3), dtype=np.float32))
    data.create_dataset(
        "robot0_eef_rot_axis_angle",
        data=np.zeros((n, 3), dtype=np.float32),
    )
    data.create_dataset("robot0_gripper_width", data=np.ones((n, 1), dtype=np.float32))
    data.create_dataset("action", data=np.zeros((n, 7), dtype=np.float32))
    data.create_dataset("magnet_xyz", data=np.full((n, 2, 4, 3), 2, dtype=np.float32))
    data.create_dataset("magnet_sample_count", data=np.full((n, 1), 2, dtype=np.int32))
    data.create_dataset("magnet2_xyz", data=np.full((n, 2, 4, 3), 7, dtype=np.float32))
    data.create_dataset("magnet2_sample_count", data=np.full((n, 1), 2, dtype=np.int32))
    meta.create_dataset("episode_ends", data=np.asarray([n], dtype=np.int64))

    subprocess.run(
        [
            sys.executable,
            "scripts/convert_umi_traj_to_rdp_dataset.py",
            "--input",
            str(source_path),
            "--output",
            str(output_path),
            "--preset",
            "rdp10d",
            "--image-size",
            "8",
            "8",
            "--action-source",
            "source",
            "--require-magnet2",
            "--validate",
        ],
        check=True,
    )

    converted = zarr.open(str(output_path / "replay_buffer.zarr"), mode="r")["data"]
    first = np.asarray(converted["left_gripper1_marker_offset_emb"][:])
    second = np.asarray(converted["left_gripper2_marker_offset_emb"][:])
    np.testing.assert_array_equal(first[:, :12], 2.0)
    np.testing.assert_array_equal(second[:, :12], 7.0)
    np.testing.assert_array_equal(first[:, 12:], 0.0)
    np.testing.assert_array_equal(second[:, 12:], 0.0)


def test_converter_splits_arpose_dual_board_magnet_input(tmp_path):
    source_path = tmp_path / "arpose_dual.zarr"
    output_path = tmp_path / "output"
    root = zarr.open(str(source_path), mode="w")
    root.attrs["magnet_board_order"] = ["right", "left"]
    data = root.create_group("data")
    meta = root.create_group("meta")
    n = 3
    data.create_dataset("camera0_rgb", data=np.zeros((n, 8, 8, 3), dtype=np.uint8))
    data.create_dataset("robot0_eef_pos", data=np.zeros((n, 3), dtype=np.float32))
    data.create_dataset(
        "robot0_eef_rot_axis_angle",
        data=np.zeros((n, 3), dtype=np.float32),
    )
    data.create_dataset("robot0_gripper_width", data=np.ones((n, 1), dtype=np.float32))
    data.create_dataset("action", data=np.zeros((n, 7), dtype=np.float32))
    magnet = np.empty((n, 2, 4, 3), dtype=np.float32)
    magnet[:, 0] = 2.0
    magnet[:, 1] = 7.0
    data.create_dataset("magnet_xyz", data=magnet)
    data.create_dataset("magnet_sample_count", data=np.ones((n, 2), dtype=np.int32))
    data.create_dataset(
        "magnet_timestamp_ns",
        data=np.tile(np.array([[100, 200]], dtype=np.int64), (n, 1)),
    )
    meta.create_dataset("episode_ends", data=np.asarray([n], dtype=np.int64))

    subprocess.run(
        [
            sys.executable,
            "scripts/convert_umi_traj_to_rdp_dataset.py",
            "--input",
            str(source_path),
            "--output",
            str(output_path),
            "--preset",
            "rdp10d",
            "--image-size",
            "8",
            "8",
            "--action-source",
            "source",
            "--require-magnet2",
            "--copy-magnet-timestamps",
            "--validate",
        ],
        check=True,
    )

    converted = zarr.open(str(output_path / "replay_buffer.zarr"), mode="r")["data"]
    first = np.asarray(converted["left_gripper1_marker_offset_emb"][:])
    second = np.asarray(converted["left_gripper2_marker_offset_emb"][:])
    np.testing.assert_array_equal(first[:, :12], np.full((n, 12), 2.0))
    np.testing.assert_array_equal(second[:, :12], np.full((n, 12), 7.0))
    np.testing.assert_array_equal(converted["magnet_timestamp_ns"][:], np.full(n, 100))
    np.testing.assert_array_equal(converted["magnet2_timestamp_ns"][:], np.full(n, 200))


def test_converter_rejects_duplicate_magnet_output_keys():
    args = type(
        "Args",
        (),
        {
            "magnet_tactile_key": "duplicate",
            "magnet_wrench_key": "wrench1",
            "magnet2_tactile_key": "duplicate",
            "magnet2_wrench_key": "wrench2",
        },
    )()

    with pytest.raises(ValueError, match="must be unique"):
        validate_magnet_output_keys(args, ["tactile"], ["tactile"])


def test_next_obs_tcp_keeps_current_7d_source_gripper_command():
    next_pos = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    next_rotvec = np.zeros((2, 3), dtype=np.float32)
    source_action = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.75],
        ],
        dtype=np.float32,
    )

    action = action_from_pose_gripper(
        next_pos,
        next_rotvec,
        gripper_from_source_action(source_action),
    )

    np.testing.assert_array_equal(action[:, :3], next_pos)
    np.testing.assert_array_equal(action[:, 6], source_action[:, 6])


def test_gripper_from_source_action_supports_10d_and_fallback_layouts():
    action10 = np.arange(20, dtype=np.float32).reshape(2, 10)
    action6 = np.zeros((2, 6), dtype=np.float32)
    fallback = np.array([0.2, 0.8], dtype=np.float32)

    np.testing.assert_array_equal(
        gripper_from_source_action(action10),
        action10[:, 9:10],
    )
    np.testing.assert_array_equal(
        gripper_from_source_action(action6, fallback),
        fallback[:, None],
    )


def test_magnetic_time_mean_uses_only_active_arpose_board_slot():
    magnet = np.zeros((3, 2, 4, 3), dtype=np.float32)
    magnet[0, 0] = 1.0
    magnet[1, 0] = 2.0
    magnet[:, 1] = 999.0
    counts = np.array([[1, 0], [1, 0], [0, 0]], dtype=np.int32)

    mean = magnetic_time_mean(magnet, sample_count=counts)

    assert mean.shape == (3, 4, 3)
    np.testing.assert_array_equal(mean[0], np.ones((4, 3), dtype=np.float32))
    np.testing.assert_array_equal(mean[1], np.full((4, 3), 2.0, dtype=np.float32))
    np.testing.assert_array_equal(mean[2], np.zeros((4, 3), dtype=np.float32))


def test_magnetic_time_mean_rejects_multiple_active_arpose_boards():
    magnet = np.ones((2, 2, 4, 3), dtype=np.float32)
    counts = np.ones((2, 2), dtype=np.int32)

    with pytest.raises(ValueError, match="exactly one active magnetic board slot"):
        magnetic_time_mean(magnet, sample_count=counts)


def test_magnetic_time_mean_handles_all_missing_arpose_chunk():
    magnet = np.full((2, 2, 4, 3), 999.0, dtype=np.float32)
    counts = np.zeros((2, 2), dtype=np.int32)

    mean = magnetic_time_mean(magnet, sample_count=counts)

    np.testing.assert_array_equal(mean, np.zeros((2, 4, 3), dtype=np.float32))


def test_magnetic_time_mean_preserves_standard_sample_count_layout():
    magnet = np.array(
        [
            [[[100.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]], [[4.0, 0.0, 0.0]]],
            [[[1.0, 0.0, 0.0]], [[3.0, 0.0, 0.0]], [[5.0, 0.0, 0.0]]],
        ],
        dtype=np.float32,
    )
    counts = np.array([[2], [3]], dtype=np.int32)

    mean = magnetic_time_mean(magnet, sample_count=counts)

    np.testing.assert_array_equal(mean[:, 0, 0], np.array([3.0, 3.0], dtype=np.float32))
