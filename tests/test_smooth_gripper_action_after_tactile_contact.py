import tempfile
from pathlib import Path

import numpy as np
import zarr

from scripts.smooth_gripper_action_after_tactile_contact import (
    build_gripper_smoothing_plan,
    validate_output,
    write_smoothed_dataset,
)


def test_smoothing_holds_open_before_contact_and_contact_width_afterward():
    action = np.array(
        [
            [1.0, 0.140],
            [2.0, 0.140],
            [3.0, 0.140],
            [4.0, 0.140],
            [5.0, 0.140],
            [6.0, 0.110],
            [7.0, 0.109],
            [8.0, 0.108],
            [9.0, 0.107],
            [10.0, 0.106],
        ],
        dtype=np.float32,
    )
    tactile = np.zeros((10, 6), dtype=np.float32)
    tactile[5:, 0] = 10

    smoothed, plans = build_gripper_smoothing_plan(
        action,
        tactile,
        np.array([10]),
        sensor_dims=6,
        baseline_frames=3,
        absolute_threshold=8,
        baseline_delta=5,
        consecutive_frames=3,
        search_start_frame=3,
        open_reference_quantile=0.9,
        contact_reference_frames=3,
    )

    np.testing.assert_array_equal(smoothed[:, 0], action[:, 0])
    np.testing.assert_allclose(smoothed[:5, -1], 0.140)
    np.testing.assert_allclose(smoothed[5:, -1], 0.109)
    assert plans[0].contact_frame == 5
    np.testing.assert_allclose(plans[0].open_reference, 0.14)
    np.testing.assert_allclose(plans[0].contact_reference, 0.109)


def test_smoothing_preserves_no_contact_episode_without_modification():
    action = np.array(
        [[1.0, 0.14], [2.0, 0.13], [3.0, 0.12], [4.0, 0.11]], dtype=np.float32
    )
    tactile = np.zeros((4, 6), dtype=np.float32)

    smoothed, plans = build_gripper_smoothing_plan(
        action,
        tactile,
        np.array([4]),
        sensor_dims=6,
        baseline_frames=2,
        absolute_threshold=8,
        baseline_delta=5,
        consecutive_frames=2,
        search_start_frame=2,
    )

    np.testing.assert_array_equal(smoothed, action)
    assert plans[0].contact_frame is None
    assert plans[0].open_reference is None
    assert plans[0].contact_reference is None


def test_smoothing_only_changes_the_final_action_dimension_per_episode():
    action = np.array(
        [
            [1.0, 2.0, 0.14],
            [3.0, 4.0, 0.14],
            [5.0, 6.0, 0.11],
            [7.0, 8.0, 0.10],
            [9.0, 10.0, 0.09],
            [11.0, 12.0, 0.08],
            [13.0, 14.0, 0.07],
            [15.0, 16.0, 0.06],
        ],
        dtype=np.float32,
    )
    tactile = np.zeros((8, 6), dtype=np.float32)
    tactile[2:4, 0] = 10

    smoothed, plans = build_gripper_smoothing_plan(
        action,
        tactile,
        np.array([4, 8]),
        sensor_dims=6,
        baseline_frames=2,
        absolute_threshold=8,
        baseline_delta=5,
        consecutive_frames=2,
        search_start_frame=2,
        open_reference_quantile=0.9,
        contact_reference_frames=2,
    )

    np.testing.assert_array_equal(smoothed[:, :-1], action[:, :-1])
    np.testing.assert_allclose(smoothed[:2, -1], 0.14)
    np.testing.assert_allclose(smoothed[2:4, -1], 0.105)
    np.testing.assert_array_equal(smoothed[4:], action[4:])
    assert plans[0].contact_frame == 2
    assert plans[1].contact_frame is None


def test_writer_preserves_observations_and_metadata_while_replacing_action():
    action = np.array(
        [[1.0, 0.14], [2.0, 0.14], [3.0, 0.11], [4.0, 0.10]], dtype=np.float32
    )
    tactile = np.zeros((4, 6), dtype=np.float32)
    tactile[2:, 0] = 10
    smoothed, _ = build_gripper_smoothing_plan(
        action,
        tactile,
        np.array([4]),
        sensor_dims=6,
        baseline_frames=2,
        absolute_threshold=8,
        baseline_delta=5,
        consecutive_frames=2,
        search_start_frame=2,
        contact_reference_frames=2,
    )

    with tempfile.TemporaryDirectory() as directory:
        source_path = Path(directory) / "source.zarr"
        output_path = Path(directory) / "output.zarr"
        source = zarr.group(store=zarr.DirectoryStore(str(source_path)), overwrite=True)
        source.attrs["dataset_name"] = "synthetic"
        data = source.create_group("data")
        data.create_dataset("action", data=action, chunks=(2, 2))
        data.create_dataset("left_gripper1_marker_offset_emb", data=tactile, chunks=(2, 6))
        data.create_dataset("rgb", data=np.arange(12).reshape(4, 3), chunks=(2, 3))
        meta = source.create_group("meta")
        meta.create_dataset("episode_ends", data=np.array([4]), chunks=(1,))
        meta.create_dataset("episode_ids", data=np.array([17]), chunks=(1,))

        write_smoothed_dataset(
            source, output_path, "action", smoothed, copy_batch_rows=2, overwrite=False
        )
        validate_output(source, output_path, "action", smoothed)

        output = zarr.open(str(output_path), mode="r")
        np.testing.assert_array_equal(output["data"]["rgb"][:], source["data"]["rgb"][:])
        np.testing.assert_array_equal(
            output["meta"]["episode_ids"][:], source["meta"]["episode_ids"][:]
        )
        assert output.attrs["dataset_name"] == "synthetic"
