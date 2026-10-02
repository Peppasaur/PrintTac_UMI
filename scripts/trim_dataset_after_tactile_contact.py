#!/usr/bin/env python3
"""Trim each replay-buffer episode after its first sustained tactile contact."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr


DEFAULT_INPUT = "dataset/traj_rdp10d_command_downsample2"
DEFAULT_OUTPUT = "dataset/traj_rdp10d_command_downsample2_trimmed"


@dataclass(frozen=True)
class EpisodeTrim:
    episode: int
    source_start: int
    source_end: int
    kept_end: int
    trigger_frame: int | None
    trigger_score: float | None
    baseline: float
    threshold: float

    @property
    def old_length(self):
        return self.source_end - self.source_start

    @property
    def new_length(self):
        return self.kept_end - self.source_start


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "For every episode, keep the first sustained tactile-contact frame "
            "and remove all later frames."
        )
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Input dataset directory or replay_buffer.zarr path.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output dataset directory or replay_buffer.zarr path.",
    )
    parser.add_argument(
        "--tactile-key",
        default="left_gripper1_marker_offset_emb",
        help="Tactile array under the data group.",
    )
    parser.add_argument(
        "--sensor-dims",
        type=int,
        default=12,
        help="Leading tactile dimensions containing sensor xyz values.",
    )
    parser.add_argument(
        "--sensor-vector-dim",
        type=int,
        default=3,
        help="Number of components in each tactile sensor vector.",
    )
    parser.add_argument(
        "--baseline-frames",
        type=int,
        default=20,
        help="Initial episode frames used to estimate the no-contact baseline.",
    )
    parser.add_argument(
        "--absolute-threshold",
        type=float,
        default=50.0,
        help="Minimum max sensor-vector norm required for contact.",
    )
    parser.add_argument(
        "--baseline-delta",
        type=float,
        default=30.0,
        help="Required increase above the episode baseline.",
    )
    parser.add_argument(
        "--consecutive-frames",
        type=int,
        default=3,
        help="Consecutive threshold crossings required to confirm contact.",
    )
    parser.add_argument(
        "--search-start-frame",
        type=int,
        default=20,
        help="Do not detect contact before this episode-local frame.",
    )
    parser.add_argument(
        "--keep-after-trigger",
        type=int,
        default=0,
        help="Additional frames to retain after the first contact frame.",
    )
    parser.add_argument(
        "--copy-batch-rows",
        type=int,
        default=256,
        help="Maximum output rows copied per batch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the proposed trim without writing output.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output replay_buffer.zarr.",
    )
    return parser.parse_args()


def resolve_replay_buffer_path(path):
    path = Path(path).expanduser().resolve()
    if path.name == "replay_buffer.zarr" or path.suffix == ".zarr":
        return path
    return path / "replay_buffer.zarr"


def validate_detection_args(args):
    if args.sensor_dims <= 0:
        raise ValueError("--sensor-dims must be positive")
    if args.sensor_vector_dim <= 0:
        raise ValueError("--sensor-vector-dim must be positive")
    if args.sensor_dims % args.sensor_vector_dim != 0:
        raise ValueError("--sensor-dims must be divisible by --sensor-vector-dim")
    if args.baseline_frames <= 0:
        raise ValueError("--baseline-frames must be positive")
    if args.consecutive_frames <= 0:
        raise ValueError("--consecutive-frames must be positive")
    if args.search_start_frame < 0:
        raise ValueError("--search-start-frame must be non-negative")
    if args.keep_after_trigger < 0:
        raise ValueError("--keep-after-trigger must be non-negative")
    if args.copy_batch_rows <= 0:
        raise ValueError("--copy-batch-rows must be positive")


def tactile_contact_score(tactile, sensor_dims=12, sensor_vector_dim=3):
    tactile = np.asarray(tactile)
    if tactile.ndim != 2:
        raise ValueError(f"Expected a 2D tactile array, got shape {tactile.shape}")
    if sensor_dims > tactile.shape[1]:
        raise ValueError(
            f"Requested {sensor_dims} tactile dimensions, but array has {tactile.shape[1]}"
        )
    if sensor_dims % sensor_vector_dim != 0:
        raise ValueError("sensor_dims must be divisible by sensor_vector_dim")

    sensor_vectors = tactile[:, :sensor_dims].reshape(
        tactile.shape[0], sensor_dims // sensor_vector_dim, sensor_vector_dim
    )
    return np.linalg.norm(sensor_vectors, axis=-1).max(axis=-1)


def detect_contact_frame(
    scores,
    baseline_frames=20,
    absolute_threshold=50.0,
    baseline_delta=30.0,
    consecutive_frames=3,
    search_start_frame=20,
):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    baseline_values = scores[: min(len(scores), baseline_frames)]
    baseline_values = baseline_values[np.isfinite(baseline_values)]
    if baseline_values.size == 0:
        return None, float("nan"), float("nan")

    baseline = float(np.median(baseline_values))
    threshold = max(float(absolute_threshold), baseline + float(baseline_delta))
    above = np.isfinite(scores) & (scores >= threshold)
    above[: min(search_start_frame, len(above))] = False

    run_length = 0
    for frame, is_above in enumerate(above):
        run_length = run_length + 1 if is_above else 0
        if run_length >= consecutive_frames:
            return frame - consecutive_frames + 1, baseline, threshold
    return None, baseline, threshold


def validate_source(root, tactile_key):
    if "data" not in root or "meta" not in root:
        raise KeyError("Replay buffer must contain data and meta groups")
    if "episode_ends" not in root["meta"]:
        raise KeyError("Replay buffer is missing meta/episode_ends")
    if tactile_key not in root["data"]:
        available = ", ".join(sorted(root["data"].keys()))
        raise KeyError(f"Missing data/{tactile_key}. Available keys: {available}")

    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    if episode_ends.ndim != 1 or episode_ends.size == 0:
        raise ValueError("meta/episode_ends must be a non-empty 1D array")
    if np.any(np.diff(episode_ends) <= 0):
        raise ValueError("meta/episode_ends must be strictly increasing")

    total_rows = int(episode_ends[-1])
    for key, array in root["data"].arrays():
        if array.ndim == 0 or array.shape[0] != total_rows:
            raise ValueError(
                f"data/{key} has shape {array.shape}; first dimension must be {total_rows}"
            )
    return episode_ends


def build_trim_plan(
    tactile,
    episode_ends,
    sensor_dims=12,
    sensor_vector_dim=3,
    baseline_frames=20,
    absolute_threshold=50.0,
    baseline_delta=30.0,
    consecutive_frames=3,
    search_start_frame=20,
    keep_after_trigger=0,
):
    scores = tactile_contact_score(tactile, sensor_dims, sensor_vector_dim)
    selected_parts = []
    trims = []
    source_start = 0
    output_end = 0

    for episode, source_end_value in enumerate(episode_ends):
        source_end = int(source_end_value)
        episode_scores = scores[source_start:source_end]
        trigger, baseline, threshold = detect_contact_frame(
            episode_scores,
            baseline_frames=baseline_frames,
            absolute_threshold=absolute_threshold,
            baseline_delta=baseline_delta,
            consecutive_frames=consecutive_frames,
            search_start_frame=search_start_frame,
        )
        if trigger is None:
            kept_end = source_end
            trigger_score = None
        else:
            kept_end = min(
                source_start + trigger + keep_after_trigger + 1,
                source_end,
            )
            trigger_score = float(episode_scores[trigger])

        selected_parts.append(np.arange(source_start, kept_end, dtype=np.int64))
        output_end += kept_end - source_start
        trims.append(
            EpisodeTrim(
                episode=episode,
                source_start=source_start,
                source_end=source_end,
                kept_end=kept_end,
                trigger_frame=trigger,
                trigger_score=trigger_score,
                baseline=baseline,
                threshold=threshold,
            )
        )
        source_start = source_end

    selected_indices = np.concatenate(selected_parts)
    new_episode_ends = np.cumsum(
        [trim.new_length for trim in trims], dtype=np.int64
    )
    return selected_indices, new_episode_ends, trims


def print_trim_plan(input_path, output_path, trims, old_rows, new_rows, dry_run):
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(
        "episode  old  new  removed  trigger  score    baseline  threshold"
    )
    for trim in trims:
        trigger = "none" if trim.trigger_frame is None else str(trim.trigger_frame)
        score = "-" if trim.trigger_score is None else f"{trim.trigger_score:.1f}"
        print(
            f"{trim.episode:7d}  {trim.old_length:3d}  {trim.new_length:3d}  "
            f"{trim.old_length - trim.new_length:7d}  {trigger:>7}  {score:>7}  "
            f"{trim.baseline:8.1f}  {trim.threshold:9.1f}"
        )
    detected = sum(trim.trigger_frame is not None for trim in trims)
    print(
        f"Summary: episodes={len(trims)}, detected={detected}, "
        f"rows={old_rows}->{new_rows}, removed={old_rows - new_rows}"
    )
    if dry_run:
        print("Dry run only; no output was written.")


def array_create_kwargs(source, shape):
    chunks = list(source.chunks)
    chunks[0] = min(chunks[0], shape[0])
    return {
        "shape": shape,
        "dtype": source.dtype,
        "chunks": tuple(chunks),
        "compressor": source.compressor,
        "filters": source.filters,
        "fill_value": source.fill_value,
        "order": source.order,
        "overwrite": True,
    }


def read_rows(array, indices):
    if len(indices) == 0:
        return array[:0]
    if np.all(np.diff(indices) == 1):
        return array[int(indices[0]) : int(indices[-1]) + 1]
    selection = (indices,) + tuple(slice(None) for _ in array.shape[1:])
    return array.get_orthogonal_selection(selection)


def copy_attrs(source, destination):
    destination.attrs.update(dict(source.attrs))


def terminal_hold_actions(data, episode_ends):
    required_keys = ("action", "left_robot_tcp_pose", "left_robot_gripper_width")
    missing = [key for key in required_keys if key not in data]
    if missing:
        raise KeyError(
            "Cannot rebuild terminal hold actions; missing data keys: "
            + ", ".join(missing)
        )

    action = data["action"]
    tcp_pose = data["left_robot_tcp_pose"]
    gripper = data["left_robot_gripper_width"]
    if action.ndim != 2 or tcp_pose.ndim != 2 or gripper.ndim != 2:
        raise ValueError("Action, TCP pose, and gripper arrays must all be 2D")
    if action.shape[1] != tcp_pose.shape[1] + gripper.shape[1]:
        raise ValueError(
            f"Cannot build hold action with dimensions action={action.shape[1]}, "
            f"tcp_pose={tcp_pose.shape[1]}, gripper={gripper.shape[1]}"
        )

    terminal_indices = np.asarray(episode_ends, dtype=np.int64) - 1
    holds = [
        np.concatenate(
            [np.asarray(tcp_pose[index]), np.asarray(gripper[index])]
        )
        for index in terminal_indices
    ]
    return terminal_indices, np.asarray(holds, dtype=action.dtype)


def rebuild_terminal_hold_actions(data, episode_ends):
    terminal_indices, holds = terminal_hold_actions(data, episode_ends)
    action = data["action"]
    for index, hold in zip(terminal_indices, holds):
        action[int(index)] = hold
    print(f"Rebuilt terminal hold actions: {len(terminal_indices)} episodes")


def write_trimmed_dataset(
    source_root,
    output_path,
    selected_indices,
    new_episode_ends,
    copy_batch_rows,
    overwrite,
):
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_path} already exists; pass --overwrite to replace it"
            )
        shutil.rmtree(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_store = zarr.DirectoryStore(str(output_path))
    output_root = zarr.group(store=output_store, overwrite=True)
    output_data = output_root.create_group("data")
    output_meta = output_root.create_group("meta")
    copy_attrs(source_root, output_root)
    copy_attrs(source_root["data"], output_data)
    copy_attrs(source_root["meta"], output_meta)

    new_rows = len(selected_indices)
    for key, source_array in source_root["data"].arrays():
        shape = (new_rows,) + source_array.shape[1:]
        output_array = output_data.create_dataset(
            key, **array_create_kwargs(source_array, shape)
        )
        copy_attrs(source_array, output_array)
        for output_start in range(0, new_rows, copy_batch_rows):
            output_end = min(output_start + copy_batch_rows, new_rows)
            source_indices = selected_indices[output_start:output_end]
            output_array[output_start:output_end] = read_rows(
                source_array, source_indices
            )
        print(f"Copied data/{key}: {new_rows} rows")

    rebuild_terminal_hold_actions(output_data, new_episode_ends)

    source_episode_ends = source_root["meta"]["episode_ends"]
    episode_ends_array = output_meta.create_dataset(
        "episode_ends",
        data=new_episode_ends,
        dtype=source_episode_ends.dtype,
        chunks=(min(source_episode_ends.chunks[0], len(new_episode_ends)),),
        compressor=source_episode_ends.compressor,
        filters=source_episode_ends.filters,
        fill_value=source_episode_ends.fill_value,
        order=source_episode_ends.order,
        overwrite=True,
    )
    copy_attrs(source_episode_ends, episode_ends_array)

    for key, source_array in source_root["meta"].arrays():
        if key == "episode_ends":
            continue
        output_array = output_meta.create_dataset(
            key,
            data=source_array[:],
            dtype=source_array.dtype,
            chunks=source_array.chunks,
            compressor=source_array.compressor,
            filters=source_array.filters,
            fill_value=source_array.fill_value,
            order=source_array.order,
            overwrite=True,
        )
        copy_attrs(source_array, output_array)


def validate_output(output_path, expected_episode_ends, source_keys):
    output_root = zarr.open(str(output_path), mode="r")
    output_episode_ends = np.asarray(
        output_root["meta"]["episode_ends"][:], dtype=np.int64
    )
    if not np.array_equal(output_episode_ends, expected_episode_ends):
        raise AssertionError("Output episode_ends does not match the trim plan")
    if np.any(np.diff(output_episode_ends) <= 0):
        raise AssertionError("Output episode_ends is not strictly increasing")

    output_keys = sorted(output_root["data"].keys())
    if output_keys != sorted(source_keys):
        raise AssertionError("Output data keys differ from source data keys")
    expected_rows = int(output_episode_ends[-1])
    for key, array in output_root["data"].arrays():
        if array.shape[0] != expected_rows:
            raise AssertionError(
                f"Output data/{key} has {array.shape[0]} rows, expected {expected_rows}"
            )
    terminal_indices, expected_holds = terminal_hold_actions(
        output_root["data"], output_episode_ends
    )
    actual_holds = np.stack(
        [output_root["data/action"][int(index)] for index in terminal_indices]
    )
    if not np.array_equal(actual_holds, expected_holds):
        raise AssertionError("Output terminal actions are not hold actions")
    print("Validation passed.")


def main():
    args = parse_args()
    validate_detection_args(args)
    input_path = resolve_replay_buffer_path(args.input)
    output_path = resolve_replay_buffer_path(args.output)
    if input_path == output_path:
        raise ValueError("Input and output must be different; source data is never modified")
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input replay buffer not found: {input_path}")

    source_root = zarr.open(str(input_path), mode="r")
    episode_ends = validate_source(source_root, args.tactile_key)
    tactile = source_root["data"][args.tactile_key][:]
    selected_indices, new_episode_ends, trims = build_trim_plan(
        tactile,
        episode_ends,
        sensor_dims=args.sensor_dims,
        sensor_vector_dim=args.sensor_vector_dim,
        baseline_frames=args.baseline_frames,
        absolute_threshold=args.absolute_threshold,
        baseline_delta=args.baseline_delta,
        consecutive_frames=args.consecutive_frames,
        search_start_frame=args.search_start_frame,
        keep_after_trigger=args.keep_after_trigger,
    )
    print_trim_plan(
        input_path,
        output_path,
        trims,
        old_rows=int(episode_ends[-1]),
        new_rows=len(selected_indices),
        dry_run=args.dry_run,
    )
    if args.dry_run:
        return

    source_keys = list(source_root["data"].keys())
    write_trimmed_dataset(
        source_root,
        output_path,
        selected_indices,
        new_episode_ends,
        copy_batch_rows=args.copy_batch_rows,
        overwrite=args.overwrite,
    )
    validate_output(output_path, new_episode_ends, source_keys)
    print(f"Done: {output_path}")


if __name__ == "__main__":
    main()
