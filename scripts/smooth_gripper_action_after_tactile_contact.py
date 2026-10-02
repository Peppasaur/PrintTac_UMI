#!/usr/bin/env python3
"""Create a replay buffer with tactile-contact-aware gripper action labels.

The source dataset is never changed.  For an episode with sustained tactile
contact, the gripper action is held at a robust open value before contact and
at the measured contact value from the first stable contact frames afterward.
All TCP actions and every observation stream are copied unchanged.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr

try:
    from scripts.trim_dataset_after_tactile_contact import (
        array_create_kwargs,
        copy_attrs,
        detect_contact_frame,
        resolve_replay_buffer_path,
        tactile_contact_score,
        validate_source,
    )
except ModuleNotFoundError:  # Direct execution: python scripts/<script>.py
    from trim_dataset_after_tactile_contact import (
        array_create_kwargs,
        copy_attrs,
        detect_contact_frame,
        resolve_replay_buffer_path,
        tactile_contact_score,
        validate_source,
    )


DEFAULT_INPUT = "dataset/traj_rdp10d_command_downsample2"
DEFAULT_OUTPUT = "dataset/traj_rdp10d_command_downsample2_gripper_hold"


@dataclass(frozen=True)
class EpisodeGripperSmoothing:
    episode: int
    source_start: int
    source_end: int
    contact_frame: int | None
    contact_score: float | None
    baseline: float
    threshold: float
    open_reference: float | None
    contact_reference: float | None

    @property
    def length(self):
        return self.source_end - self.source_start


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Copy a replay buffer while making gripper action labels constant "
            "before and after the first sustained tactile contact."
        )
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--action-key",
        default="action",
        help="Action array under data; the final dimension is the gripper label.",
    )
    parser.add_argument(
        "--tactile-key",
        default="left_gripper1_marker_offset_emb",
        help="Tactile array under the data group.",
    )
    parser.add_argument("--sensor-dims", type=int, default=12)
    parser.add_argument("--sensor-vector-dim", type=int, default=3)
    parser.add_argument("--baseline-frames", type=int, default=20)
    parser.add_argument("--absolute-threshold", type=float, default=50.0)
    parser.add_argument("--baseline-delta", type=float, default=30.0)
    parser.add_argument("--consecutive-frames", type=int, default=3)
    parser.add_argument("--search-start-frame", type=int, default=20)
    parser.add_argument(
        "--open-reference-quantile",
        type=float,
        default=0.9,
        help=(
            "Quantile of pre-contact gripper labels used as the held-open "
            "reference. The current dataset uses larger values for wider opening."
        ),
    )
    parser.add_argument(
        "--contact-reference-frames",
        type=int,
        default=3,
        help="Frames from the detected contact point whose median is held after contact.",
    )
    parser.add_argument("--copy-batch-rows", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args):
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
    if not 0.0 <= args.open_reference_quantile <= 1.0:
        raise ValueError("--open-reference-quantile must be between 0 and 1")
    if args.contact_reference_frames <= 0:
        raise ValueError("--contact-reference-frames must be positive")
    if args.copy_batch_rows <= 0:
        raise ValueError("--copy-batch-rows must be positive")


def validate_action_array(source_root, action_key, total_rows):
    if action_key not in source_root["data"]:
        available = ", ".join(sorted(source_root["data"].keys()))
        raise KeyError(f"Missing data/{action_key}. Available keys: {available}")
    action = source_root["data"][action_key]
    if action.ndim != 2 or action.shape[0] != total_rows or action.shape[1] < 1:
        raise ValueError(
            f"data/{action_key} must have shape (rows, action_dims); got {action.shape}"
        )
    return action


def _finite_quantile(values, quantile, description):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError(f"No finite gripper labels available for {description}")
    return float(np.quantile(values, quantile))


def build_gripper_smoothing_plan(
    action,
    tactile,
    episode_ends,
    *,
    sensor_dims=12,
    sensor_vector_dim=3,
    baseline_frames=20,
    absolute_threshold=50.0,
    baseline_delta=30.0,
    consecutive_frames=3,
    search_start_frame=20,
    open_reference_quantile=0.9,
    contact_reference_frames=3,
):
    """Return a copy of ``action`` with only its gripper column adjusted."""
    action = np.asarray(action)
    if action.ndim != 2 or action.shape[1] < 1:
        raise ValueError(f"Expected action shape (rows, dims), got {action.shape}")
    if not 0.0 <= open_reference_quantile <= 1.0:
        raise ValueError("open_reference_quantile must be between 0 and 1")
    if contact_reference_frames <= 0:
        raise ValueError("contact_reference_frames must be positive")

    episode_ends = np.asarray(episode_ends, dtype=np.int64)
    if episode_ends.ndim != 1 or episode_ends.size == 0:
        raise ValueError("episode_ends must be a non-empty 1D array")
    if int(episode_ends[-1]) != len(action) or len(tactile) != len(action):
        raise ValueError("action, tactile, and episode_ends row counts do not agree")

    smoothed_action = action.copy()
    scores = tactile_contact_score(tactile, sensor_dims, sensor_vector_dim)
    plans = []
    source_start = 0
    for episode, source_end_value in enumerate(episode_ends):
        source_end = int(source_end_value)
        episode_scores = scores[source_start:source_end]
        contact_frame, baseline, threshold = detect_contact_frame(
            episode_scores,
            baseline_frames=baseline_frames,
            absolute_threshold=absolute_threshold,
            baseline_delta=baseline_delta,
            consecutive_frames=consecutive_frames,
            search_start_frame=search_start_frame,
        )
        if contact_frame is None:
            plans.append(
                EpisodeGripperSmoothing(
                    episode=episode,
                    source_start=source_start,
                    source_end=source_end,
                    contact_frame=None,
                    contact_score=None,
                    baseline=baseline,
                    threshold=threshold,
                    open_reference=None,
                    contact_reference=None,
                )
            )
            source_start = source_end
            continue

        gripper = action[source_start:source_end, -1]
        open_reference = _finite_quantile(
            gripper[:contact_frame],
            open_reference_quantile,
            f"episode {episode} before contact",
        )
        contact_end = min(contact_frame + contact_reference_frames, len(gripper))
        contact_reference = _finite_quantile(
            gripper[contact_frame:contact_end],
            0.5,
            f"episode {episode} at contact",
        )
        smoothed_action[source_start : source_start + contact_frame, -1] = open_reference
        smoothed_action[source_start + contact_frame : source_end, -1] = contact_reference
        plans.append(
            EpisodeGripperSmoothing(
                episode=episode,
                source_start=source_start,
                source_end=source_end,
                contact_frame=contact_frame,
                contact_score=float(episode_scores[contact_frame]),
                baseline=baseline,
                threshold=threshold,
                open_reference=open_reference,
                contact_reference=contact_reference,
            )
        )
        source_start = source_end
    return smoothed_action, plans


def print_smoothing_plan(input_path, output_path, plans, dry_run):
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print("episode  frames  contact  score    open_ref  contact_ref  baseline  threshold")
    for plan in plans:
        contact = "none" if plan.contact_frame is None else str(plan.contact_frame)
        score = "-" if plan.contact_score is None else f"{plan.contact_score:.1f}"
        open_ref = "-" if plan.open_reference is None else f"{plan.open_reference:.6f}"
        contact_ref = (
            "-" if plan.contact_reference is None else f"{plan.contact_reference:.6f}"
        )
        print(
            f"{plan.episode:7d}  {plan.length:6d}  {contact:>7}  {score:>7}  "
            f"{open_ref:>8}  {contact_ref:>11}  {plan.baseline:8.1f}  "
            f"{plan.threshold:9.1f}"
        )
    detected = sum(plan.contact_frame is not None for plan in plans)
    print(f"Summary: episodes={len(plans)}, contact_detected={detected}")
    if dry_run:
        print("Dry run only; no output was written.")


def write_smoothed_dataset(
    source_root,
    output_path,
    action_key,
    smoothed_action,
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
    output_root = zarr.group(store=zarr.DirectoryStore(str(output_path)), overwrite=True)
    output_data = output_root.create_group("data")
    output_meta = output_root.create_group("meta")
    copy_attrs(source_root, output_root)
    copy_attrs(source_root["data"], output_data)
    copy_attrs(source_root["meta"], output_meta)

    for key, source_array in source_root["data"].arrays():
        output_array = output_data.create_dataset(
            key, **array_create_kwargs(source_array, source_array.shape)
        )
        copy_attrs(source_array, output_array)
        for start in range(0, source_array.shape[0], copy_batch_rows):
            end = min(start + copy_batch_rows, source_array.shape[0])
            if key == action_key:
                output_array[start:end] = smoothed_action[start:end]
            else:
                output_array[start:end] = source_array[start:end]
        print(f"Copied data/{key}: {source_array.shape[0]} rows")

    for key, source_array in source_root["meta"].arrays():
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


def validate_output(source_root, output_path, action_key, expected_action):
    output_root = zarr.open(str(output_path), mode="r")
    if sorted(output_root["data"].keys()) != sorted(source_root["data"].keys()):
        raise AssertionError("Output data keys differ from source data keys")
    if sorted(output_root["meta"].keys()) != sorted(source_root["meta"].keys()):
        raise AssertionError("Output meta keys differ from source meta keys")
    for key, source_array in source_root["data"].arrays():
        output_array = output_root["data"][key]
        if output_array.shape != source_array.shape or output_array.dtype != source_array.dtype:
            raise AssertionError(f"Output data/{key} shape or dtype differs from source")
        if key != action_key and not np.array_equal(output_array[:], source_array[:]):
            raise AssertionError(f"Output data/{key} differs from source")
    if not np.array_equal(output_root["data"][action_key][:], expected_action):
        raise AssertionError("Output action does not match the smoothing plan")
    if not np.array_equal(
        output_root["meta"]["episode_ends"][:], source_root["meta"]["episode_ends"][:]
    ):
        raise AssertionError("Output episode_ends differs from source")
    for key, source_array in source_root["meta"].arrays():
        if not np.array_equal(output_root["meta"][key][:], source_array[:]):
            raise AssertionError(f"Output meta/{key} differs from source")
    print("Validation passed.")


def main():
    args = parse_args()
    validate_args(args)
    input_path = resolve_replay_buffer_path(args.input)
    output_path = resolve_replay_buffer_path(args.output)
    if input_path == output_path:
        raise ValueError("Input and output must differ; source data is never modified")
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input replay buffer not found: {input_path}")

    source_root = zarr.open(str(input_path), mode="r")
    episode_ends = validate_source(source_root, args.tactile_key)
    action = validate_action_array(source_root, args.action_key, int(episode_ends[-1]))
    smoothed_action, plans = build_gripper_smoothing_plan(
        action[:],
        source_root["data"][args.tactile_key][:],
        episode_ends,
        sensor_dims=args.sensor_dims,
        sensor_vector_dim=args.sensor_vector_dim,
        baseline_frames=args.baseline_frames,
        absolute_threshold=args.absolute_threshold,
        baseline_delta=args.baseline_delta,
        consecutive_frames=args.consecutive_frames,
        search_start_frame=args.search_start_frame,
        open_reference_quantile=args.open_reference_quantile,
        contact_reference_frames=args.contact_reference_frames,
    )
    print_smoothing_plan(input_path, output_path, plans, args.dry_run)
    if args.dry_run:
        return

    write_smoothed_dataset(
        source_root,
        output_path,
        args.action_key,
        smoothed_action,
        args.copy_batch_rows,
        args.overwrite,
    )
    validate_output(source_root, output_path, args.action_key, smoothed_action)
    print(f"Done: {output_path}")


if __name__ == "__main__":
    main()
