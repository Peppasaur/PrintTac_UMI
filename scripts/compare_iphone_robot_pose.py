#!/usr/bin/env python3
"""Compare timestamp-aligned iPhone and robot TCP pose observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import zarr
from scipy.spatial.transform import Rotation

try:
    from scripts.fuse_gello_into_arpose_zarr import (
        episode_bounds,
        interpolate_pose6,
        list_gello_episode_dirs,
        load_gello_episode,
        match_episode_names,
        read_receiver_clock_offset_ms,
    )
    from scripts.pose_error_metrics import (
        compare_pose_sequences,
        error_summary,
        pose6_to_matrices,
        vector_component_summary,
    )
except ModuleNotFoundError:
    from fuse_gello_into_arpose_zarr import (
        episode_bounds,
        interpolate_pose6,
        list_gello_episode_dirs,
        load_gello_episode,
        match_episode_names,
        read_receiver_clock_offset_ms,
    )
    from pose_error_metrics import (
        compare_pose_sequences,
        error_summary,
        pose6_to_matrices,
        vector_component_summary,
    )


DEFAULT_GELLO_INPUT = Path("dataset/gello_polymetis_command_raw")
DEFAULT_ARPOSE_INPUT = Path(
    "/home/shuwang/CodeFile/umi_data/ARPoseStreamer/uploads/arpose_all_source.zarr"
)
DEFAULT_REPORT_OUTPUT = Path("data/pose_comparison/iphone_robot_pose_error.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match GELLO and ARPose episodes by recording time, interpolate actual "
            "robot TCP observations to iPhone timestamps, and compare their poses."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gello-input", type=Path, default=DEFAULT_GELLO_INPUT)
    parser.add_argument("--arpose-input", type=Path, default=DEFAULT_ARPOSE_INPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--episode-match-tolerance-sec", type=float, default=2.0)
    parser.add_argument(
        "--time-alignment",
        choices=("receiver_clock", "movement_start", "none"),
        default="receiver_clock",
        help=(
            "How to convert iPhone timestamps to the PC/robot time domain. "
            "'receiver_clock' uses receiver_transport.csv, 'movement_start' "
            "aligns the first sustained robot motion in both observations, and "
            "'none' compares the raw wall-clock timestamps."
        ),
    )
    parser.add_argument(
        "--iphone-clock-offset-ms",
        type=float,
        default=None,
        help=(
            "Override the offset added to iPhone sender timestamps before robot "
            "interpolation. By default, read clock_offset_ms from "
            "<arpose parent>/<episode>/receiver_transport.csv, with legacy "
            "raw/corrected latency recovery. Pass 0 to disable correction."
        ),
    )
    parser.add_argument(
        "--movement-start-position-threshold-mm",
        type=float,
        default=2.0,
        help="Translation from the first pose required to detect movement.",
    )
    parser.add_argument(
        "--movement-start-rotation-threshold-deg",
        type=float,
        default=0.5,
        help="Rotation from the first pose required to detect movement.",
    )
    parser.add_argument(
        "--movement-start-sustain-frames",
        type=int,
        default=3,
        help="Consecutive above-threshold frames required to detect movement.",
    )
    parser.add_argument(
        "--temporal-downsample",
        type=int,
        default=1,
        help="Keep every Nth iPhone frame independently inside each episode.",
    )
    parser.add_argument(
        "--require-all-matched",
        action="store_true",
        help="Fail unless every episode in both inputs has a unique time match.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.episode_match_tolerance_sec < 0:
        parser.error("--episode-match-tolerance-sec must be non-negative")
    if args.temporal_downsample < 1:
        parser.error("--temporal-downsample must be at least 1")
    if args.iphone_clock_offset_ms is not None and args.time_alignment != "receiver_clock":
        parser.error(
            "--iphone-clock-offset-ms can only be used with "
            "--time-alignment receiver_clock"
        )
    if args.movement_start_position_threshold_mm < 0:
        parser.error("--movement-start-position-threshold-mm must be non-negative")
    if args.movement_start_rotation_threshold_deg < 0:
        parser.error("--movement-start-rotation-threshold-deg must be non-negative")
    if args.movement_start_sustain_frames < 1:
        parser.error("--movement-start-sustain-frames must be at least 1")
    return args


def require_arpose_data(root: zarr.hierarchy.Group) -> Tuple[List[str], np.ndarray]:
    required = (
        "timestamp",
        "robot0_eef_pos",
        "robot0_eef_rot_axis_angle",
    )
    if "data" not in root or "meta" not in root:
        raise KeyError("ARPose zarr must contain data and meta groups")
    missing = [key for key in required if key not in root["data"]]
    if missing:
        raise KeyError(f"ARPose zarr is missing data keys: {missing}")
    if "episode_ends" not in root["meta"]:
        raise KeyError("ARPose zarr is missing meta/episode_ends")
    episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    names = [str(name) for name in root.attrs.get("source_directories", [])]
    if len(names) != len(episode_ends):
        raise ValueError(
            "ARPose source_directories must contain one name per episode: "
            f"got {len(names)} names and {len(episode_ends)} episodes"
        )
    return names, episode_ends


def format_names(names: Sequence[str], limit: int = 6) -> str:
    shown = list(names[:limit])
    suffix = " ..." if len(names) > limit else ""
    return ", ".join(shown) + suffix


def resolve_clock_offset(
    override_ms: float | None, arpose_path: Path, episode_name: str
) -> Dict[str, object]:
    if override_ms is not None:
        return {
            "offset_ms": float(override_ms),
            "source": "command_line_override",
            "sample_count": 0,
            "min_ms": float(override_ms),
            "max_ms": float(override_ms),
        }
    transport_path = arpose_path.parent / episode_name / "receiver_transport.csv"
    return read_receiver_clock_offset_ms(transport_path)


def detect_movement_start(
    timestamp_ns: np.ndarray,
    pose: np.ndarray,
    position_threshold_mm: float,
    rotation_threshold_deg: float,
    sustain_frames: int,
) -> Dict[str, object]:
    timestamp_ns = np.asarray(timestamp_ns, dtype=np.int64).reshape(-1)
    pose = np.asarray(pose, dtype=np.float64)
    if len(timestamp_ns) != len(pose):
        raise ValueError(
            f"Movement timestamps and poses differ in length: "
            f"{len(timestamp_ns)} vs {len(pose)}"
        )
    if len(pose) < sustain_frames + 1:
        raise ValueError(
            f"Need at least {sustain_frames + 1} poses to detect sustained movement"
        )

    matrices = pose6_to_matrices(pose)
    displacement = np.linalg.inv(matrices[0])[None] @ matrices
    position_mm = np.linalg.norm(displacement[:, :3, 3], axis=1) * 1000.0
    rotation_deg = np.rad2deg(
        Rotation.from_matrix(displacement[:, :3, :3]).magnitude()
    )
    above_threshold = (
        (position_mm >= float(position_threshold_mm))
        | (rotation_deg >= float(rotation_threshold_deg))
    )
    candidates = np.flatnonzero(above_threshold)
    for index in candidates:
        end = int(index) + int(sustain_frames)
        if end <= len(above_threshold) and np.all(above_threshold[index:end]):
            return {
                "index": int(index),
                "timestamp_ns": int(timestamp_ns[index]),
                "timestamp_unix_sec": float(timestamp_ns[index] * 1e-9),
                "position_from_first_mm": float(position_mm[index]),
                "rotation_from_first_deg": float(rotation_deg[index]),
            }
    raise ValueError(
        "No sustained movement found with thresholds "
        f"{position_threshold_mm:g} mm / {rotation_threshold_deg:g} deg "
        f"for {sustain_frames} frames"
    )


def resolve_time_offset(
    args: argparse.Namespace,
    arpose_path: Path,
    episode_name: str,
    iphone_timestamp_ns: np.ndarray,
    iphone_pose: np.ndarray,
    robot_timestamp_ns: np.ndarray,
    robot_pose: np.ndarray,
) -> Dict[str, object]:
    if args.time_alignment == "receiver_clock":
        result = resolve_clock_offset(
            args.iphone_clock_offset_ms,
            arpose_path,
            episode_name,
        )
        result["mode"] = "receiver_clock"
        return result
    if args.time_alignment == "none":
        return {
            "offset_ms": 0.0,
            "source": "raw_wall_clock_timestamps",
            "sample_count": 0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "mode": "none",
        }
    if args.time_alignment != "movement_start":
        raise ValueError(f"Unsupported time alignment: {args.time_alignment}")

    iphone_start = detect_movement_start(
        iphone_timestamp_ns,
        iphone_pose,
        args.movement_start_position_threshold_mm,
        args.movement_start_rotation_threshold_deg,
        args.movement_start_sustain_frames,
    )
    robot_start = detect_movement_start(
        robot_timestamp_ns,
        robot_pose,
        args.movement_start_position_threshold_mm,
        args.movement_start_rotation_threshold_deg,
        args.movement_start_sustain_frames,
    )
    offset_ms = (
        int(robot_start["timestamp_ns"]) - int(iphone_start["timestamp_ns"])
    ) * 1e-6
    return {
        "offset_ms": float(offset_ms),
        "source": "first_sustained_movement",
        "sample_count": 2,
        "min_ms": float(offset_ms),
        "max_ms": float(offset_ms),
        "mode": "movement_start",
        "iphone_movement_start": iphone_start,
        "robot_movement_start": robot_start,
        "position_threshold_mm": float(
            args.movement_start_position_threshold_mm
        ),
        "rotation_threshold_deg": float(
            args.movement_start_rotation_threshold_deg
        ),
        "sustain_frames": int(args.movement_start_sustain_frames),
    }


def analyze(args: argparse.Namespace) -> Dict[str, object]:
    arpose_path = args.arpose_input.expanduser().resolve()
    if not arpose_path.exists():
        raise FileNotFoundError(f"ARPose zarr not found: {arpose_path}")
    arpose = zarr.open(str(arpose_path), mode="r")
    arpose_names, episode_ends = require_arpose_data(arpose)

    gello_dirs = list_gello_episode_dirs(args.gello_input)
    gello_by_name = {path.name: path for path in gello_dirs}
    matches, unmatched_arpose, unmatched_gello = match_episode_names(
        arpose_names,
        [path.name for path in gello_dirs],
        tolerance_sec=args.episode_match_tolerance_sec,
    )
    if not matches:
        raise ValueError(
            "No episodes matched by recording time. "
            f"ARPose names: {format_names(arpose_names)}; "
            f"GELLO names: {format_names([path.name for path in gello_dirs])}"
        )
    if args.require_all_matched and (unmatched_arpose or unmatched_gello):
        raise ValueError(
            "Not all episodes matched: "
            f"unmatched ARPose={unmatched_arpose}, unmatched GELLO={unmatched_gello}"
        )

    absolute_position_errors = []
    absolute_rotation_errors = []
    absolute_position_deltas = []
    relative_translation_errors = []
    relative_rotation_errors = []
    relative_translation_deltas = []
    episode_reports = []

    for match in matches:
        source_start, source_end = episode_bounds(episode_ends, match.arpose_index)
        full_source_indices = np.arange(source_start, source_end, dtype=np.int64)
        source_indices = np.arange(
            source_start, source_end, args.temporal_downsample, dtype=np.int64
        )
        full_iphone_timestamp_ns = np.rint(
            np.asarray(
                arpose["data/timestamp"][full_source_indices], dtype=np.float64
            )
            * 1e9
        ).astype(np.int64)
        full_iphone_pose = np.concatenate(
            [
                np.asarray(
                    arpose["data/robot0_eef_pos"][full_source_indices],
                    dtype=np.float64,
                ),
                np.asarray(
                    arpose["data/robot0_eef_rot_axis_angle"][full_source_indices],
                    dtype=np.float64,
                ),
            ],
            axis=1,
        )
        target_timestamp_ns = np.rint(
            np.asarray(arpose["data/timestamp"][source_indices], dtype=np.float64)
            * 1e9
        ).astype(np.int64)

        gello = load_gello_episode(gello_by_name[match.gello_name])
        required_robot = ("robot0_eef_pos", "robot0_eef_rot_axis_angle")
        missing_robot = [key for key in required_robot if key not in gello]
        if missing_robot:
            raise KeyError(
                f"GELLO episode {match.gello_name} is missing robot observations: "
                f"{missing_robot}"
            )
        source_timestamp_ns = np.asarray(
            gello["gripper_timestamp_ns"], dtype=np.int64
        )
        robot_source_pose = np.concatenate(
            [gello["robot0_eef_pos"], gello["robot0_eef_rot_axis_angle"]], axis=1
        )
        time_offset = resolve_time_offset(
            args,
            arpose_path,
            match.arpose_name,
            full_iphone_timestamp_ns,
            full_iphone_pose,
            source_timestamp_ns,
            robot_source_pose,
        )
        robot_query_timestamp_ns = target_timestamp_ns + int(
            round(time_offset["offset_ms"] * 1e6)
        )
        overlap = (
            (robot_query_timestamp_ns >= np.min(source_timestamp_ns))
            & (robot_query_timestamp_ns <= np.max(source_timestamp_ns))
        )
        selected_indices = source_indices[overlap]
        selected_iphone_timestamps_ns = target_timestamp_ns[overlap]
        selected_robot_query_timestamps_ns = robot_query_timestamp_ns[overlap]
        if len(selected_indices) < 2:
            raise ValueError(
                f"Episode {match.arpose_name} has only {len(selected_indices)} "
                "iPhone frames in the robot timestamp range"
            )

        iphone_pose = np.concatenate(
            [
                np.asarray(
                    arpose["data/robot0_eef_pos"][selected_indices],
                    dtype=np.float64,
                ),
                np.asarray(
                    arpose["data/robot0_eef_rot_axis_angle"][selected_indices],
                    dtype=np.float64,
                ),
            ],
            axis=1,
        )
        robot_pose, _ = interpolate_pose6(
            selected_robot_query_timestamps_ns,
            source_timestamp_ns,
            robot_source_pose,
        )
        errors = compare_pose_sequences(iphone_pose, robot_pose)

        absolute_position_errors.append(errors["position_error_mm"][1:])
        absolute_rotation_errors.append(errors["rotation_error_deg"][1:])
        absolute_position_deltas.append(errors["position_delta_mm"][1:])
        relative_translation_errors.append(errors["relative_translation_error_mm"])
        relative_rotation_errors.append(errors["relative_rotation_error_deg"])
        relative_translation_deltas.append(
            errors["relative_translation_delta_mm"]
        )

        frame_dt = np.diff(selected_iphone_timestamps_ns).astype(np.float64) * 1e-9
        episode_reports.append(
            {
                "arpose_name": match.arpose_name,
                "gello_name": match.gello_name,
                "name_time_delta_sec": match.name_time_delta_sec,
                "arpose_episode_frames": int(source_end - source_start),
                "selected_overlap_frames": int(len(selected_indices)),
                "excluded_outside_robot_time_range": int(np.count_nonzero(~overlap)),
                "iphone_to_robot_time_offset": time_offset,
                "selected_median_fps": float(1.0 / np.median(frame_dt)),
                "overlap_duration_sec": float(
                    (
                        selected_iphone_timestamps_ns[-1]
                        - selected_iphone_timestamps_ns[0]
                    )
                    * 1e-9
                ),
                "first_frame_alignment_matrix": errors["alignment_matrix"].tolist(),
                "first_frame_aligned_pose": {
                    "position_error_mm": error_summary(
                        errors["position_error_mm"][1:]
                    ),
                    "rotation_error_deg": error_summary(
                        errors["rotation_error_deg"][1:]
                    ),
                    "endpoint_position_error_mm": float(
                        errors["position_error_mm"][-1]
                    ),
                    "endpoint_rotation_error_deg": float(
                        errors["rotation_error_deg"][-1]
                    ),
                },
                "relative_action": {
                    "translation_error_mm": error_summary(
                        errors["relative_translation_error_mm"]
                    ),
                    "rotation_error_deg": error_summary(
                        errors["relative_rotation_error_deg"]
                    ),
                },
            }
        )

    absolute_position_errors_array = np.concatenate(absolute_position_errors)
    absolute_rotation_errors_array = np.concatenate(absolute_rotation_errors)
    absolute_position_deltas_array = np.concatenate(absolute_position_deltas)
    relative_translation_errors_array = np.concatenate(relative_translation_errors)
    relative_rotation_errors_array = np.concatenate(relative_rotation_errors)
    relative_translation_deltas_array = np.concatenate(relative_translation_deltas)

    return {
        "method": {
            "episode_matching": "recording time parsed from episode directory name",
            "timestamp_alignment": (
                "convert iPhone timestamps to the robot/PC time domain using "
                f"{args.time_alignment}, then interpolate robot position linearly "
                "and rotation with SLERP"
            ),
            "time_alignment": str(args.time_alignment),
            "timestamp_boundary_handling": "exclude iPhone frames outside robot range",
            "first_frame_alignment": (
                "T_iphone_aligned[t] = "
                "T_robot[0] @ inverse(T_iphone[0]) @ T_iphone[t]"
            ),
            "relative_action": "inverse(T_observation[t]) @ T_observation[t+1]",
            "temporal_downsample": int(args.temporal_downsample),
        },
        "inputs": {
            "gello": str(args.gello_input.expanduser().resolve()),
            "arpose": str(arpose_path),
        },
        "matched_episode_count": int(len(matches)),
        "unmatched_arpose": unmatched_arpose,
        "unmatched_gello": unmatched_gello,
        "global": {
            "first_frame_aligned_pose": {
                "position_error_mm": error_summary(
                    absolute_position_errors_array
                ),
                "position_delta_components_mm": vector_component_summary(
                    absolute_position_deltas_array
                ),
                "rotation_error_deg": error_summary(
                    absolute_rotation_errors_array
                ),
            },
            "relative_action": {
                "translation_error_mm": error_summary(
                    relative_translation_errors_array
                ),
                "translation_delta_components_mm": vector_component_summary(
                    relative_translation_deltas_array
                ),
                "rotation_error_deg": error_summary(
                    relative_rotation_errors_array
                ),
            },
        },
        "episodes": episode_reports,
    }


def print_error_line(label: str, values: Dict[str, float], unit: str) -> None:
    print(
        f"  {label}: median={values['median']:.3f} {unit}, "
        f"mean={values['mean']:.3f} {unit}, p95={values['p95']:.3f} {unit}, "
        f"max={values['max']:.3f} {unit}"
    )


def print_report(report: Dict[str, object]) -> None:
    print(f"Matched episodes: {report['matched_episode_count']}")
    print(f"Unmatched ARPose: {report['unmatched_arpose']}")
    print(f"Unmatched GELLO: {report['unmatched_gello']}")
    print("iPhone timestamp to robot timestamp offsets:")
    for episode in report["episodes"]:
        offset = episode["iphone_to_robot_time_offset"]
        print(
            f"  {episode['arpose_name']}: {offset['offset_ms']:+.3f} ms "
            f"({offset['source']})"
        )
        if offset["mode"] == "movement_start":
            iphone_start = offset["iphone_movement_start"]
            robot_start = offset["robot_movement_start"]
            print(
                f"    movement indices: iPhone={iphone_start['index']}, "
                f"robot={robot_start['index']}"
            )
    print("First-frame-aligned pose error (artificial zero first frames excluded):")
    absolute = report["global"]["first_frame_aligned_pose"]
    print_error_line("position", absolute["position_error_mm"], "mm")
    print_error_line("rotation", absolute["rotation_error_deg"], "deg")
    print("Relative-action error:")
    relative = report["global"]["relative_action"]
    print_error_line("translation", relative["translation_error_mm"], "mm")
    print_error_line("rotation", relative["rotation_error_deg"], "deg")
    print("Episodes:")
    for episode in report["episodes"]:
        absolute_episode = episode["first_frame_aligned_pose"]
        relative_episode = episode["relative_action"]
        print(
            f"  {episode['arpose_name']}: {episode['selected_overlap_frames']} frames, "
            f"aligned position median={absolute_episode['position_error_mm']['median']:.3f} mm, "
            f"endpoint={absolute_episode['endpoint_position_error_mm']:.3f} mm, "
            f"relative translation median={relative_episode['translation_error_mm']['median']:.3f} mm"
        )


def write_report(path: Path, report: Dict[str, object], overwrite: bool) -> None:
    path = path.expanduser().resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f"Report exists: {path}; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {path}")


def main() -> None:
    args = parse_args()
    report = analyze(args)
    print_report(report)
    write_report(args.report_output, report, args.overwrite)


if __name__ == "__main__":
    main()
