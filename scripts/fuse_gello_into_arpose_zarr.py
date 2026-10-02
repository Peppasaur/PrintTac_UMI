#!/usr/bin/env python3
"""Fuse timestamp-aligned GELLO gripper and magnet data into ARPose episodes."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import zarr
from scipy.spatial.transform import Rotation, Slerp

try:
    from scripts.pack_gello_polymetis_raw_to_zarr import save_rgb_magnet_video
except ModuleNotFoundError:
    from pack_gello_polymetis_raw_to_zarr import save_rgb_magnet_video

try:
    from scripts.pose_error_metrics import compare_pose_sequences, error_summary
except ModuleNotFoundError:
    from pose_error_metrics import compare_pose_sequences, error_summary


DEFAULT_GELLO_INPUT = Path("dataset/gello_polymetis_command_raw")
DEFAULT_ARPOSE_INPUT = Path(
    "/home/shuwang/CodeFile/umi_data/ARPoseStreamer/uploads/arpose_all_source.zarr"
)
DEFAULT_ARPOSE_OUTPUT = Path(
    "/home/shuwang/CodeFile/umi_data/ARPoseStreamer/uploads/"
    "arpose_all_source_gello_fused.zarr"
)
DEFAULT_GRIPPER_STROKE_M = 0.08
DEFAULT_MAGNET_SAMPLES_PER_FRAME = 8
DEFAULT_MATCH_TOLERANCE_SEC = 2.0
DEFAULT_VIDEO_PANEL_WIDTH = 420
MAGNET_ABNORMAL_ABS_THRESHOLD = 5000.0


@dataclass(frozen=True)
class EpisodeMatch:
    arpose_index: int
    arpose_name: str
    gello_name: str
    name_time_delta_sec: float


@dataclass
class EpisodeFusion:
    match: EpisodeMatch
    source_start: int
    source_end: int
    gripper_value: np.ndarray
    magnet_xyz: np.ndarray
    magnet_timestamp_ns: np.ndarray
    magnet_sample_count: np.ndarray
    report: Dict[str, object]
    aligned_timestamp_sec: Optional[np.ndarray] = None
    robot_pose: Optional[np.ndarray] = None
    action: Optional[np.ndarray] = None
    trajectory_error_values: Optional[Dict[str, np.ndarray]] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match ARPose and GELLO episodes by recording time in their directory "
            "names, then align gripper and magnet samples using Unix timestamps."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gello-input", type=Path, default=DEFAULT_GELLO_INPUT)
    parser.add_argument("--arpose-input", type=Path, default=DEFAULT_ARPOSE_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARPOSE_OUTPUT)
    parser.add_argument(
        "--robot-data-source",
        choices=("arpose", "gello"),
        default="arpose",
        help=(
            "Source for robot TCP observations and actions. 'arpose' preserves "
            "the phone-derived pose/action. 'gello' aligns the teleop recorder's "
            "actual robot TCP and commanded action to each ARPose image timestamp."
        ),
    )
    parser.add_argument(
        "--video-output",
        type=Path,
        default=None,
        help="Defaults to <output name without .zarr>_videos beside --output.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=None,
        help="Defaults to <output name without .zarr>_fusion_report.json.",
    )
    parser.add_argument(
        "--episode-match-tolerance-sec",
        type=float,
        default=DEFAULT_MATCH_TOLERANCE_SEC,
    )
    parser.add_argument(
        "--time-alignment",
        choices=("receiver_clock", "none"),
        default="receiver_clock",
        help=(
            "How to convert iPhone sender timestamps to the PC/GELLO clock. "
            "'receiver_clock' reads each episode's receiver_transport.csv; "
            "'none' preserves the old uncorrected behavior."
        ),
    )
    parser.add_argument(
        "--iphone-clock-offset-ms",
        type=float,
        default=None,
        help=(
            "Override the offset added to every iPhone timestamp. By default, "
            "use clock_offset_ms from each episode's receiver_transport.csv, "
            "or recover it from legacy raw/corrected latency columns."
        ),
    )
    parser.add_argument(
        "--exclude-arpose-episode",
        action="append",
        default=[],
        metavar="YYYYMMDD-HHMMSS",
        help=(
            "Exclude an ARPose episode from fusion. Repeat the option to exclude "
            "multiple episodes."
        ),
    )
    parser.add_argument(
        "--gripper-stroke-m",
        type=float,
        default=DEFAULT_GRIPPER_STROKE_M,
        help=(
            "Physical opening represented by GELLO raw value 1.0. The teleop "
            "default for --gripper-type franka_hand is 0.08 m."
        ),
    )
    parser.add_argument(
        "--gripper-value-mode",
        choices=("raw_ratio", "width_m"),
        default="width_m",
        help=(
            "How to write GELLO robot0_gripper_width and action[...,6]. "
            "'raw_ratio' preserves the original 0..1 teleop open ratio; "
            "'width_m' multiplies it by --gripper-stroke-m."
        ),
    )
    parser.add_argument(
        "--magnet-samples-per-frame",
        type=int,
        default=DEFAULT_MAGNET_SAMPLES_PER_FRAME,
    )
    parser.add_argument("--video-fps", type=float, default=60.0)
    parser.add_argument("--video-panel-width", type=int, default=DEFAULT_VIDEO_PANEL_WIDTH)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--require-all-matched",
        action="store_true",
        help="Fail unless every ARPose and GELLO episode has a unique match.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    if args.episode_match_tolerance_sec < 0:
        parser.error("--episode-match-tolerance-sec must be non-negative")
    if args.iphone_clock_offset_ms is not None and args.time_alignment != "receiver_clock":
        parser.error(
            "--iphone-clock-offset-ms can only be used with "
            "--time-alignment receiver_clock"
        )
    if args.gripper_stroke_m <= 0:
        parser.error("--gripper-stroke-m must be positive")
    if args.magnet_samples_per_frame <= 0:
        parser.error("--magnet-samples-per-frame must be positive")
    if args.video_fps <= 0:
        parser.error("--video-fps must be positive")
    return args


def recording_time_from_name(name: str) -> datetime:
    normalized = name.strip().replace("_", "-")
    try:
        return datetime.strptime(normalized, "%Y%m%d-%H%M%S")
    except ValueError as exc:
        raise ValueError(
            f"Episode name must use YYYYMMDD-HHMMSS or YYYYMMDD_HHMMSS: {name}"
        ) from exc


def list_gello_episode_dirs(input_dir: Path) -> List[Path]:
    input_dir = input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"GELLO input directory not found: {input_dir}")
    episodes = [
        path
        for path in sorted(input_dir.iterdir())
        if path.is_dir() and any(path.glob("frame_*.pkl"))
    ]
    if not episodes:
        raise FileNotFoundError(f"No GELLO frame_*.pkl episodes found in {input_dir}")
    return episodes


def read_receiver_clock_offset_ms(path: Path) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Receiver transport CSV not found: {path}")
    explicit_values = []
    legacy_values = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        for row in reader:
            try:
                value = float(row.get("clock_offset_ms", ""))
            except (TypeError, ValueError):
                value = np.nan
            if np.isfinite(value) and row.get("kind") in (None, "", "video"):
                explicit_values.append(value)
            try:
                raw_latency_ms = float(row.get("raw_latency_ms", ""))
                corrected_latency_ms = float(row.get("corrected_latency_ms", ""))
            except (TypeError, ValueError):
                continue
            legacy_value = raw_latency_ms - corrected_latency_ms
            if np.isfinite(legacy_value):
                legacy_values.append(legacy_value)

    if explicit_values:
        values = explicit_values
        method = "clock_offset_ms"
    elif legacy_values:
        values = legacy_values
        method = "raw_latency_ms_minus_corrected_latency_ms"
    else:
        raise ValueError(
            f"{path} has neither finite clock_offset_ms values nor recoverable "
            "raw/corrected latency pairs"
        )
    values_array = np.asarray(values, dtype=np.float64)
    return {
        "offset_ms": float(np.median(values_array)),
        "source": str(path.resolve()),
        "method": method,
        "sample_count": int(len(values_array)),
        "min_ms": float(np.min(values_array)),
        "max_ms": float(np.max(values_array)),
    }


def resolve_iphone_to_pc_clock_offset(
    arpose_input: Path,
    episode_name: str,
    time_alignment: str,
    override_ms: Optional[float],
) -> Dict[str, object]:
    if time_alignment == "none":
        return {
            "mode": "none",
            "offset_ms": 0.0,
            "source": "raw_wall_clock_timestamps",
            "method": "none",
            "sample_count": 0,
            "min_ms": 0.0,
            "max_ms": 0.0,
        }
    if time_alignment != "receiver_clock":
        raise ValueError(f"Unsupported time alignment: {time_alignment}")
    if override_ms is not None:
        value = float(override_ms)
        return {
            "mode": "receiver_clock",
            "offset_ms": value,
            "source": "command_line_override",
            "method": "command_line_override",
            "sample_count": 0,
            "min_ms": value,
            "max_ms": value,
        }
    result = read_receiver_clock_offset_ms(
        arpose_input.parent / episode_name / "receiver_transport.csv"
    )
    result["mode"] = "receiver_clock"
    return result


def apply_clock_offset_ns(
    iphone_timestamp_ns: np.ndarray, offset_ms: float
) -> np.ndarray:
    timestamp_ns = np.asarray(iphone_timestamp_ns, dtype=np.int64)
    offset_ns = int(round(float(offset_ms) * 1e6))
    return timestamp_ns + offset_ns


def match_episode_names(
    arpose_names: Sequence[str],
    gello_names: Sequence[str],
    tolerance_sec: float,
) -> Tuple[List[EpisodeMatch], List[str], List[str]]:
    arpose_times = [recording_time_from_name(name) for name in arpose_names]
    gello_times = [recording_time_from_name(name) for name in gello_names]
    candidates = []
    for arpose_index, arpose_time in enumerate(arpose_times):
        for gello_index, gello_time in enumerate(gello_times):
            delta = abs((arpose_time - gello_time).total_seconds())
            if delta <= tolerance_sec:
                candidates.append((delta, arpose_index, gello_index))

    matched_arpose = set()
    matched_gello = set()
    matches = []
    for delta, arpose_index, gello_index in sorted(candidates):
        if arpose_index in matched_arpose or gello_index in matched_gello:
            continue
        matched_arpose.add(arpose_index)
        matched_gello.add(gello_index)
        matches.append(
            EpisodeMatch(
                arpose_index=arpose_index,
                arpose_name=str(arpose_names[arpose_index]),
                gello_name=str(gello_names[gello_index]),
                name_time_delta_sec=float(delta),
            )
        )

    matches.sort(key=lambda match: match.arpose_index)
    unmatched_arpose = [
        str(name) for index, name in enumerate(arpose_names) if index not in matched_arpose
    ]
    unmatched_gello = [
        str(name) for index, name in enumerate(gello_names) if index not in matched_gello
    ]
    return matches, unmatched_arpose, unmatched_gello


def scalar(frame: Dict[str, object], key: str) -> float:
    value = np.asarray(frame[key]).reshape(-1)
    if value.size != 1:
        raise ValueError(f"Expected scalar frame field {key}, got shape {np.asarray(frame[key]).shape}")
    return float(value[0])


def load_gello_episode(episode_dir: Path) -> Dict[str, np.ndarray]:
    frame_paths = sorted(episode_dir.glob("frame_*.pkl"))
    if not frame_paths:
        raise FileNotFoundError(f"No frame_*.pkl files in {episode_dir}")

    frames = []
    required = (
        "robot0_gripper_width",
        "magnet_xyz",
        "magnet_timestamp_ns",
        "magnet_sample_count",
    )
    for frame_path in frame_paths:
        with frame_path.open("rb") as file:
            frame = pickle.load(file)
        missing = [key for key in required if key not in frame]
        if missing:
            raise KeyError(f"{frame_path} is missing keys: {missing}")
        frames.append(frame)

    gripper_timestamp_ns = []
    for frame in frames:
        if "collection_time_ns" in frame:
            gripper_timestamp_ns.append(int(round(scalar(frame, "collection_time_ns"))))
        elif "timestamp" in frame:
            gripper_timestamp_ns.append(int(round(scalar(frame, "timestamp") * 1e9)))
        else:
            raise KeyError(f"GELLO frame in {episode_dir} has no timestamp")

    gripper_raw = np.asarray(
        [scalar(frame, "robot0_gripper_width") for frame in frames], dtype=np.float64
    )
    if not np.all(np.isfinite(gripper_raw)):
        raise ValueError(f"Non-finite GELLO gripper values in {episode_dir}")
    if np.any(gripper_raw < -1e-6) or np.any(gripper_raw > 1.0 + 1e-6):
        raise ValueError(
            f"GELLO gripper values must be raw opening ratios in [0, 1], got "
            f"[{gripper_raw.min():.6f}, {gripper_raw.max():.6f}] in {episode_dir}"
        )

    result = {
        "gripper_timestamp_ns": np.asarray(gripper_timestamp_ns, dtype=np.int64),
        "gripper_raw": np.clip(gripper_raw, 0.0, 1.0),
        "magnet_xyz": np.asarray([frame["magnet_xyz"] for frame in frames], dtype=np.float32),
        "magnet_timestamp_ns": np.asarray(
            [frame["magnet_timestamp_ns"] for frame in frames], dtype=np.int64
        ),
        "magnet_sample_count": np.asarray(
            [frame["magnet_sample_count"] for frame in frames], dtype=np.int32
        ).reshape(-1),
    }
    robot_fields = ("robot0_eef_pos", "robot0_eef_rot_axis_angle", "action")
    if all(all(key in frame for key in robot_fields) for frame in frames):
        result.update(
            {
                "robot0_eef_pos": np.asarray(
                    [frame["robot0_eef_pos"] for frame in frames], dtype=np.float32
                ),
                "robot0_eef_rot_axis_angle": np.asarray(
                    [frame["robot0_eef_rot_axis_angle"] for frame in frames],
                    dtype=np.float32,
                ),
                "action": np.asarray(
                    [frame["action"] for frame in frames], dtype=np.float32
                ),
            }
        )
    return result


def unique_time_series(
    timestamp_ns: np.ndarray, values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    timestamp_ns = np.asarray(timestamp_ns, dtype=np.int64).reshape(-1)
    values = np.asarray(values)
    if len(timestamp_ns) != len(values):
        raise ValueError("Timestamp and value lengths differ")
    if len(timestamp_ns) == 0:
        return timestamp_ns, values
    order = np.argsort(timestamp_ns, kind="stable")
    sorted_time = timestamp_ns[order]
    sorted_values = values[order]
    unique_time, first_indices = np.unique(sorted_time, return_index=True)
    return unique_time, sorted_values[first_indices]


def interpolate_gripper_width(
    target_timestamp_ns: np.ndarray,
    source_timestamp_ns: np.ndarray,
    source_gripper_raw: np.ndarray,
    stroke_m: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    return interpolate_gripper_values(
        target_timestamp_ns,
        source_timestamp_ns,
        source_gripper_raw,
        value_mode="width_m",
        stroke_m=stroke_m,
    )


def interpolate_gripper_values(
    target_timestamp_ns: np.ndarray,
    source_timestamp_ns: np.ndarray,
    source_gripper_raw: np.ndarray,
    value_mode: str,
    stroke_m: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    source_timestamp_ns, source_gripper_raw = unique_time_series(
        source_timestamp_ns, source_gripper_raw
    )
    if len(source_timestamp_ns) == 0:
        raise ValueError("Cannot align an empty gripper time series")
    target_timestamp_ns = np.asarray(target_timestamp_ns, dtype=np.int64).reshape(-1)
    target_sec = target_timestamp_ns.astype(np.float64) * 1e-9
    source_sec = source_timestamp_ns.astype(np.float64) * 1e-9
    raw = np.interp(target_sec, source_sec, source_gripper_raw)
    if value_mode == "raw_ratio":
        values = raw.astype(np.float32).reshape(-1, 1)
        output_unit = "open_ratio"
    elif value_mode == "width_m":
        values = (raw * float(stroke_m)).astype(np.float32).reshape(-1, 1)
        output_unit = "m"
    else:
        raise ValueError(f"Unsupported gripper value mode: {value_mode}")
    before = target_timestamp_ns < source_timestamp_ns[0]
    after = target_timestamp_ns > source_timestamp_ns[-1]
    boundary_gap_ns = np.zeros(len(target_timestamp_ns), dtype=np.int64)
    boundary_gap_ns[before] = source_timestamp_ns[0] - target_timestamp_ns[before]
    boundary_gap_ns[after] = target_timestamp_ns[after] - source_timestamp_ns[-1]
    report = {
        "source_raw_min": float(np.min(source_gripper_raw)),
        "source_raw_max": float(np.max(source_gripper_raw)),
        "output_value_mode": value_mode,
        "output_unit": output_unit,
        "aligned_value_min": float(np.min(values)),
        "aligned_value_max": float(np.max(values)),
        "frames_before_source": int(np.count_nonzero(before)),
        "frames_after_source": int(np.count_nonzero(after)),
        "max_boundary_gap_ms": float(np.max(boundary_gap_ns, initial=0) * 1e-6),
    }
    if value_mode == "width_m":
        report.update(
            {
                "aligned_width_min_m": float(np.min(values)),
                "aligned_width_max_m": float(np.max(values)),
            }
        )
    return values, report


def interpolate_pose6(
    target_timestamp_ns: np.ndarray,
    source_timestamp_ns: np.ndarray,
    source_pose6: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Interpolate position linearly and axis-angle rotation with SLERP."""
    source_timestamp_ns, source_pose6 = unique_time_series(
        source_timestamp_ns, source_pose6
    )
    source_pose6 = np.asarray(source_pose6, dtype=np.float64)
    if source_pose6.ndim != 2 or source_pose6.shape[1] != 6:
        raise ValueError(f"Expected source pose [T,6], got {source_pose6.shape}")
    if len(source_timestamp_ns) == 0:
        raise ValueError("Cannot align an empty pose time series")
    if not np.all(np.isfinite(source_pose6)):
        raise ValueError("Cannot align non-finite robot poses")

    target_timestamp_ns = np.asarray(target_timestamp_ns, dtype=np.int64).reshape(-1)
    origin_ns = int(source_timestamp_ns[0])
    source_sec = (source_timestamp_ns - origin_ns).astype(np.float64) * 1e-9
    target_sec = (target_timestamp_ns - origin_ns).astype(np.float64) * 1e-9
    clipped_target_sec = np.clip(target_sec, source_sec[0], source_sec[-1])

    position = np.column_stack(
        [
            np.interp(clipped_target_sec, source_sec, source_pose6[:, axis])
            for axis in range(3)
        ]
    )
    source_rotation = Rotation.from_rotvec(source_pose6[:, 3:6])
    if len(source_timestamp_ns) == 1:
        rotation_vector = np.repeat(source_pose6[:1, 3:6], len(target_sec), axis=0)
    else:
        rotation_vector = Slerp(source_sec, source_rotation)(clipped_target_sec).as_rotvec()

    before = target_timestamp_ns < source_timestamp_ns[0]
    after = target_timestamp_ns > source_timestamp_ns[-1]
    boundary_gap_ns = np.zeros(len(target_timestamp_ns), dtype=np.int64)
    boundary_gap_ns[before] = source_timestamp_ns[0] - target_timestamp_ns[before]
    boundary_gap_ns[after] = target_timestamp_ns[after] - source_timestamp_ns[-1]
    report = {
        "source_samples": int(len(source_timestamp_ns)),
        "frames_before_source": int(np.count_nonzero(before)),
        "frames_after_source": int(np.count_nonzero(after)),
        "max_boundary_gap_ms": float(np.max(boundary_gap_ns, initial=0) * 1e-6),
    }
    pose = np.concatenate([position, rotation_vector], axis=1).astype(np.float32)
    return pose, report


def flatten_magnet_samples(
    magnet_xyz: np.ndarray,
    magnet_timestamp_ns: np.ndarray,
    magnet_sample_count: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    magnet_xyz = np.asarray(magnet_xyz, dtype=np.float32)
    magnet_timestamp_ns = np.asarray(magnet_timestamp_ns, dtype=np.int64)
    magnet_sample_count = np.asarray(magnet_sample_count, dtype=np.int64).reshape(-1)
    if magnet_xyz.ndim != 4 or magnet_xyz.shape[-1] != 3:
        raise ValueError(f"Expected GELLO magnet_xyz [T,S,N,3], got {magnet_xyz.shape}")
    if magnet_timestamp_ns.shape != magnet_xyz.shape[:2]:
        raise ValueError(
            f"Magnet timestamp shape {magnet_timestamp_ns.shape} does not match "
            f"magnet data {magnet_xyz.shape[:2]}"
        )
    if len(magnet_sample_count) != len(magnet_xyz):
        raise ValueError("Magnet sample count length does not match frame count")

    valid = magnet_timestamp_ns > 0
    sample_indices = np.arange(magnet_xyz.shape[1])[None, :]
    counts = np.clip(magnet_sample_count, 0, magnet_xyz.shape[1])
    valid &= sample_indices >= magnet_xyz.shape[1] - counts[:, None]
    timestamps = magnet_timestamp_ns[valid]
    values = magnet_xyz[valid]
    timestamps, values = unique_time_series(timestamps, values)
    return timestamps, replace_abnormal_magnet_readings(values)


def replace_abnormal_magnet_readings(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return values
    abnormal = np.isfinite(values) & (np.abs(values) > MAGNET_ABNORMAL_ABS_THRESHOLD)
    if not np.any(abnormal):
        return values
    cleaned = values.copy()
    for sensor_idx in range(values.shape[1]):
        for axis_idx in range(values.shape[2]):
            series = values[:, sensor_idx, axis_idx]
            bad_indices = np.flatnonzero(abnormal[:, sensor_idx, axis_idx])
            good_indices = np.flatnonzero(
                ~abnormal[:, sensor_idx, axis_idx] & np.isfinite(series)
            )
            if len(good_indices) == 0:
                continue
            for bad_index in bad_indices:
                nearest = good_indices[np.argmin(np.abs(good_indices - bad_index))]
                cleaned[bad_index, sensor_idx, axis_idx] = series[nearest]
    return cleaned


def align_causal_magnet_windows(
    target_timestamp_ns: np.ndarray,
    source_timestamp_ns: np.ndarray,
    source_magnet_xyz: np.ndarray,
    samples_per_frame: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    target_timestamp_ns = np.asarray(target_timestamp_ns, dtype=np.int64).reshape(-1)
    source_timestamp_ns = np.asarray(source_timestamp_ns, dtype=np.int64).reshape(-1)
    source_magnet_xyz = np.asarray(source_magnet_xyz, dtype=np.float32)
    if source_magnet_xyz.ndim != 3 or source_magnet_xyz.shape[-1] != 3:
        raise ValueError(f"Expected magnet samples [M,N,3], got {source_magnet_xyz.shape}")
    if len(source_timestamp_ns) != len(source_magnet_xyz):
        raise ValueError("Magnet timestamp and sample lengths differ")
    if samples_per_frame <= 0:
        raise ValueError("samples_per_frame must be positive")

    sensor_count = source_magnet_xyz.shape[1]
    aligned_xyz = np.full(
        (len(target_timestamp_ns), samples_per_frame, sensor_count, 3),
        np.nan,
        dtype=np.float32,
    )
    aligned_timestamp_ns = np.zeros(
        (len(target_timestamp_ns), samples_per_frame), dtype=np.int64
    )
    aligned_count = np.zeros((len(target_timestamp_ns), 1), dtype=np.int32)
    latest_age_ns = np.full(len(target_timestamp_ns), -1, dtype=np.int64)

    for frame_index, target_time in enumerate(target_timestamp_ns):
        source_end = int(np.searchsorted(source_timestamp_ns, target_time, side="right"))
        source_start = max(0, source_end - samples_per_frame)
        count = source_end - source_start
        if count == 0:
            continue
        output_start = samples_per_frame - count
        aligned_xyz[frame_index, output_start:] = source_magnet_xyz[source_start:source_end]
        aligned_timestamp_ns[frame_index, output_start:] = source_timestamp_ns[source_start:source_end]
        aligned_count[frame_index, 0] = count
        latest_age_ns[frame_index] = target_time - source_timestamp_ns[source_end - 1]

    valid_age_ms = latest_age_ns[latest_age_ns >= 0].astype(np.float64) * 1e-6
    if len(valid_age_ms):
        age_percentiles = np.percentile(valid_age_ms, [50, 95, 100])
    else:
        age_percentiles = np.array([np.nan, np.nan, np.nan])
    report = {
        "unique_source_samples": int(len(source_timestamp_ns)),
        "frames_without_past_sample": int(np.count_nonzero(aligned_count[:, 0] == 0)),
        "sample_count_min": int(np.min(aligned_count)) if len(aligned_count) else 0,
        "sample_count_max": int(np.max(aligned_count)) if len(aligned_count) else 0,
        "latest_sample_age_p50_ms": float(age_percentiles[0]),
        "latest_sample_age_p95_ms": float(age_percentiles[1]),
        "latest_sample_age_max_ms": float(age_percentiles[2]),
    }
    return aligned_xyz, aligned_timestamp_ns, aligned_count, report


def episode_bounds(episode_ends: np.ndarray, episode_index: int) -> Tuple[int, int]:
    end = int(episode_ends[episode_index])
    start = 0 if episode_index == 0 else int(episode_ends[episode_index - 1])
    return start, end


def calculate_trajectory_error(
    iphone_pose: np.ndarray,
    robot_pose: np.ndarray,
    target_timestamp_ns: np.ndarray,
    robot_timestamp_ns: np.ndarray,
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    iphone_pose = np.asarray(iphone_pose, dtype=np.float64)
    robot_pose = np.asarray(robot_pose, dtype=np.float64)
    target_timestamp_ns = np.asarray(target_timestamp_ns, dtype=np.int64).reshape(-1)
    robot_timestamp_ns = np.asarray(robot_timestamp_ns, dtype=np.int64).reshape(-1)
    if len(iphone_pose) != len(robot_pose) or len(iphone_pose) != len(target_timestamp_ns):
        raise ValueError("Trajectory poses and target timestamps have different lengths")
    robot_time_min = np.min(robot_timestamp_ns)
    robot_time_max = np.max(robot_timestamp_ns)
    overlap = (
        (target_timestamp_ns >= robot_time_min)
        & (target_timestamp_ns <= robot_time_max)
    )
    if np.count_nonzero(overlap) < 2:
        raise ValueError("Fewer than two trajectory frames overlap the robot time range")
    errors = compare_pose_sequences(iphone_pose[overlap], robot_pose[overlap])
    values = {
        "position_error_mm": errors["position_error_mm"][1:],
        "rotation_error_deg": errors["rotation_error_deg"][1:],
        "relative_translation_error_mm": errors["relative_translation_error_mm"],
        "relative_rotation_error_deg": errors["relative_rotation_error_deg"],
    }
    report = {
        "method": (
            "first-frame rigid alignment on timestamp-overlap frames; artificial "
            "zero first-frame error excluded"
        ),
        "overlap_frames": int(np.count_nonzero(overlap)),
        "excluded_outside_robot_time_range": int(np.count_nonzero(~overlap)),
        "first_frame_alignment_matrix": errors["alignment_matrix"].tolist(),
        "first_frame_aligned_pose": {
            "position_error_mm": error_summary(values["position_error_mm"]),
            "rotation_error_deg": error_summary(values["rotation_error_deg"]),
            "endpoint_position_error_mm": float(errors["position_error_mm"][-1]),
            "endpoint_rotation_error_deg": float(errors["rotation_error_deg"][-1]),
        },
        "relative_action": {
            "translation_error_mm": error_summary(
                values["relative_translation_error_mm"]
            ),
            "rotation_error_deg": error_summary(
                values["relative_rotation_error_deg"]
            ),
        },
    }
    return report, values


def summarize_trajectory_errors(
    fusions: Sequence[EpisodeFusion],
) -> Optional[Dict[str, object]]:
    available = [
        fusion.trajectory_error_values
        for fusion in fusions
        if fusion.trajectory_error_values is not None
    ]
    if not available:
        return None

    metric_units = {
        "position_error_mm": "mm",
        "rotation_error_deg": "deg",
        "relative_translation_error_mm": "mm",
        "relative_rotation_error_deg": "deg",
    }
    episode_equal_mean = {}
    frame_weighted = {}
    for metric, unit in metric_units.items():
        episode_means = np.asarray(
            [np.mean(values[metric]) for values in available], dtype=np.float64
        )
        episode_equal_mean[metric] = {
            "unit": unit,
            "episode_count": int(len(episode_means)),
            "mean": float(np.mean(episode_means)),
            "median_episode_mean": float(np.median(episode_means)),
            "max_episode_mean": float(np.max(episode_means)),
        }
        frame_weighted[metric] = {
            "unit": unit,
            **error_summary(
                np.concatenate([values[metric] for values in available], axis=0)
            ),
        }
    return {
        "method": (
            "iPhone trajectory first-frame aligned to timestamp-interpolated GELLO "
            "robot observations; only timestamp-overlap frames are included"
        ),
        "episode_equal_mean": episode_equal_mean,
        "frame_weighted": frame_weighted,
    }


def print_trajectory_error_summary(
    fusions: Sequence[EpisodeFusion], summary: Optional[Dict[str, object]]
) -> None:
    if summary is None:
        print("Trajectory error: unavailable because GELLO robot poses are missing")
        return
    print("Trajectory errors by episode:")
    for fusion in fusions:
        trajectory_error = fusion.report.get("trajectory_error")
        if trajectory_error is None:
            continue
        pose = trajectory_error["first_frame_aligned_pose"]
        relative = trajectory_error["relative_action"]
        print(
            f"  {fusion.match.arpose_name}: "
            f"position mean={pose['position_error_mm']['mean']:.3f} mm, "
            f"rotation mean={pose['rotation_error_deg']['mean']:.3f} deg, "
            f"relative translation mean="
            f"{relative['translation_error_mm']['mean']:.3f} mm, "
            f"relative rotation mean="
            f"{relative['rotation_error_deg']['mean']:.3f} deg"
        )
    equal = summary["episode_equal_mean"]
    weighted = summary["frame_weighted"]
    print("Average trajectory error (each episode weighted equally):")
    print(
        f"  position={equal['position_error_mm']['mean']:.3f} mm, "
        f"rotation={equal['rotation_error_deg']['mean']:.3f} deg, "
        f"relative translation="
        f"{equal['relative_translation_error_mm']['mean']:.3f} mm, "
        f"relative rotation="
        f"{equal['relative_rotation_error_deg']['mean']:.3f} deg"
    )
    print("Frame-weighted trajectory error:")
    print(
        f"  position mean={weighted['position_error_mm']['mean']:.3f} mm, "
        f"rotation mean={weighted['rotation_error_deg']['mean']:.3f} deg, "
        f"relative translation mean="
        f"{weighted['relative_translation_error_mm']['mean']:.3f} mm, "
        f"relative rotation mean="
        f"{weighted['relative_rotation_error_deg']['mean']:.3f} deg"
    )


def fuse_episode(
    match: EpisodeMatch,
    episode_ends: np.ndarray,
    arpose_timestamps_sec: np.ndarray,
    arpose_pose6: np.ndarray,
    gello_dir: Path,
    gripper_stroke_m: float,
    magnet_samples_per_frame: int,
    robot_data_source: str,
    gripper_value_mode: str,
    clock_offset: Dict[str, object],
) -> EpisodeFusion:
    source_start, source_end = episode_bounds(episode_ends, match.arpose_index)
    iphone_timestamp_ns = np.rint(
        np.asarray(arpose_timestamps_sec[source_start:source_end], dtype=np.float64) * 1e9
    ).astype(np.int64)
    target_timestamp_ns = apply_clock_offset_ns(
        iphone_timestamp_ns, float(clock_offset["offset_ms"])
    )
    gello = load_gello_episode(gello_dir)
    gripper_value, gripper_report = interpolate_gripper_values(
        target_timestamp_ns,
        gello["gripper_timestamp_ns"],
        gello["gripper_raw"],
        value_mode=gripper_value_mode,
        stroke_m=gripper_stroke_m,
    )
    magnet_time, magnet_values = flatten_magnet_samples(
        gello["magnet_xyz"],
        gello["magnet_timestamp_ns"],
        gello["magnet_sample_count"],
    )
    magnet_xyz, magnet_timestamp_ns, magnet_sample_count, magnet_report = (
        align_causal_magnet_windows(
            target_timestamp_ns,
            magnet_time,
            magnet_values,
            magnet_samples_per_frame,
        )
    )
    robot_pose = None
    action = None
    robot_report = None
    trajectory_error_report = None
    trajectory_error_values = None
    actual_pose = None
    if "robot0_eef_pos" in gello and "robot0_eef_rot_axis_angle" in gello:
        actual_pose = np.concatenate(
            [gello["robot0_eef_pos"], gello["robot0_eef_rot_axis_angle"]], axis=1
        )
        robot_pose, observation_report = interpolate_pose6(
            target_timestamp_ns,
            gello["gripper_timestamp_ns"],
            actual_pose,
        )
        trajectory_error_report, trajectory_error_values = calculate_trajectory_error(
            arpose_pose6[source_start:source_end],
            robot_pose,
            target_timestamp_ns,
            gello["gripper_timestamp_ns"],
        )
    if robot_data_source == "gello":
        required_robot_fields = (
            "robot0_eef_pos",
            "robot0_eef_rot_axis_angle",
            "action",
        )
        missing = [key for key in required_robot_fields if key not in gello]
        if missing:
            raise KeyError(
                f"GELLO episode {gello_dir} cannot provide robot data; missing {missing}"
            )
        if actual_pose is None or robot_pose is None:
            raise KeyError(f"GELLO episode {gello_dir} has no robot observations")
        command_action = np.asarray(gello["action"], dtype=np.float32)
        if command_action.ndim != 2 or command_action.shape[1] != 7:
            raise ValueError(
                f"Expected GELLO action [T,7], got {command_action.shape} in {gello_dir}"
            )
        command_pose, action_report = interpolate_pose6(
            target_timestamp_ns,
            gello["gripper_timestamp_ns"],
            command_action[:, :6],
        )
        action = np.concatenate([command_pose, gripper_value], axis=1).astype(
            np.float32
        )
        robot_report = {
            "observation": observation_report,
            "action": action_report,
            "observation_source": "GELLO recorder actual TCP",
            "action_source": "GELLO recorder commanded target TCP",
        }
    elif robot_data_source != "arpose":
        raise ValueError(f"Unsupported robot data source: {robot_data_source}")

    report = {
        **asdict(match),
        "arpose_frames": int(source_end - source_start),
        "gello_frames": int(len(gello["gripper_raw"])),
        "iphone_start_unix_sec": float(iphone_timestamp_ns[0] * 1e-9),
        "iphone_end_unix_sec": float(iphone_timestamp_ns[-1] * 1e-9),
        "aligned_pc_start_unix_sec": float(target_timestamp_ns[0] * 1e-9),
        "aligned_pc_end_unix_sec": float(target_timestamp_ns[-1] * 1e-9),
        "gello_start_unix_sec": float(gello["gripper_timestamp_ns"][0] * 1e-9),
        "gello_end_unix_sec": float(gello["gripper_timestamp_ns"][-1] * 1e-9),
        "iphone_to_pc_clock_offset": clock_offset,
        "gripper": gripper_report,
        "magnet": magnet_report,
        "robot": robot_report,
        "trajectory_error": trajectory_error_report,
    }
    return EpisodeFusion(
        match=match,
        source_start=source_start,
        source_end=source_end,
        gripper_value=gripper_value,
        magnet_xyz=magnet_xyz,
        magnet_timestamp_ns=magnet_timestamp_ns,
        magnet_sample_count=magnet_sample_count,
        report=report,
        aligned_timestamp_sec=target_timestamp_ns.astype(np.float64) * 1e-9,
        robot_pose=robot_pose,
        action=action,
        trajectory_error_values=trajectory_error_values,
    )


def selected_row_indices(fusions: Sequence[EpisodeFusion]) -> np.ndarray:
    parts = [
        np.arange(fusion.source_start, fusion.source_end, dtype=np.int64)
        for fusion in fusions
    ]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)


def next_observation_indices(episode_lengths: Iterable[int]) -> np.ndarray:
    parts = []
    offset = 0
    for length_value in episode_lengths:
        length = int(length_value)
        indices = np.arange(offset, offset + length, dtype=np.int64)
        if length > 1:
            indices[:-1] += 1
        parts.append(indices)
        offset += length
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)


def build_compatible_magnetic_fields(
    magnet_xyz: np.ndarray,
    magnet_sample_count: np.ndarray,
    chip_count: int = 5,
) -> Dict[str, np.ndarray]:
    magnet_xyz = np.asarray(magnet_xyz, dtype=np.float32)
    magnet_sample_count = np.asarray(magnet_sample_count, dtype=np.int64).reshape(-1)
    if magnet_xyz.ndim != 4 or magnet_xyz.shape[-1] != 3:
        raise ValueError(f"Expected aligned magnet data [T,S,N,3], got {magnet_xyz.shape}")
    if len(magnet_xyz) != len(magnet_sample_count):
        raise ValueError("Aligned magnet data and sample counts have different lengths")
    if magnet_xyz.shape[2] > chip_count:
        raise ValueError(
            f"Aligned magnet has {magnet_xyz.shape[2]} sensors, but compatibility "
            f"fields only support {chip_count} chips"
        )

    magnetic_txyz = np.zeros((len(magnet_xyz), chip_count, 4), dtype=np.float32)
    magnetic_valid = np.zeros((len(magnet_xyz), chip_count), dtype=bool)
    for frame_index, count_value in enumerate(magnet_sample_count):
        count = min(max(int(count_value), 0), magnet_xyz.shape[1])
        if count == 0:
            continue
        latest = magnet_xyz[frame_index, -1]
        finite_sensor = np.all(np.isfinite(latest), axis=1)
        sensor_count = magnet_xyz.shape[2]
        magnetic_txyz[frame_index, :sensor_count, 1:4] = np.nan_to_num(
            latest, nan=0.0
        )
        magnetic_valid[frame_index, :sensor_count] = finite_sensor

    return {
        "magnetic_txyz": magnetic_txyz,
        "magnetic_valid": magnetic_valid,
        "magnetic_left_txyz": np.zeros_like(magnetic_txyz),
        "magnetic_left_valid": np.zeros_like(magnetic_valid),
    }


def default_sidecar_path(output: Path, suffix: str) -> Path:
    name = output.name[:-5] if output.name.endswith(".zarr") else output.stem
    return output.parent / f"{name}{suffix}"


def check_output_paths(
    output: Path,
    report_output: Path,
    video_output: Path,
    write_video: bool,
    overwrite: bool,
) -> None:
    paths = [output, report_output]
    if write_video:
        paths.append(video_output)
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Output path(s) already exist: {formatted}; pass --overwrite to replace them"
        )


def main() -> int:
    args = parse_args()
    arpose_input = args.arpose_input.expanduser().resolve()
    gello_input = args.gello_input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    video_output = (
        args.video_output.expanduser().resolve()
        if args.video_output is not None
        else default_sidecar_path(output, "_videos")
    )
    report_output = (
        args.report_output.expanduser().resolve()
        if args.report_output is not None
        else default_sidecar_path(output, "_fusion_report.json")
    )
    if output == arpose_input:
        raise ValueError("--output must differ from --arpose-input; source data is never modified")
    if not arpose_input.is_dir():
        raise FileNotFoundError(f"ARPose zarr not found: {arpose_input}")

    source = zarr.open(str(arpose_input), mode="r")
    action_source = str(source.attrs.get("action_source", ""))
    if args.robot_data_source == "arpose" and action_source != "next_obs":
        raise ValueError(
            "This fusion script rebuilds the gripper action with next-observation "
            f"semantics, but source action_source is {action_source!r}"
        )
    if "source_directories" not in source.attrs:
        raise KeyError("ARPose zarr attrs are missing source_directories")
    arpose_names = [str(name) for name in source.attrs["source_directories"]]
    gello_dirs = list_gello_episode_dirs(gello_input)
    gello_by_name = {path.name: path for path in gello_dirs}
    matches, unmatched_arpose, unmatched_gello = match_episode_names(
        arpose_names,
        list(gello_by_name),
        args.episode_match_tolerance_sec,
    )
    if not matches:
        raise ValueError("No ARPose/GELLO episodes matched by recording time")
    if args.require_all_matched and (unmatched_arpose or unmatched_gello):
        raise ValueError(
            "Strict matching failed: "
            f"unmatched ARPose={unmatched_arpose}, unmatched GELLO={unmatched_gello}"
        )
    excluded_arpose = set(args.exclude_arpose_episode)
    unknown_exclusions = sorted(excluded_arpose.difference(arpose_names))
    if unknown_exclusions:
        raise ValueError(
            f"Excluded ARPose episodes are not present in the input: {unknown_exclusions}"
        )
    excluded_matches = [
        match for match in matches if match.arpose_name in excluded_arpose
    ]
    matches = [match for match in matches if match.arpose_name not in excluded_arpose]
    for match in excluded_matches:
        print(f"[EXCLUDE] {match.arpose_name} <- {match.gello_name}")
    if not matches:
        raise ValueError("No matched episodes remain after exclusions")

    episode_ends = np.asarray(source["meta/episode_ends"][:], dtype=np.int64)
    if len(episode_ends) != len(arpose_names):
        raise ValueError(
            f"source_directories has {len(arpose_names)} entries but episode_ends has "
            f"{len(episode_ends)}"
        )
    arpose_timestamps = np.asarray(source["data/timestamp"][:], dtype=np.float64)
    required_arpose_pose_keys = (
        "robot0_eef_pos",
        "robot0_eef_rot_axis_angle",
    )
    missing_arpose_pose_keys = [
        key for key in required_arpose_pose_keys if key not in source["data"]
    ]
    if missing_arpose_pose_keys:
        raise KeyError(
            f"ARPose data is missing pose keys required for trajectory error: "
            f"{missing_arpose_pose_keys}"
        )
    arpose_pose6 = np.concatenate(
        [
            np.asarray(source["data/robot0_eef_pos"][:], dtype=np.float64),
            np.asarray(
                source["data/robot0_eef_rot_axis_angle"][:], dtype=np.float64
            ),
        ],
        axis=1,
    )
    fusions = []
    for match in matches:
        print(f"[ALIGN] {match.arpose_name} <- {match.gello_name}")
        clock_offset = resolve_iphone_to_pc_clock_offset(
            arpose_input,
            match.arpose_name,
            args.time_alignment,
            args.iphone_clock_offset_ms,
        )
        print(
            f"[CLOCK] {match.arpose_name}: "
            f"{clock_offset['offset_ms']:+.3f} ms "
            f"[{clock_offset['method']}] ({clock_offset['source']})"
        )
        fusions.append(
            fuse_episode(
                match,
                episode_ends,
                arpose_timestamps,
                arpose_pose6,
                gello_by_name[match.gello_name],
                args.gripper_stroke_m,
                args.magnet_samples_per_frame,
                args.robot_data_source,
                args.gripper_value_mode,
                clock_offset,
            )
        )

    trajectory_error_summary = summarize_trajectory_errors(fusions)
    report = {
        "arpose_input": str(arpose_input),
        "gello_input": str(gello_input),
        "output": str(output),
        "robot_data_source": args.robot_data_source,
        "time_alignment": args.time_alignment,
        "timestamp_clock_domain": (
            "pc_wall_clock" if args.time_alignment == "receiver_clock" else "iphone_wall_clock"
        ),
        "episode_match_tolerance_sec": args.episode_match_tolerance_sec,
        "gripper_stroke_m": args.gripper_stroke_m,
        "gripper_value_mode": args.gripper_value_mode,
        "magnet_samples_per_frame": args.magnet_samples_per_frame,
        "matched_episode_count": len(fusions),
        "excluded_arpose": sorted(excluded_arpose),
        "excluded_matches": [asdict(match) for match in excluded_matches],
        "unmatched_arpose": unmatched_arpose,
        "unmatched_gello": unmatched_gello,
        "trajectory_error_summary": trajectory_error_summary,
        "episodes": [fusion.report for fusion in fusions],
    }
    print(
        f"Matched {len(fusions)} episode(s); unmatched ARPose={unmatched_arpose}; "
        f"unmatched GELLO={unmatched_gello}"
    )
    print_trajectory_error_summary(fusions, trajectory_error_summary)
    if args.dry_run:
        print(json.dumps(report, indent=2))
        print("Dry run only; no output was written.")
        return 0

    check_output_paths(
        output,
        report_output,
        video_output,
        write_video=not args.no_video,
        overwrite=args.overwrite,
    )
    write_fused_zarr(source, output, fusions, report, args.overwrite)
    write_json(report_output, report, args.overwrite)
    if not args.no_video:
        write_visualizations(
            output,
            video_output,
            [fusion.match.arpose_name for fusion in fusions],
            args.video_fps,
            args.video_panel_width,
            args.overwrite,
        )
    if args.validate:
        validate_output(source, output, fusions)
    print(f"Wrote fused zarr: {output}")
    print(f"Wrote alignment report: {report_output}")
    if not args.no_video:
        print(f"Wrote synchronized videos: {video_output}")
    return 0


def write_fused_zarr(
    source: zarr.hierarchy.Group,
    output: Path,
    fusions: Sequence[EpisodeFusion],
    report: Dict[str, object],
    overwrite: bool,
) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {output}; pass --overwrite to replace it")
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    destination = zarr.group(store=zarr.DirectoryStore(str(output)), overwrite=True)
    destination_attrs = dict(source.attrs)
    destination_attrs.pop("magnet_board_order", None)
    destination_attrs.pop("magnet_source_key", None)
    destination.attrs.update(destination_attrs)
    destination.attrs.update(
        {
            "source_directories": [fusion.match.arpose_name for fusion in fusions],
            "gello_source_directories": [fusion.match.gello_name for fusion in fusions],
            "robot_data_source": str(report.get("robot_data_source", "arpose")),
            "time_alignment": str(report.get("time_alignment", "none")),
            "timestamp_clock_domain": str(
                report.get("timestamp_clock_domain", "iphone_wall_clock")
            ),
            "timestamp_source": (
                "iphone_sender_time_plus_receiver_clock_offset"
                if report.get("time_alignment") == "receiver_clock"
                else "iphone_sender_time_uncorrected"
            ),
            "gripper_value_mode": str(report.get("gripper_value_mode", "width_m")),
            "gello_gripper_stroke_m": float(report["gripper_stroke_m"]),
            "magnet_key": "magnet_xyz",
            "magnet_source": "gello_magnet_xyz_aligned_by_magnet_timestamp_ns",
            "magnet_axis_order": ["frame", "time_sample", "sensor", "xyz"],
            "magnet_alignment": "latest causal samples at or before each ARPose frame",
            "magnet_samples_per_frame": int(report["magnet_samples_per_frame"]),
            "magnetic_txyz_source": (
                "latest_gello_magnet_sample_with_unavailable_temperature_set_to_zero"
            ),
            "magnetic_left_txyz_source": "zero_unavailable",
            "fusion_report": report,
        }
    )
    robot_data_source = str(report.get("robot_data_source", "arpose"))
    gripper_value_mode = str(report.get("gripper_value_mode", "width_m"))
    if gripper_value_mode == "raw_ratio":
        destination.attrs.update(
            {
                "gripper_width_source": "gello_robot0_gripper_width_raw_ratio_unmodified",
                "gripper_width_unit": "open_ratio",
            }
        )
    elif gripper_value_mode == "width_m":
        destination.attrs.update(
            {
                "gripper_width_source": "gello_robot0_gripper_width_raw_ratio_times_stroke",
                "gripper_width_unit": "m",
            }
        )
    else:
        raise ValueError(f"Unsupported gripper value mode: {gripper_value_mode}")
    if robot_data_source == "gello":
        destination.attrs.update(
            {
                "eef_pose_source": "gello_actual_tcp_timestamp_aligned",
                "action_source": "gello_command_timestamp_aligned",
            }
        )
    output_data = destination.create_group("data")
    output_meta = destination.create_group("meta")
    output_data.attrs.update(dict(source["data"].attrs))
    output_meta.attrs.update(dict(source["meta"].attrs))

    row_indices = selected_row_indices(fusions)
    lengths = [fusion.source_end - fusion.source_start for fusion in fusions]
    total_rows = int(sum(lengths))
    replacement = {
        "timestamp": np.concatenate(
            [
                fusion.aligned_timestamp_sec
                if fusion.aligned_timestamp_sec is not None
                else np.asarray(
                    source["data/timestamp"][fusion.source_start:fusion.source_end],
                    dtype=np.float64,
                )
                for fusion in fusions
            ],
            axis=0,
        ),
        "robot0_gripper_width": np.concatenate(
            [fusion.gripper_value for fusion in fusions], axis=0
        ),
        "magnet_xyz": np.concatenate([fusion.magnet_xyz for fusion in fusions], axis=0),
        "magnet_timestamp_ns": np.concatenate(
            [fusion.magnet_timestamp_ns for fusion in fusions], axis=0
        ),
        "magnet_sample_count": np.concatenate(
            [fusion.magnet_sample_count for fusion in fusions], axis=0
        ),
    }
    replacement.update(
        build_compatible_magnetic_fields(
            replacement["magnet_xyz"], replacement["magnet_sample_count"]
        )
    )
    if robot_data_source == "gello":
        if any(fusion.robot_pose is None or fusion.action is None for fusion in fusions):
            raise ValueError("GELLO robot data source selected but aligned robot data is missing")
        robot_pose = np.concatenate([fusion.robot_pose for fusion in fusions], axis=0)
        replacement["robot0_eef_pos"] = robot_pose[:, :3]
        replacement["robot0_eef_rot_axis_angle"] = robot_pose[:, 3:6]
        replacement["action"] = np.concatenate(
            [fusion.action for fusion in fusions], axis=0
        )
        replacement["robot0_demo_start_pose"] = np.concatenate(
            [
                np.repeat(fusion.robot_pose[:1], length, axis=0)
                for fusion, length in zip(fusions, lengths)
            ],
            axis=0,
        )
        replacement["robot0_demo_end_pose"] = np.concatenate(
            [
                np.repeat(fusion.robot_pose[-1:], length, axis=0)
                for fusion, length in zip(fusions, lengths)
            ],
            axis=0,
        )
    elif robot_data_source == "arpose":
        action = read_rows(source["data/action"], row_indices).copy()
        if action.ndim != 2 or action.shape[1] < 7:
            raise ValueError(f"Expected source data/action [T,>=7], got {action.shape}")
        action[:, 6:7] = replacement["robot0_gripper_width"][
            next_observation_indices(lengths)
        ]
        replacement["action"] = action
    else:
        raise ValueError(f"Unsupported robot data source: {robot_data_source}")

    for key, source_array in source["data"].arrays():
        values = replacement.get(key)
        shape = (total_rows,) + (
            tuple(values.shape[1:]) if values is not None else tuple(source_array.shape[1:])
        )
        chunks = (min(source_array.chunks[0], total_rows),) + tuple(shape[1:])
        output_array = output_data.create_dataset(
            key,
            shape=shape,
            dtype=values.dtype if values is not None else source_array.dtype,
            chunks=chunks,
            compressor=source_array.compressor,
            filters=source_array.filters,
            fill_value=source_array.fill_value,
            order=source_array.order,
            overwrite=True,
        )
        output_array.attrs.update(dict(source_array.attrs))
        if values is not None:
            output_array[:] = values
        else:
            copy_selected_rows(source_array, output_array, row_indices)
        print(f"[WRITE] data/{key}: {shape}")

    matched_episode_indices = np.asarray(
        [fusion.match.arpose_index for fusion in fusions], dtype=np.int64
    )
    new_episode_ends = np.cumsum(lengths, dtype=np.int64)
    for key, source_array in source["meta"].arrays():
        if key == "episode_ends":
            values = new_episode_ends.astype(source_array.dtype, copy=False)
        elif (
            source_array.ndim > 0
            and source_array.shape[0] == len(source.attrs["source_directories"])
        ):
            values = read_rows(source_array, matched_episode_indices)
        else:
            values = source_array[:]
        chunks = tuple(
            min(chunk, size) for chunk, size in zip(source_array.chunks, values.shape)
        )
        output_array = output_meta.create_dataset(
            key,
            data=values,
            dtype=values.dtype,
            chunks=chunks,
            compressor=source_array.compressor,
            filters=source_array.filters,
            fill_value=source_array.fill_value,
            order=source_array.order,
            overwrite=True,
        )
        output_array.attrs.update(dict(source_array.attrs))


def read_rows(array: zarr.core.Array, row_indices: np.ndarray) -> np.ndarray:
    row_indices = np.asarray(row_indices, dtype=np.int64)
    if len(row_indices) == 0:
        return array[:0]
    selection = (row_indices,) + tuple(slice(None) for _ in array.shape[1:])
    return array.get_orthogonal_selection(selection)


def copy_selected_rows(
    source: zarr.core.Array,
    destination: zarr.core.Array,
    row_indices: np.ndarray,
    batch_rows: int = 256,
) -> None:
    for start in range(0, len(row_indices), batch_rows):
        end = min(start + batch_rows, len(row_indices))
        destination[start:end] = read_rows(source, row_indices[start:end])


def write_json(path: Path, values: Dict[str, object], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Report exists: {path}; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")


def write_visualizations(
    output: Path,
    video_output: Path,
    episode_names: Sequence[str],
    fallback_fps: float,
    panel_width: int,
    overwrite: bool,
) -> None:
    if video_output.exists():
        if not overwrite:
            raise FileExistsError(
                f"Video output exists: {video_output}; pass --overwrite to replace it"
            )
        shutil.rmtree(video_output)
    video_output.mkdir(parents=True, exist_ok=True)
    root = zarr.open(str(output), mode="r")
    episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    for episode_index, episode_name in enumerate(episode_names):
        start, end = episode_bounds(episode_ends, episode_index)
        video_path = video_output / f"{episode_name}_camera0_magnet.mp4"
        timestamps = np.asarray(root["data/timestamp"][start:end])
        if len(timestamps):
            timestamps = timestamps - timestamps[0]
        save_rgb_magnet_video(
            frames=np.asarray(root["data/camera0_rgb"][start:end]),
            magnet_xyz=np.asarray(root["data/magnet_xyz"][start:end]),
            timestamps=timestamps,
            video_path=video_path,
            fallback_fps=fallback_fps,
            panel_width=panel_width,
        )
        print(f"[VIDEO] {video_path}")


def validate_output(
    source: zarr.hierarchy.Group,
    output: Path,
    fusions: Sequence[EpisodeFusion],
) -> None:
    root = zarr.open(str(output), mode="r")
    lengths = [fusion.source_end - fusion.source_start for fusion in fusions]
    expected_rows = int(sum(lengths))
    expected_ends = np.cumsum(lengths, dtype=np.int64)
    if not np.array_equal(root["meta/episode_ends"][:], expected_ends):
        raise AssertionError("Output meta/episode_ends is incorrect")
    if sorted(root["data"].keys()) != sorted(source["data"].keys()):
        raise AssertionError("Output data keys differ from source")
    for key, output_array in root["data"].arrays():
        if output_array.shape[0] != expected_rows:
            raise AssertionError(f"Output data/{key} has wrong row count")
    expected_timestamp = np.concatenate(
        [
            fusion.aligned_timestamp_sec
            if fusion.aligned_timestamp_sec is not None
            else np.asarray(
                source["data/timestamp"][fusion.source_start:fusion.source_end],
                dtype=np.float64,
            )
            for fusion in fusions
        ],
        axis=0,
    )
    if not np.allclose(
        root["data/timestamp"][:], expected_timestamp, rtol=0.0, atol=1e-7
    ):
        raise AssertionError("Output timestamps differ from clock-aligned timestamps")
    gripper = np.concatenate([fusion.gripper_value for fusion in fusions], axis=0)
    if not np.allclose(root["data/robot0_gripper_width"][:], gripper):
        raise AssertionError("Output gripper values differ from aligned GELLO values")
    robot_data_source = str(root.attrs.get("robot_data_source", "arpose"))
    if robot_data_source == "gello":
        expected_pose = np.concatenate([fusion.robot_pose for fusion in fusions], axis=0)
        expected_action = np.concatenate([fusion.action for fusion in fusions], axis=0)
        actual_pose = np.concatenate(
            [
                root["data/robot0_eef_pos"][:],
                root["data/robot0_eef_rot_axis_angle"][:],
            ],
            axis=1,
        )
        if not np.allclose(actual_pose, expected_pose):
            raise AssertionError("Output robot poses differ from aligned GELLO poses")
        if not np.allclose(root["data/action"][:], expected_action):
            raise AssertionError("Output actions differ from aligned GELLO commands")
        if str(root.attrs.get("action_source", "")) != "gello_command_timestamp_aligned":
            raise AssertionError("Output action_source does not describe GELLO commands")
    else:
        next_indices = next_observation_indices(lengths)
        if not np.allclose(root["data/action"][:, 6:7], gripper[next_indices]):
            raise AssertionError(
                "Output action gripper column does not preserve next_obs semantics"
            )
    magnet_timestamp = np.asarray(root["data/magnet_timestamp_ns"][:], dtype=np.int64)
    frame_timestamp = np.rint(np.asarray(root["data/timestamp"][:]) * 1e9).astype(np.int64)
    if np.any(magnet_timestamp > frame_timestamp[:, None]):
        raise AssertionError("Output contains future magnet samples")
    print("Validation passed.")


if __name__ == "__main__":
    raise SystemExit(main())
