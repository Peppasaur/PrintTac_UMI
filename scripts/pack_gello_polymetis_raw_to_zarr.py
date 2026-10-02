#!/usr/bin/env python3
"""Pack raw GELLO/Polymetis recording frames into a UMI-style zarr zip.

The raw frames are written by scripts/gello_polymetis_tcp_teleop.py when
--record-output-dir is enabled. This packer creates the source zarr expected by
scripts/convert_umi_traj_to_rdp_dataset.py. In particular, data/action is the
commanded end-effector pose, not the next measured observation.
"""

from __future__ import annotations

import argparse
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import zarr
import cv2
from numcodecs import Blosc


REQUIRED_FRAME_KEYS = (
    "robot0_eef_pos",
    "robot0_eef_rot_axis_angle",
    "robot0_gripper_width",
    "action",
    "magnet_xyz",
    "magnet_timestamp_ns",
    "magnet_sample_count",
)
SECONDARY_MAGNET_KEYS = (
    "magnet2_xyz",
    "magnet2_timestamp_ns",
    "magnet2_sample_count",
)

MAGNET_ABNORMAL_ABS_THRESHOLD = 5000.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pack raw GELLO/Polymetis command-action episodes into a zarr zip.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default="data/gello_polymetis_command_raw")
    parser.add_argument("--output", default="dataset/traj.zarr.zip")
    parser.add_argument("--min-frames", type=int, default=10)
    parser.add_argument("--video-output", default=None)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square image size after the convert_zarr.py-compatible center crop.",
    )
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--video-panel-width", type=int, default=420)
    parser.add_argument("--no-trim-static", action="store_true")
    parser.add_argument(
        "--static-start-pos-threshold",
        type=float,
        default=1e-4,
        help="Position threshold in meters for trimming static frames at episode start and middle.",
    )
    parser.add_argument(
        "--static-start-rot-threshold",
        type=float,
        default=1e-3,
        help="Rotation threshold in radians for trimming static frames at episode start and middle.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args()


def list_episode_dirs(input_dir: Path, min_frames: int) -> List[Path]:
    episodes = []
    for child in sorted(input_dir.expanduser().iterdir()):
        if not child.is_dir():
            continue
        frame_count = len(list(child.glob("frame_*.pkl")))
        if frame_count >= min_frames:
            episodes.append(child)
        else:
            print(f"[WARN] Skipping {child}: only {frame_count} frames")
    if not episodes:
        raise FileNotFoundError(f"No valid episode dirs found in {input_dir}")
    return episodes


def load_frame(path: Path) -> Dict[str, np.ndarray]:
    with open(path, "rb") as f:
        frame = pickle.load(f)
    missing = [key for key in REQUIRED_FRAME_KEYS if key not in frame]
    if missing:
        raise KeyError(f"{path} is missing required keys: {missing}")
    return frame


def stack_optional(frames: List[Dict[str, np.ndarray]], key: str):
    present = [key in frame for frame in frames]
    if not any(present):
        return None
    if not all(present):
        missing_indices = [idx for idx, value in enumerate(present) if not value]
        raise KeyError(
            f"Optional key {key!r} is missing from frame indices {missing_indices[:10]}"
        )
    return np.asarray([frame[key] for frame in frames])


def add_optional_secondary_magnet_data(data, frames):
    values = {key: stack_optional(frames, key) for key in SECONDARY_MAGNET_KEYS}
    present_keys = [key for key, value in values.items() if value is not None]
    if present_keys and len(present_keys) != len(SECONDARY_MAGNET_KEYS):
        missing = [key for key, value in values.items() if value is None]
        raise KeyError(
            "Secondary magnetometer fields must be recorded together; "
            f"missing {missing}"
        )
    if not present_keys:
        return
    data["magnet2_xyz"] = values["magnet2_xyz"].astype(np.float32)
    data["magnet2_timestamp_ns"] = values["magnet2_timestamp_ns"].astype(np.int64)
    data["magnet2_sample_count"] = values["magnet2_sample_count"].astype(
        np.int32
    ).reshape(-1, 1)


def preprocess_rgb(rgb: np.ndarray, image_size: int) -> np.ndarray:
    """Match gello_software/experiments/convert_zarr.py preprocessing exactly."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {rgb.shape}")

    h, w = rgb.shape[:2]
    sq = min(h, w)
    y0 = (h - sq) // 2
    x0 = (w - sq) // 2
    rgb = rgb[y0:y0 + sq, x0:x0 + sq]
    rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return rgb.astype(np.uint8)


def replace_abnormal_magnet_readings(magnet_xyz: np.ndarray) -> tuple[np.ndarray, int]:
    if len(magnet_xyz) == 0:
        return magnet_xyz, 0

    abnormal_mask = (
        np.isfinite(magnet_xyz)
        & (np.abs(magnet_xyz) > MAGNET_ABNORMAL_ABS_THRESHOLD)
    )
    abnormal_indices = np.argwhere(abnormal_mask)
    if abnormal_indices.size == 0:
        return magnet_xyz, 0

    cleaned = magnet_xyz.copy()
    for frame_idx, sample_idx, sensor_idx, axis_idx in abnormal_indices:
        normal_frame_indices = np.flatnonzero(
            ~abnormal_mask[:, sample_idx, sensor_idx, axis_idx]
            & np.isfinite(magnet_xyz[:, sample_idx, sensor_idx, axis_idx])
        )
        if normal_frame_indices.size == 0:
            continue
        nearest_normal_frame_idx = normal_frame_indices[
            np.argmin(np.abs(normal_frame_indices - frame_idx))
        ]
        cleaned[frame_idx, sample_idx, sensor_idx, axis_idx] = magnet_xyz[
            nearest_normal_frame_idx, sample_idx, sensor_idx, axis_idx
        ]

    replaced_count = int(np.count_nonzero(cleaned != magnet_xyz))
    return cleaned, replaced_count


def find_static_start_frame_count(
    eef_pos_arr: np.ndarray,
    eef_rot_arr: np.ndarray,
    pos_threshold: float,
    rot_threshold: float,
) -> int:
    pos_delta = np.linalg.norm(eef_pos_arr - eef_pos_arr[0], axis=-1)
    rot_delta = np.linalg.norm(eef_rot_arr - eef_rot_arr[0], axis=-1)
    moving_indices = np.flatnonzero(
        (pos_delta > pos_threshold) | (rot_delta > rot_threshold)
    )
    if len(moving_indices) == 0:
        return len(eef_pos_arr)
    return int(moving_indices[0])


def find_nonstatic_frame_mask(
    eef_pos_arr: np.ndarray,
    eef_rot_arr: np.ndarray,
    pos_threshold: float,
    rot_threshold: float,
) -> np.ndarray:
    if len(eef_pos_arr) <= 2:
        return np.ones(len(eef_pos_arr), dtype=bool)

    pos_step = np.linalg.norm(np.diff(eef_pos_arr, axis=0), axis=-1)
    rot_step = np.linalg.norm(np.diff(eef_rot_arr, axis=0), axis=-1)
    moving_step = (pos_step > pos_threshold) | (rot_step > rot_threshold)

    keep_mask = np.zeros(len(eef_pos_arr), dtype=bool)
    keep_mask[0] = True
    keep_mask[-1] = True
    keep_mask[:-1] |= moving_step
    keep_mask[1:] |= moving_step
    return keep_mask


def apply_frame_mask(data: Dict[str, np.ndarray], keep_mask: np.ndarray) -> Dict[str, np.ndarray]:
    n = len(keep_mask)
    out = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray) and value.shape[:1] == (n,):
            out[key] = value[keep_mask]
        else:
            out[key] = value
    return out


def trim_static_frames(
    episode_dir: Path,
    data: Dict[str, np.ndarray],
    pos_threshold: float,
    rot_threshold: float,
) -> Dict[str, np.ndarray]:
    eef_pos_arr = data["robot0_eef_pos"]
    eef_rot_arr = data["robot0_eef_rot_axis_angle"]
    trim_start = find_static_start_frame_count(
        eef_pos_arr=eef_pos_arr,
        eef_rot_arr=eef_rot_arr,
        pos_threshold=pos_threshold,
        rot_threshold=rot_threshold,
    )
    if trim_start >= len(eef_pos_arr):
        raise ValueError(f"Episode {episode_dir} has no detected robot movement after static trimming.")
    if trim_start > 0:
        print(f"[WARN] Trimmed {trim_start} static start frames from {episode_dir}")
        keep = np.zeros(len(eef_pos_arr), dtype=bool)
        keep[trim_start:] = True
        data = apply_frame_mask(data, keep)
        eef_pos_arr = data["robot0_eef_pos"]
        eef_rot_arr = data["robot0_eef_rot_axis_angle"]

    nonstatic_mask = find_nonstatic_frame_mask(
        eef_pos_arr=eef_pos_arr,
        eef_rot_arr=eef_rot_arr,
        pos_threshold=pos_threshold,
        rot_threshold=rot_threshold,
    )
    removed_static_frames = int(len(nonstatic_mask) - np.count_nonzero(nonstatic_mask))
    if removed_static_frames > 0:
        print(f"[WARN] Removed {removed_static_frames} near-static middle frames from {episode_dir}")
        data = apply_frame_mask(data, nonstatic_mask)
    return data


def build_episode_data(
    episode_dir: Path,
    image_size: int,
    trim_static: bool,
    static_start_pos_threshold: float,
    static_start_rot_threshold: float,
) -> Dict[str, np.ndarray]:
    frame_paths = sorted(episode_dir.glob("frame_*.pkl"))
    if not frame_paths:
        raise ValueError(f"{episode_dir} has no frame_*.pkl files")
    frames = [load_frame(path) for path in frame_paths]

    data = {
        "robot0_eef_pos": np.asarray([f["robot0_eef_pos"] for f in frames], dtype=np.float32).reshape(-1, 3),
        "robot0_eef_rot_axis_angle": np.asarray(
            [f["robot0_eef_rot_axis_angle"] for f in frames],
            dtype=np.float32,
        ).reshape(-1, 3),
        "robot0_gripper_width": np.asarray(
            [f["robot0_gripper_width"] for f in frames],
            dtype=np.float32,
        ).reshape(-1, 1),
        "action": np.asarray([f["action"] for f in frames], dtype=np.float32).reshape(-1, 7),
        "magnet_xyz": np.asarray([f["magnet_xyz"] for f in frames], dtype=np.float32),
        "magnet_timestamp_ns": np.asarray([f["magnet_timestamp_ns"] for f in frames], dtype=np.int64),
        "magnet_sample_count": np.asarray(
            [f["magnet_sample_count"] for f in frames],
            dtype=np.int32,
        ).reshape(-1, 1),
    }

    timestamps = []
    for idx, frame in enumerate(frames):
        if "timestamp" in frame:
            timestamps.append(float(np.asarray(frame["timestamp"]).reshape(-1)[0]))
        elif "collection_time_ns" in frame:
            timestamps.append(float(np.asarray(frame["collection_time_ns"]).reshape(-1)[0]) * 1e-9)
        else:
            timestamps.append(float(idx))
    timestamps = np.asarray(timestamps, dtype=np.float64)
    data["timestamp"] = timestamps

    camera = stack_optional(frames, "camera0_rgb")
    if camera is not None:
        data["camera0_rgb"] = np.asarray(
            [preprocess_rgb(np.asarray(rgb), image_size=image_size) for rgb in camera],
            dtype=np.uint8,
        )
    command_tcp = stack_optional(frames, "command_tcp_pose_wxyz")
    if command_tcp is not None:
        data["command_tcp_pose_wxyz"] = command_tcp.astype(np.float32)
    gello_command = stack_optional(frames, "gello_command")
    if gello_command is not None:
        data["gello_command"] = gello_command.astype(np.float32)
    gello_gripper_input_raw = stack_optional(frames, "gello_gripper_input_raw")
    if gello_gripper_input_raw is not None:
        data["gello_gripper_input_raw"] = gello_gripper_input_raw.astype(np.float32)

    add_optional_secondary_magnet_data(data, frames)

    for magnet_key in ("magnet_xyz", "magnet2_xyz"):
        if magnet_key not in data:
            continue
        cleaned, replaced_count = replace_abnormal_magnet_readings(data[magnet_key])
        data[magnet_key] = cleaned
        if replaced_count > 0:
            print(
                f"[WARN] {episode_dir.name}: replaced {replaced_count} abnormal "
                f"{magnet_key} readings with nearest normal readings "
                f"(abs threshold {MAGNET_ABNORMAL_ABS_THRESHOLD:g})"
            )

    if trim_static:
        data = trim_static_frames(
            episode_dir=episode_dir,
            data=data,
            pos_threshold=static_start_pos_threshold,
            rot_threshold=static_start_rot_threshold,
        )

    data["timestamp"] = data["timestamp"] - data["timestamp"][0]
    start_pose = np.concatenate(
        [data["robot0_eef_pos"][0], data["robot0_eef_rot_axis_angle"][0]],
        axis=0,
    ).astype(np.float32)
    end_pose = np.concatenate(
        [data["robot0_eef_pos"][-1], data["robot0_eef_rot_axis_angle"][-1]],
        axis=0,
    ).astype(np.float32)
    data["robot0_demo_start_pose"] = np.repeat(start_pose[None, :], len(data["action"]), axis=0)
    data["robot0_demo_end_pose"] = np.repeat(end_pose[None, :], len(data["action"]), axis=0)
    return data


def make_chunks(name: str, arr: np.ndarray):
    if arr.ndim == 1:
        return (min(1024, arr.shape[0]),)
    if name.endswith("_rgb"):
        return (1,) + tuple(arr.shape[1:])
    return (min(1024, arr.shape[0]),) + tuple(arr.shape[1:])


def default_video_output_path(output_path: Path) -> Path:
    output_path = output_path.expanduser()
    name = output_path.name
    if name.endswith(".zarr.zip"):
        stem = name[: -len(".zarr.zip")]
    else:
        stem = output_path.stem
    return output_path.parent / f"{stem}_videos"


def infer_video_fps(timestamps: np.ndarray, fallback_fps: float) -> float:
    if len(timestamps) < 2:
        return fallback_fps
    duration = float(timestamps[-1] - timestamps[0])
    if duration <= 0:
        return fallback_fps
    fps = float((len(timestamps) - 1) / duration)
    if not np.isfinite(fps) or fps <= 0:
        return fallback_fps
    return fps


def get_magnet_plot_limit(magnet_xyz: np.ndarray) -> float:
    finite_values = magnet_xyz[np.isfinite(magnet_xyz)]
    if finite_values.size == 0:
        return 1.0
    limit = float(np.percentile(np.abs(finite_values), 99.0))
    return max(limit, 1.0)


def draw_magnet_panel(
    magnet_frame: np.ndarray,
    panel_height: int,
    panel_width: int,
    value_limit: float,
    title: str = "magnet xyz",
) -> np.ndarray:
    magnet_frame = np.asarray(magnet_frame, dtype=np.float32)
    sensor_count = int(magnet_frame.shape[1])
    panel = np.full((panel_height, panel_width, 3), 245, dtype=np.uint8)
    header_height = 26
    footer_height = 24
    plot_height = max(panel_height - header_height - footer_height, sensor_count)
    row_height = max(plot_height // max(sensor_count, 1), 1)
    colors = {
        "X": (220, 60, 60),
        "Y": (60, 170, 60),
        "Z": (60, 100, 220),
    }

    cv2.putText(
        panel,
        title,
        (10, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )

    for sensor_idx in range(sensor_count):
        y0 = header_height + sensor_idx * row_height
        y1 = (
            panel_height - footer_height
            if sensor_idx == sensor_count - 1
            else header_height + (sensor_idx + 1) * row_height
        )
        top = y0 + 18
        bottom = y1 - 5
        if bottom <= top:
            continue
        center_y = (top + bottom) // 2
        graph_left = 58
        graph_right = panel_width - 12

        cv2.line(panel, (0, y0), (panel_width, y0), (210, 210, 210), 1)
        cv2.line(panel, (graph_left, center_y), (graph_right, center_y), (200, 200, 200), 1)
        cv2.putText(
            panel,
            f"S{sensor_idx + 1}",
            (10, center_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

        latest_values = []
        for axis_idx, axis_name in enumerate(("X", "Y", "Z")):
            values = magnet_frame[:, sensor_idx, axis_idx]
            valid_indices = np.flatnonzero(np.isfinite(values))
            if valid_indices.size == 0:
                latest_values.append(np.nan)
                continue

            latest_values.append(float(values[valid_indices[-1]]))
            points = []
            denom = max(magnet_frame.shape[0] - 1, 1)
            amplitude = max((bottom - top) * 0.45, 1.0)
            for sample_idx in valid_indices:
                x = int(graph_left + (graph_right - graph_left) * sample_idx / denom)
                y = int(center_y - np.clip(values[sample_idx] / value_limit, -1.0, 1.0) * amplitude)
                points.append((x, y))

            if len(points) >= 2:
                cv2.polylines(
                    panel,
                    [np.asarray(points, dtype=np.int32)],
                    isClosed=False,
                    color=colors[axis_name],
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )
            elif points:
                cv2.circle(panel, points[0], 2, colors[axis_name], -1, cv2.LINE_AA)

        latest_text = " ".join(
            f"{axis}={value:.1f}" if np.isfinite(value) else f"{axis}=nan"
            for axis, value in zip(("X", "Y", "Z"), latest_values)
        )
        cv2.putText(
            panel,
            latest_text,
            (graph_left, y0 + 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )

    return panel


def draw_magnet_inputs_panel(
    magnet_frame: np.ndarray,
    panel_height: int,
    panel_width: int,
    value_limit: float,
    magnet2_frame: Optional[np.ndarray] = None,
    magnet2_value_limit: Optional[float] = None,
) -> np.ndarray:
    if magnet2_frame is None:
        return draw_magnet_panel(
            magnet_frame=magnet_frame,
            panel_height=panel_height,
            panel_width=panel_width,
            value_limit=value_limit,
        )

    left_width = panel_width // 2
    right_width = panel_width - left_width
    left = draw_magnet_panel(
        magnet_frame=magnet_frame,
        panel_height=panel_height,
        panel_width=left_width,
        value_limit=value_limit,
        title="Magnet 1 delta",
    )
    right = draw_magnet_panel(
        magnet_frame=magnet2_frame,
        panel_height=panel_height,
        panel_width=right_width,
        value_limit=(
            value_limit if magnet2_value_limit is None else magnet2_value_limit
        ),
        title="Magnet 2 delta",
    )
    return np.concatenate([left, right], axis=1)


def save_rgb_magnet_video(
    frames: np.ndarray,
    magnet_xyz: np.ndarray,
    timestamps: np.ndarray,
    video_path: Path,
    fallback_fps: float,
    panel_width: int,
    magnet2_xyz: Optional[np.ndarray] = None,
) -> None:
    if len(frames) == 0:
        return
    video_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    dual_magnet = magnet2_xyz is not None
    if dual_magnet and len(magnet2_xyz) != len(frames):
        raise ValueError(
            "Second magnetometer frame count does not match RGB frame count: "
            f"{len(magnet2_xyz)} != {len(frames)}"
        )
    panel_width = max(int(panel_width), 560 if dual_magnet else 280)
    if panel_width % 2 != 0:
        panel_width += 1
    if width % 2 != 0:
        frames = frames[:, :, :-1]
        width -= 1
    if height % 2 != 0:
        frames = frames[:, :-1]
        height -= 1

    magnet_plot_limit = get_magnet_plot_limit(magnet_xyz)
    magnet2_plot_limit = (
        get_magnet_plot_limit(magnet2_xyz) if dual_magnet else None
    )
    fps = infer_video_fps(timestamps, fallback_fps)
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width + panel_width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {video_path}")

    try:
        for frame_idx, rgb in enumerate(frames):
            magnet_panel = draw_magnet_inputs_panel(
                magnet_frame=magnet_xyz[frame_idx],
                panel_height=height,
                panel_width=panel_width,
                value_limit=magnet_plot_limit,
                magnet2_frame=(magnet2_xyz[frame_idx] if dual_magnet else None),
                magnet2_value_limit=magnet2_plot_limit,
            )
            cv2.putText(
                magnet_panel,
                f"frame={frame_idx}  t={timestamps[frame_idx]:.3f}s",
                (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (30, 30, 30),
                1,
                cv2.LINE_AA,
            )
            output_rgb = np.concatenate([rgb[:height, :width], magnet_panel], axis=1)
            writer.write(cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def save_episode_video(
    episode_data: Dict[str, np.ndarray],
    episode_dir: Path,
    video_output_dir: Path,
    fallback_fps: float,
    panel_width: int,
) -> None:
    if "camera0_rgb" not in episode_data:
        print(f"[WARN] No camera0_rgb in {episode_dir}; skipping video.")
        return
    episode_name = episode_dir.name
    video_path = video_output_dir / f"{episode_name}_camera0_magnet.mp4"
    save_rgb_magnet_video(
        frames=episode_data["camera0_rgb"],
        magnet_xyz=episode_data["magnet_xyz"],
        timestamps=episode_data["timestamp"],
        video_path=video_path,
        fallback_fps=fallback_fps,
        panel_width=panel_width,
        magnet2_xyz=episode_data.get("magnet2_xyz"),
    )
    print(f"[INFO] Wrote video: {video_path}")


def validate_secondary_magnet_episode_consistency(episodes):
    if not episodes:
        return
    present = [all(key in episode for key in SECONDARY_MAGNET_KEYS) for episode in episodes]
    partial = [
        index
        for index, episode in enumerate(episodes)
        if any(key in episode for key in SECONDARY_MAGNET_KEYS) and not present[index]
    ]
    if partial:
        raise ValueError(
            f"Episodes {partial} contain incomplete secondary magnetometer fields"
        )
    if any(present) and not all(present):
        missing = [index for index, value in enumerate(present) if not value]
        raise ValueError(
            "Cannot mix single- and dual-magnetometer episodes; secondary data "
            f"is missing from episode indices {missing}"
        )


def write_zarr_zip(output_path: Path, episodes: List[Dict[str, np.ndarray]], overwrite: bool):
    validate_secondary_magnet_episode_consistency(episodes)
    output_path = output_path.expanduser()
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"{output_path} exists. Pass --overwrite to replace it.")
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_keys = sorted({key for episode in episodes for key in episode.keys()})
    total_len = sum(len(episode["action"]) for episode in episodes)
    episode_ends = np.cumsum([len(episode["action"]) for episode in episodes], dtype=np.int64)
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)

    temp_dir = Path(tempfile.mkdtemp(prefix="gello_polymetis_zarr_"))
    try:
        store = zarr.DirectoryStore(str(temp_dir / "dataset.zarr"))
        root = zarr.group(store=store, overwrite=True)
        data_group = root.create_group("data")
        meta_group = root.create_group("meta")

        for key in all_keys:
            first = next(episode[key] for episode in episodes if key in episode)
            shape = (total_len,) + tuple(first.shape[1:])
            arr = data_group.create_dataset(
                key,
                shape=shape,
                dtype=first.dtype,
                chunks=make_chunks(key, first),
                compressor=compressor,
                overwrite=True,
            )
            offset = 0
            for episode in episodes:
                n = len(episode["action"])
                if key in episode:
                    arr[offset:offset + n] = episode[key]
                else:
                    arr[offset:offset + n] = np.zeros((n,) + shape[1:], dtype=first.dtype)
                offset += n

        meta_group.create_dataset(
            "episode_ends",
            data=episode_ends,
            dtype=np.int64,
            chunks=(min(1024, len(episode_ends)),),
            compressor=compressor,
            overwrite=True,
        )

        zip_store = zarr.ZipStore(str(output_path), mode="w")
        try:
            zarr.copy_store(store, zip_store)
        finally:
            zip_store.close()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return episode_ends


def validate(output_path: Path):
    store = zarr.ZipStore(str(output_path), mode="r")
    try:
        root = zarr.group(store=store)
        required = [
            "camera0_rgb",
            "robot0_eef_pos",
            "robot0_eef_rot_axis_angle",
            "robot0_gripper_width",
            "action",
            "magnet_xyz",
            "magnet_timestamp_ns",
            "magnet_sample_count",
        ]
        missing = [key for key in required if key not in root["data"]]
        if missing:
            raise KeyError(f"Output is missing required data keys: {missing}")
        n = int(root["data/action"].shape[0])
        for key in required:
            if int(root["data"][key].shape[0]) != n:
                raise ValueError(f"data/{key} length does not match action length")
        if root["data/action"].shape[-1] != 7:
            raise ValueError("data/action must be 7D command xyz+rotvec+gripper")
        if root["data/magnet_xyz"].ndim != 4 or root["data/magnet_xyz"].shape[-1] != 3:
            raise ValueError("data/magnet_xyz must be [T, S, N, 3]")
        secondary_present = [key in root["data"] for key in SECONDARY_MAGNET_KEYS]
        if any(secondary_present) and not all(secondary_present):
            missing = [
                key for key, present in zip(SECONDARY_MAGNET_KEYS, secondary_present)
                if not present
            ]
            raise ValueError(f"Secondary magnetometer data is incomplete: {missing}")
        if all(secondary_present):
            for key in SECONDARY_MAGNET_KEYS:
                if int(root["data"][key].shape[0]) != n:
                    raise ValueError(f"data/{key} length does not match action length")
            if root["data/magnet2_xyz"].ndim != 4 or root["data/magnet2_xyz"].shape[-1] != 3:
                raise ValueError("data/magnet2_xyz must be [T, S, N, 3]")
    finally:
        store.close()


def main():
    args = parse_args()
    input_dir = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    video_output_dir = None
    if not args.no_video:
        video_output_dir = (
            Path(args.video_output).expanduser()
            if args.video_output is not None
            else default_video_output_path(output_path)
        )
        if video_output_dir.exists() and args.overwrite:
            shutil.rmtree(video_output_dir)
        video_output_dir.mkdir(parents=True, exist_ok=True)

    episode_dirs = list_episode_dirs(input_dir, args.min_frames)
    episodes = []
    for episode_dir in episode_dirs:
        try:
            data = build_episode_data(
                episode_dir,
                image_size=args.image_size,
                trim_static=not args.no_trim_static,
                static_start_pos_threshold=args.static_start_pos_threshold,
                static_start_rot_threshold=args.static_start_rot_threshold,
            )
        except ValueError as exc:
            print(f"[WARN] Skipping {episode_dir}: {exc}")
            continue
        if len(data["action"]) < args.min_frames:
            print(
                f"[WARN] Skipping {episode_dir}: only {len(data['action'])} frames "
                "remain after static trimming"
            )
            continue
        episodes.append(data)
        print(
            f"[INFO] Loaded {episode_dir}: {len(data['action'])} frames, "
            f"keys={sorted(data.keys())}"
        )
        if video_output_dir is not None:
            save_episode_video(
                episode_data=data,
                episode_dir=episode_dir,
                video_output_dir=video_output_dir,
                fallback_fps=args.fps,
                panel_width=args.video_panel_width,
            )

    if not episodes:
        raise RuntimeError("No episodes remain after loading/static trimming.")

    episode_ends = write_zarr_zip(output_path, episodes, overwrite=args.overwrite)
    if args.validate:
        validate(output_path)
        print("[INFO] Validation passed.")
    print(f"[INFO] Wrote {output_path}")
    if video_output_dir is not None:
        print(f"[INFO] Wrote synchronized videos to {video_output_dir}")
    print(f"[INFO] Episodes: {len(episode_ends)}, total frames: {int(episode_ends[-1])}")
    print("[INFO] data/action is command xyz+rotvec+gripper_raw; use --action-source source when converting.")


if __name__ == "__main__":
    main()
