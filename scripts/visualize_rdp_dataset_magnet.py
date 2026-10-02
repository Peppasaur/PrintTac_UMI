#!/usr/bin/env python3
"""Render replay-buffer RGB, magnetic signals, contact, and gripper labels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import zarr

try:
    from scripts.pack_gello_polymetis_raw_to_zarr import save_rgb_magnet_video
    from scripts.trim_dataset_after_tactile_contact import (
        detect_contact_frame,
        resolve_replay_buffer_path,
        tactile_contact_score,
    )
except ModuleNotFoundError:  # Direct execution from scripts/.
    from pack_gello_polymetis_raw_to_zarr import save_rgb_magnet_video
    from trim_dataset_after_tactile_contact import (
        detect_contact_frame,
        resolve_replay_buffer_path,
        tactile_contact_score,
    )


DEFAULT_INPUT = "dataset/traj_rdp10d_command_downsample2_gripper_hold"
DEFAULT_OUTPUT = "data/visualizations/traj_rdp10d_command_downsample2_gripper_hold"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--episodes",
        default="all",
        help="Comma-separated episode indices, ranges such as 3-5, or all.",
    )
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--panel-width", type=int, default=400)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_episode_selection(spec, episode_count):
    if spec.strip().lower() == "all":
        return list(range(episode_count))
    selected = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid episode range: {token}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(token))
    invalid = sorted(index for index in selected if index < 0 or index >= episode_count)
    if invalid:
        raise ValueError(f"Episode indices out of range 0..{episode_count - 1}: {invalid}")
    if not selected:
        raise ValueError("No episodes were selected")
    return sorted(selected)


def episode_bounds(episode_ends, episode):
    start = 0 if episode == 0 else int(episode_ends[episode - 1])
    return start, int(episode_ends[episode])


def annotate_rgb_frames(frames, gripper_action, contact_frame, episode):
    frames = np.asarray(frames).copy()
    height, width = frames.shape[1:3]
    band_height = min(48, max(32, height // 5))
    values = np.asarray(gripper_action, dtype=np.float32)
    value_min = float(np.min(values))
    value_max = float(np.max(values))
    value_span = max(value_max - value_min, 1e-6)
    x_values = np.linspace(8, width - 8, len(values)).astype(np.int32)

    for frame_index, frame in enumerate(frames):
        cv2.rectangle(frame, (0, 0), (width, 24), (244, 244, 244), -1)
        phase = "contact" if contact_frame is not None and frame_index >= contact_frame else "pre-contact"
        cv2.putText(
            frame,
            f"ep={episode:03d} frame={frame_index:03d} {phase}",
            (6, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        top = height - band_height
        cv2.rectangle(frame, (0, top), (width, height), (245, 245, 245), -1)
        cv2.putText(
            frame,
            f"gripper={values[frame_index]:.4f}  range=[{value_min:.4f}, {value_max:.4f}]",
            (6, top + 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (50, 50, 50),
            1,
            cv2.LINE_AA,
        )
        y_values = (
            height - 6 - (values - value_min) / value_span * max(1, band_height - 22)
        ).astype(np.int32)
        points = np.column_stack([x_values, y_values]).reshape(-1, 1, 2)
        cv2.polylines(frame, [points], False, (35, 90, 220), 1, cv2.LINE_AA)
        if contact_frame is not None:
            contact_x = int(x_values[contact_frame])
            cv2.line(frame, (contact_x, top), (contact_x, height), (220, 55, 55), 1, cv2.LINE_AA)
        current_x = int(x_values[frame_index])
        current_y = int(y_values[frame_index])
        cv2.circle(frame, (current_x, current_y), 3, (20, 20, 20), -1, cv2.LINE_AA)
    return frames


def rolling_magnet_windows(magnet_xyz, window_frames=24):
    magnet_xyz = np.asarray(magnet_xyz, dtype=np.float32)
    if magnet_xyz.ndim != 3 or magnet_xyz.shape[-1] != 3:
        raise ValueError(
            f"Expected magnet_xyz shape (frames, sensors, 3), got {magnet_xyz.shape}"
        )
    if window_frames <= 0:
        raise ValueError("window_frames must be positive")
    result = np.empty(
        (len(magnet_xyz), window_frames) + magnet_xyz.shape[1:], dtype=np.float32
    )
    for frame_index in range(len(magnet_xyz)):
        start = max(0, frame_index - window_frames + 1)
        history = magnet_xyz[start : frame_index + 1]
        result[frame_index, : window_frames - len(history)] = history[0]
        result[frame_index, window_frames - len(history) :] = history
    return result


def render_dataset(args):
    input_path = resolve_replay_buffer_path(args.input)
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input replay buffer not found: {input_path}")
    root = zarr.open(str(input_path), mode="r")
    data = root["data"]
    required = ("left_wrist_img", "left_gripper1_marker_offset_emb", "action", "timestamp")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Missing required dataset arrays: {missing}")
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    selected = parse_episode_selection(args.episodes, len(episode_ends))
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for episode in selected:
        start, end = episode_bounds(episode_ends, episode)
        output_path = output_dir / f"episode_{episode:03d}_rgb_magnet_gripper.mp4"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")
        frames = data["left_wrist_img"][start:end]
        tactile = data["left_gripper1_marker_offset_emb"][start:end]
        action = data["action"][start:end]
        timestamps = np.asarray(data["timestamp"][start:end], dtype=np.float64)
        contact_scores = tactile_contact_score(tactile)
        contact_frame, baseline, threshold = detect_contact_frame(contact_scores)
        annotated = annotate_rgb_frames(frames, action[:, -1], contact_frame, episode)
        relative_timestamps = timestamps - timestamps[0]
        magnet_windows = rolling_magnet_windows(tactile[:, :12].reshape(-1, 4, 3))
        save_rgb_magnet_video(
            frames=annotated,
            magnet_xyz=magnet_windows,
            timestamps=relative_timestamps,
            video_path=output_path,
            fallback_fps=args.fps,
            panel_width=args.panel_width,
        )
        summary_rows.append(
            {
                "episode": episode,
                "frames": end - start,
                "duration_seconds": float(relative_timestamps[-1]),
                "contact_frame": "" if contact_frame is None else contact_frame,
                "contact_score": "" if contact_frame is None else float(contact_scores[contact_frame]),
                "baseline": baseline,
                "threshold": threshold,
                "pre_contact_action": "" if contact_frame is None else float(action[0, -1]),
                "post_contact_action": "" if contact_frame is None else float(action[contact_frame, -1]),
                "video": output_path.name,
            }
        )
        print(f"Wrote {output_path}")

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {summary_path}")


def main():
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.panel_width <= 0:
        raise ValueError("--panel-width must be positive")
    render_dataset(args)


if __name__ == "__main__":
    main()
