#!/usr/bin/env python3
"""
Convert a UMI-style Franka zarr trajectory into an RDP training dataset.

The RDP RealImageTactileDataset in this repository expects:

    <dataset_dir>/replay_buffer.zarr/
        data/<keys>
        meta/episode_ends

This script reads a UMI-style zarr zip such as dataset/traj.zarr.zip and writes
that directory layout with key names and action representations used by the RDP
task configs.
"""

import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import zarr
from numcodecs import Blosc
from scipy.spatial.transform import Rotation


PRESET_HELP = {
    "rdp10d": (
        "Recommended for Franka DP with rotation. Produces left_wrist_img, "
        "left_robot_tcp_pose[9], left_robot_gripper_width[1], action[10]. "
        "Use with task=real_wipe_image_dp_absolute_12fps or another 10D "
        "single-arm RDP config."
    ),
    "train_dp_default": (
        "Compatibility preset for the current hardcoded train_dp.sh task "
        "real_peel_image_gelsight_emb_dp_absolute_12fps. Produces "
        "left_robot_tcp_pose[3], action[4], and a zero "
        "left_gripper1_marker_offset_emb[15]. This discards rotation."
    ),
}

MAGNET_MODE_HELP = (
    "How to export magnetic tactile data when --magnet-key exists. "
    "auto writes both standard low_dim keys; tactile writes only "
    "--magnet-tactile-key; wrench writes only --magnet-wrench-key; "
    "both is the explicit form of auto; none disables magnetic export."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert UMI/Franka zarr data to RDP replay_buffer.zarr format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        default="dataset/traj.zarr.zip",
        help="Input zarr zip or zarr directory.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "Output dataset directory. The script writes replay_buffer.zarr "
            "inside this directory."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_HELP),
        default="rdp10d",
        help="Output key/action preset.",
    )
    parser.add_argument(
        "--image-size",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        default=(240, 320),
        help="Output image size expected by most original RDP real-image configs.",
    )
    parser.add_argument(
        "--temporal-downsample",
        type=int,
        default=1,
        help=(
            "Keep every Nth step within each episode. Use 2 to turn about "
            "25Hz teleop data into about 12.5Hz data for the original 12fps DP configs."
        ),
    )
    parser.add_argument(
        "--filter-pose-jumps",
        action="store_true",
        help=(
            "Filter discontinuous pose jumps after temporal downsampling. "
            "A jump splits the episode so actions are not trained across the discontinuity."
        ),
    )
    parser.add_argument(
        "--pose-jump-pos-threshold-m",
        type=float,
        default=0.03,
        help="Position delta threshold in meters for --filter-pose-jumps.",
    )
    parser.add_argument(
        "--pose-jump-rot-threshold-deg",
        type=float,
        default=15.0,
        help="Rotation delta threshold in degrees for --filter-pose-jumps.",
    )
    parser.add_argument(
        "--pose-jump-drop-target-frame",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When filtering pose jumps, drop the frame on the target side of each "
            "jump. Disable to keep it as the first frame of the next segment."
        ),
    )
    parser.add_argument(
        "--pose-jump-min-episode-len",
        type=int,
        default=2,
        help="Minimum segment length kept after pose-jump filtering.",
    )
    parser.add_argument(
        "--image-key",
        default="camera0_rgb",
        help="Input RGB image key under data/.",
    )
    parser.add_argument(
        "--pos-key",
        default="robot0_eef_pos",
        help="Input end-effector position key under data/.",
    )
    parser.add_argument(
        "--rotvec-key",
        default="robot0_eef_rot_axis_angle",
        help="Input end-effector rotation-vector key under data/.",
    )
    parser.add_argument(
        "--gripper-key",
        default="robot0_gripper_width",
        help="Input gripper width key under data/.",
    )
    parser.add_argument(
        "--action-key",
        default="action",
        help="Input action key under data/.",
    )
    parser.add_argument(
        "--action-source",
        choices=("auto", "source", "next_obs"),
        default="auto",
        help=(
            "How to build output actions. 'source' copies --action-key, "
            "'next_obs' uses the TCP pose from the next selected observation "
            "inside each episode while keeping the current --action-key gripper "
            "command, and 'auto' uses next_obs only when --action-key is detected "
            "to be an obs[t+1] proxy. This keeps temporal downsampling from "
            "leaving TCP actions one raw frame ahead without shifting gripper commands."
        ),
    )
    parser.add_argument(
        "--action-proxy-detect-atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance used by --action-source auto when detecting obs[t+1] action proxies.",
    )
    parser.add_argument(
        "--target-image-key",
        default="left_wrist_img",
        help="Output RGB image key under data/.",
    )
    parser.add_argument(
        "--target-pose-key",
        default="left_robot_tcp_pose",
        help="Output robot pose observation key under data/.",
    )
    parser.add_argument(
        "--target-gripper-key",
        default="left_robot_gripper_width",
        help="Output gripper observation key under data/.",
    )
    parser.add_argument(
        "--magnet-key",
        default="magnet_xyz",
        help=(
            "Input magnetic tactile key under data/. Expected shape is "
            "[T, S, N, 3], where S is fast samples per policy step and "
            "N is the number of magnetic sensors."
        ),
    )
    parser.add_argument(
        "--magnet-sample-count-key",
        default="magnet_sample_count",
        help=(
            "Optional input key with valid magnetic sample count per row. "
            "If missing, all samples in --magnet-key are treated as valid."
        ),
    )
    parser.add_argument(
        "--magnet-timestamp-key",
        default="magnet_timestamp_ns",
        help="Optional input key with magnetic sub-sample timestamps; copied only if --copy-magnet-timestamps is set.",
    )
    parser.add_argument(
        "--magnet-mode",
        choices=("auto", "none", "tactile", "wrench", "both"),
        default="auto",
        help=MAGNET_MODE_HELP,
    )
    parser.add_argument(
        "--magnet-tactile-key",
        default="left_gripper1_marker_offset_emb",
        help=(
            "Output low_dim tactile key for magnetic data. The default "
            "matches original RDP tactile embedding task configs."
        ),
    )
    parser.add_argument(
        "--magnet-tactile-dim",
        type=int,
        default=15,
        help=(
            "Output dimension for --magnet-tactile-key. Magnetic sensor "
            "time-mean values are flattened, then zero-padded or truncated "
            "to this size."
        ),
    )
    parser.add_argument(
        "--magnet-wrench-key",
        default="left_robot_tcp_wrench",
        help=(
            "Output low_dim wrench-compatible key for magnetic data. "
            "The first three dimensions are averaged magnetic xyz values "
            "and the last three torque dimensions are zeros."
        ),
    )
    parser.add_argument(
        "--magnet2-key",
        default="magnet2_xyz",
        help="Optional second magnetic input key under data/, with the same layout as --magnet-key.",
    )
    parser.add_argument(
        "--magnet2-sample-count-key",
        default="magnet2_sample_count",
        help="Optional valid sample count key for --magnet2-key.",
    )
    parser.add_argument(
        "--magnet2-timestamp-key",
        default="magnet2_timestamp_ns",
        help="Optional timestamp key for --magnet2-key.",
    )
    parser.add_argument(
        "--magnet2-tactile-key",
        default="left_gripper2_marker_offset_emb",
        help="Output low_dim tactile key for the second magnetic input.",
    )
    parser.add_argument(
        "--magnet2-tactile-dim",
        type=int,
        default=15,
        help="Output dimension for --magnet2-tactile-key.",
    )
    parser.add_argument(
        "--magnet2-wrench-key",
        default="left_robot_tcp_wrench2",
        help="Output wrench-compatible key for the second magnetic input.",
    )
    parser.add_argument(
        "--require-magnet2",
        action="store_true",
        help="Fail unless the second magnetic input exists and is exported.",
    )
    parser.add_argument(
        "--copy-magnet-timestamps",
        action="store_true",
        help=(
            "Also copy selected magnetic sub-sample timestamp rows for every "
            "available magnetic input. This is for debugging only; RDP training "
            "reads the low_dim magnetic outputs."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=64,
        help="Number of timesteps processed per chunk.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional limit for smoke tests; truncates after this many converted rows.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing output replay_buffer.zarr before writing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print source/target summary without writing data.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the converted zarr after writing.",
    )
    return parser.parse_args()


def open_zarr(path):
    path = Path(path).expanduser()
    if path.is_file() and path.name.endswith(".zip"):
        store = zarr.ZipStore(str(path), mode="r")
        return zarr.open(store=store, mode="r"), store
    if path.is_dir():
        if (path / "replay_buffer.zarr").is_dir():
            path = path / "replay_buffer.zarr"
        store = zarr.DirectoryStore(str(path))
        return zarr.open(store=store, mode="r"), store
    raise FileNotFoundError(f"Input must be a .zarr.zip file or zarr directory: {path}")


def require_data_key(root, key):
    if "data" not in root:
        raise KeyError("Input zarr is missing group 'data'.")
    if key not in root["data"]:
        available = ", ".join(sorted(root["data"].keys()))
        raise KeyError(f"Input data key '{key}' not found. Available keys: {available}")
    return root["data"][key]


def require_episode_ends(root):
    if "meta" not in root or "episode_ends" not in root["meta"]:
        raise KeyError("Input zarr is missing meta/episode_ends.")
    return np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)


def build_downsample_indices(episode_ends, total_len, downsample, max_steps=None):
    if downsample < 1:
        raise ValueError("--temporal-downsample must be >= 1")

    indices = []
    new_episode_ends = []
    start = 0
    for end in episode_ends:
        end = int(end)
        if end > total_len:
            raise ValueError(
                f"episode_ends contains {end}, but data length is only {total_len}"
            )
        episode_indices = np.arange(start, end, downsample, dtype=np.int64)
        if max_steps is not None:
            remaining = max_steps - len(indices)
            if remaining <= 0:
                break
            episode_indices = episode_indices[:remaining]
        if len(episode_indices) > 0:
            indices.extend(episode_indices.tolist())
            new_episode_ends.append(len(indices))
        start = end

    if not indices:
        raise ValueError("No samples selected. Check --temporal-downsample/--max-steps.")

    return np.asarray(indices, dtype=np.int64), np.asarray(new_episode_ends, dtype=np.int64)


def build_next_selected_indices(indices, episode_ends):
    """Map each selected output row to the next selected row in the same episode."""
    next_indices = indices.copy()
    start = 0
    for end in episode_ends:
        end = int(end)
        if end > start:
            next_indices[start : end - 1] = indices[start + 1 : end]
            next_indices[end - 1] = indices[end - 1]
        start = end
    return next_indices


def rotation_angle_between_rotvecs_deg(rotvec_a, rotvec_b):
    rot_a = Rotation.from_rotvec(rotvec_a.astype(np.float64))
    rot_b = Rotation.from_rotvec(rotvec_b.astype(np.float64))
    rel = rot_a.inv() * rot_b
    return np.rad2deg(rel.magnitude()).astype(np.float32)


def filter_pose_jump_indices(
    indices,
    episode_ends,
    pos_arr,
    rotvec_arr,
    pos_threshold_m,
    rot_threshold_deg,
    drop_target_frame=True,
    min_episode_len=2,
):
    if min_episode_len < 1:
        raise ValueError("--pose-jump-min-episode-len must be >= 1")
    if pos_threshold_m <= 0 and rot_threshold_deg <= 0:
        raise ValueError(
            "At least one pose jump threshold must be positive when --filter-pose-jumps is set."
        )

    filtered = []
    new_episode_ends = []
    stats = {
        "input_rows": int(len(indices)),
        "output_rows": 0,
        "dropped_rows": 0,
        "input_episodes": int(len(episode_ends)),
        "output_episodes": 0,
        "jump_count": 0,
        "dropped_short_segments": 0,
        "max_pos_delta_m": 0.0,
        "max_rot_delta_deg": 0.0,
    }

    def keep_segment(segment):
        if len(segment) >= min_episode_len:
            filtered.extend(segment)
            new_episode_ends.append(len(filtered))
        elif len(segment) > 0:
            stats["dropped_short_segments"] += 1
            stats["dropped_rows"] += len(segment)

    start = 0
    for episode_idx, end in enumerate(episode_ends):
        end = int(end)
        episode_indices = indices[start:end]
        start = end
        if len(episode_indices) == 0:
            continue
        if len(episode_indices) == 1:
            keep_segment(episode_indices.tolist())
            continue

        pos = read_rows(pos_arr, episode_indices).astype(np.float32)[:, :3]
        rotvec = read_rows(rotvec_arr, episode_indices).astype(np.float32)[:, :3]
        pos_delta = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        rot_delta = rotation_angle_between_rotvecs_deg(rotvec[:-1], rotvec[1:])
        stats["max_pos_delta_m"] = max(stats["max_pos_delta_m"], float(np.max(pos_delta)))
        stats["max_rot_delta_deg"] = max(stats["max_rot_delta_deg"], float(np.max(rot_delta)))

        jump_mask = np.zeros(len(pos_delta), dtype=bool)
        if pos_threshold_m > 0:
            jump_mask |= pos_delta > pos_threshold_m
        if rot_threshold_deg > 0:
            jump_mask |= rot_delta > rot_threshold_deg
        jump_after_local = np.flatnonzero(jump_mask)
        if len(jump_after_local) == 0:
            keep_segment(episode_indices.tolist())
            continue

        stats["jump_count"] += int(len(jump_after_local))
        print(
            f"[WARN] Pose jumps in episode {episode_idx}: "
            f"{len(jump_after_local)} jump(s), "
            f"max_pos={float(np.max(pos_delta)) * 1000.0:.2f} mm, "
            f"max_rot={float(np.max(rot_delta)):.2f} deg"
        )

        segment_start = 0
        for jump_local in jump_after_local:
            jump_local = int(jump_local)
            keep_segment(episode_indices[segment_start : jump_local + 1].tolist())
            if drop_target_frame:
                stats["dropped_rows"] += 1
                segment_start = jump_local + 2
            else:
                segment_start = jump_local + 1
        keep_segment(episode_indices[segment_start:].tolist())

    if not filtered:
        raise ValueError(
            "Pose-jump filtering removed all samples. Relax thresholds or disable filtering."
        )

    filtered = np.asarray(filtered, dtype=np.int64)
    new_episode_ends = np.asarray(new_episode_ends, dtype=np.int64)
    stats["output_rows"] = int(len(filtered))
    stats["output_episodes"] = int(len(new_episode_ends))
    stats["dropped_rows"] = int(len(indices) - len(filtered))
    return filtered, new_episode_ends, stats


def read_rows(arr, row_indices):
    row_indices = np.asarray(row_indices, dtype=np.int64)
    if len(row_indices) == 0:
        return arr[:0]
    if np.all(np.diff(row_indices) == 1):
        return arr[int(row_indices[0]) : int(row_indices[-1]) + 1]
    selection = (row_indices,) + tuple(slice(None) for _ in arr.shape[1:])
    return arr.get_orthogonal_selection(selection)


def rotvec_to_rot6d(rotvec):
    rot_mats = Rotation.from_rotvec(rotvec.astype(np.float64)).as_matrix()
    rot6d = rot_mats[:, :, :2].swapaxes(1, 2).reshape(rot_mats.shape[0], 6)
    return rot6d.astype(np.float32)


def pose9_from_pos_rotvec(pos, rotvec):
    return np.concatenate([pos.astype(np.float32), rotvec_to_rot6d(rotvec)], axis=-1)


def action10_from_source(action, fallback_gripper=None):
    action = action.astype(np.float32)
    dim = action.shape[-1]
    if dim >= 10:
        return action[:, :10]
    if dim >= 7:
        pos = action[:, :3]
        rot6d = rotvec_to_rot6d(action[:, 3:6])
        gripper = action[:, 6:7]
        return np.concatenate([pos, rot6d, gripper], axis=-1).astype(np.float32)
    if dim == 6:
        if fallback_gripper is None:
            raise ValueError("6D action has no gripper column; provide --gripper-key.")
        pos = action[:, :3]
        rot6d = rotvec_to_rot6d(action[:, 3:6])
        return np.concatenate([pos, rot6d, fallback_gripper], axis=-1).astype(np.float32)
    raise ValueError(
        f"Cannot convert action with shape {action.shape} to 10D xyz+rot6d+gripper."
    )


def action_from_pose_gripper(pos, rotvec, gripper):
    if gripper.ndim == 1:
        gripper = gripper[:, None]
    return np.concatenate(
        [
            pos[:, :3].astype(np.float32, copy=False),
            rotvec[:, :3].astype(np.float32, copy=False),
            gripper[:, :1].astype(np.float32, copy=False),
        ],
        axis=-1,
    ).astype(np.float32)


def gripper_from_source_action(action, fallback_gripper=None):
    action = action.astype(np.float32)
    dim = action.shape[-1]
    if dim >= 10:
        return action[:, 9:10]
    if dim >= 7:
        return action[:, 6:7]
    if dim == 6 and fallback_gripper is not None:
        if fallback_gripper.ndim == 1:
            fallback_gripper = fallback_gripper[:, None]
        return fallback_gripper[:, :1].astype(np.float32)
    if dim >= 4:
        return action[:, 3:4]
    if fallback_gripper is not None:
        if fallback_gripper.ndim == 1:
            fallback_gripper = fallback_gripper[:, None]
        return fallback_gripper[:, :1].astype(np.float32)
    raise ValueError(
        f"Cannot extract a gripper command from action with shape {action.shape}."
    )


def action4_from_source(action, fallback_gripper=None):
    action = action.astype(np.float32)
    dim = action.shape[-1]
    if dim >= 7:
        return np.concatenate([action[:, :3], action[:, 6:7]], axis=-1).astype(np.float32)
    if dim >= 4:
        return action[:, :4]
    if dim == 3 and fallback_gripper is not None:
        return np.concatenate([action[:, :3], fallback_gripper], axis=-1).astype(np.float32)
    raise ValueError(
        f"Cannot convert action with shape {action.shape} to 4D xyz+gripper."
    )


def resize_images(images, height, width):
    if images.shape[1] == height and images.shape[2] == width:
        return images.astype(np.uint8, copy=False)

    resized = np.empty((images.shape[0], height, width, images.shape[3]), dtype=np.uint8)
    interpolation = cv2.INTER_AREA
    if height > images.shape[1] or width > images.shape[2]:
        interpolation = cv2.INTER_LINEAR
    for i, image in enumerate(images):
        resized[i] = cv2.resize(image, (width, height), interpolation=interpolation)
    return resized


def create_array(group, name, shape, dtype, chunks, compressor):
    return group.create_dataset(
        name,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        compressor=compressor,
        overwrite=True,
    )


def has_combined_dual_magnet_input(args, root):
    """Detect ARPose data with right/left board snapshots packed in one key."""
    if "data" not in root or args.magnet_key not in root["data"]:
        return False
    data = root["data"]
    if args.magnet2_key in data:
        return False
    magnet = data[args.magnet_key]
    if magnet.ndim != 4 or magnet.shape[1] != 2:
        return False
    if args.magnet_sample_count_key not in data:
        return False
    counts_arr = data[args.magnet_sample_count_key]
    if counts_arr.ndim != 2 or counts_arr.shape != (magnet.shape[0], 2):
        return False

    board_order = root.attrs.get("magnet_board_order")
    if board_order is not None:
        return list(board_order) == ["right", "left"]

    # ARPose stores one snapshot per board and one 0/1 validity count per slot.
    counts = np.asarray(counts_arr[:], dtype=np.int64)
    if np.any((counts < 0) | (counts > 1)):
        return False
    return bool(np.all(np.any(counts > 0, axis=0)))


def resolve_magnet_outputs(
    args,
    root,
    magnet_key=None,
    required=False,
    combined_dual_fallback=False,
):
    magnet_key = args.magnet_key if magnet_key is None else magnet_key
    has_magnet = (
        ("data" in root and magnet_key in root["data"])
        or combined_dual_fallback
    )
    if args.magnet_mode == "none":
        if required:
            raise ValueError("--require-magnet2 cannot be used with --magnet-mode none")
        return []
    if not has_magnet:
        if args.magnet_mode == "auto" and not required:
            return []
        raise KeyError(
            f"Input data key '{magnet_key}' not found for --magnet-mode "
            f"{args.magnet_mode}."
        )
    if args.magnet_mode in ("auto", "both"):
        return ["tactile", "wrench"]
    return [args.magnet_mode]


def validate_magnet_output_keys(args, magnet_outputs, magnet2_outputs):
    output_keys = []
    if "tactile" in magnet_outputs:
        output_keys.append(args.magnet_tactile_key)
    if "wrench" in magnet_outputs:
        output_keys.append(args.magnet_wrench_key)
    if "tactile" in magnet2_outputs:
        output_keys.append(args.magnet2_tactile_key)
    if "wrench" in magnet2_outputs:
        output_keys.append(args.magnet2_wrench_key)
    if len(output_keys) != len(set(output_keys)):
        raise ValueError(f"Magnetic output keys must be unique, got {output_keys}")


def validate_magnet_array(arr, key):
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(
            f"data/{key} must have shape [T, samples_per_step, sensor_count, 3], "
            f"got {arr.shape}"
        )


def normalize_magnet_sample_layout(magnet, sample_count=None):
    """Normalize valid-count layouts before averaging magnetic samples.

    Standard recordings store one valid sample count per frame: ``[B]`` or
    ``[B, 1]``. The ARPose dual-board exporter uses ``[B, board_slots]`` when
    ``magnet_xyz`` stores one snapshot per board. If exactly one board slot
    contains valid samples across the batch, retain that board as a one-sample
    window and ignore the empty board slots.
    """
    magnet = np.asarray(magnet, dtype=np.float32)
    if magnet.ndim != 4 or magnet.shape[-1] != 3:
        raise ValueError(f"Expected magnetic batch [B, S, N, 3], got {magnet.shape}")
    if sample_count is None:
        return magnet, None

    batch_size, max_samples = magnet.shape[:2]
    counts = np.asarray(sample_count, dtype=np.int64)
    if counts.ndim == 1:
        if counts.shape[0] != batch_size:
            raise ValueError(
                "magnet sample count length does not match magnetic batch: "
                f"{counts.shape[0]} != {batch_size}"
            )
        return magnet, counts
    if counts.ndim != 2 or counts.shape[0] != batch_size:
        raise ValueError(
            "magnet sample count must have shape [B], [B, 1], or [B, S]; "
            f"got {counts.shape} for magnetic batch {magnet.shape}"
        )
    if counts.shape[1] == 1:
        return magnet, counts[:, 0]
    if counts.shape[1] != max_samples:
        raise ValueError(
            "magnet sample count width must be 1 or match the magnetic sample "
            f"axis: {counts.shape[1]} not in (1, {max_samples})"
        )
    if np.any((counts < 0) | (counts > 1)):
        raise ValueError(
            "Per-slot magnet sample counts are only supported for ARPose "
            "dual-board snapshots, where every count must be 0 or 1."
        )

    active_slots = np.flatnonzero(np.any(counts > 0, axis=0))
    if len(active_slots) == 0:
        # Every row in this chunk is missing a magnetic sample. The selected
        # slot is irrelevant because the zero counts mask all values.
        return magnet[:, :1], counts[:, 0]
    if len(active_slots) != 1:
        raise ValueError(
            "Expected exactly one active magnetic board slot in per-slot "
            f"sample counts, found {active_slots.tolist()}."
        )
    board_slot = int(active_slots[0])
    return magnet[:, board_slot : board_slot + 1], counts[:, board_slot]


def magnetic_time_mean(magnet, sample_count=None):
    magnet, sample_count = normalize_magnet_sample_layout(magnet, sample_count)
    magnet = np.nan_to_num(magnet, nan=0.0)

    batch_size, max_samples = magnet.shape[:2]
    if sample_count is None:
        return magnet.mean(axis=1)

    counts = np.asarray(sample_count, dtype=np.int64).reshape(batch_size)
    counts = np.clip(counts, 0, max_samples)
    sample_indices = np.arange(max_samples, dtype=np.int64)[None, :]
    # Magnet samples are stored right-aligned when fewer than max_samples are
    # available, with leading NaNs/zeros used only as padding.
    mask = sample_indices >= (max_samples - counts[:, None])
    weighted = magnet * mask[:, :, None, None].astype(np.float32)
    denom = np.maximum(counts, 1).astype(np.float32)[:, None, None]
    mean = weighted.sum(axis=1) / denom
    mean[counts == 0] = 0
    return mean.astype(np.float32)


def magnet_to_tactile_embedding(magnet, sample_count=None, output_dim=15):
    if output_dim <= 0:
        raise ValueError("--magnet-tactile-dim must be positive")
    mean_by_sensor = magnetic_time_mean(magnet, sample_count=sample_count)
    flat = mean_by_sensor.reshape(mean_by_sensor.shape[0], -1)
    if flat.shape[1] == output_dim:
        return flat.astype(np.float32)
    out = np.zeros((flat.shape[0], output_dim), dtype=np.float32)
    copy_dim = min(flat.shape[1], output_dim)
    out[:, :copy_dim] = flat[:, :copy_dim]
    return out


def magnet_to_wrench(magnet, sample_count=None):
    mean_by_sensor = magnetic_time_mean(magnet, sample_count=sample_count)
    force_like = mean_by_sensor.mean(axis=1)
    wrench = np.zeros((force_like.shape[0], 6), dtype=np.float32)
    wrench[:, :3] = force_like[:, :3]
    return wrench


def sample_nonfinal_rows(episode_ends, total_len, max_samples=2048):
    nonfinal = np.ones(total_len, dtype=bool)
    nonfinal[np.asarray(episode_ends, dtype=np.int64) - 1] = False
    rows = np.flatnonzero(nonfinal)
    if rows.size <= max_samples:
        return rows
    sample_idx = np.linspace(0, rows.size - 1, max_samples, dtype=np.int64)
    return rows[sample_idx]


def detect_next_obs_action_proxy(action_arr, pos_arr, rotvec_arr, gripper_arr, episode_ends, atol):
    total_len = int(action_arr.shape[0])
    rows = sample_nonfinal_rows(episode_ends, total_len)
    if rows.size == 0:
        return False, "not enough non-final rows"

    action = read_rows(action_arr, rows).astype(np.float32)
    dim = action.shape[-1]
    if dim < 4:
        return False, f"action dim {dim} is too small"

    next_rows = rows + 1
    next_pos = read_rows(pos_arr, next_rows).astype(np.float32)[:, :3]
    next_gripper = read_rows(gripper_arr, next_rows).astype(np.float32)
    if next_gripper.ndim == 1:
        next_gripper = next_gripper[:, None]
    next_gripper = next_gripper[:, :1]

    diffs = [np.max(np.abs(action[:, :3] - next_pos))]
    labels = [f"pos_max={diffs[-1]:.3g}"]

    if dim >= 7:
        next_rotvec = read_rows(rotvec_arr, next_rows).astype(np.float32)[:, :3]
        diffs.append(np.max(np.abs(action[:, 3:6] - next_rotvec)))
        labels.append(f"rot_max={diffs[-1]:.3g}")
        diffs.append(np.max(np.abs(action[:, 6:7] - next_gripper)))
        labels.append(f"gripper_max={diffs[-1]:.3g}")
    else:
        diffs.append(np.max(np.abs(action[:, 3:4] - next_gripper)))
        labels.append(f"gripper_max={diffs[-1]:.3g}")

    is_proxy = bool(max(diffs) <= atol)
    return is_proxy, ", ".join(labels)


def resolve_action_source(args, action_arr, pos_arr, rotvec_arr, gripper_arr, src_episode_ends):
    if args.action_source == "next_obs":
        return True, "forced by --action-source next_obs"
    if args.action_source == "source":
        return False, "forced by --action-source source"

    is_proxy, detail = detect_next_obs_action_proxy(
        action_arr,
        pos_arr,
        rotvec_arr,
        gripper_arr,
        src_episode_ends,
        atol=args.action_proxy_detect_atol,
    )
    if is_proxy:
        return True, f"auto-detected obs[t+1] proxy ({detail})"
    return False, f"auto kept source action ({detail})"


def print_summary(
    args,
    root,
    indices,
    episode_ends,
    action_from_next_obs=False,
    action_source_detail=None,
    pose_jump_stats=None,
):
    data = root["data"]
    combined_dual_magnet = has_combined_dual_magnet_input(args, root)
    magnet_outputs = resolve_magnet_outputs(args, root)
    magnet2_outputs = resolve_magnet_outputs(
        args,
        root,
        magnet_key=args.magnet2_key,
        required=args.require_magnet2,
        combined_dual_fallback=combined_dual_magnet,
    )
    print("Input:", args.input)
    print("Preset:", args.preset)
    print("Preset detail:", PRESET_HELP[args.preset])
    print("Magnet mode:", args.magnet_mode)
    if combined_dual_magnet:
        print(
            f"Magnet board layout: splitting data/{args.magnet_key} slots 0/1 "
            f"to data/{args.magnet_tactile_key} and "
            f"data/{args.magnet2_tactile_key}"
        )
    print("Selected rows:", len(indices))
    print("Episodes:", len(episode_ends))
    action_mode = (
        f"next selected observation TCP + current data/{args.action_key} gripper"
        if action_from_next_obs
        else f"data/{args.action_key}"
    )
    print(f"Action source: {action_mode}")
    if action_source_detail:
        print(f"Action source detail: {action_source_detail}")
    if pose_jump_stats is not None:
        print("Pose jump filter:")
        print(
            f"  input_rows={pose_jump_stats['input_rows']}, "
            f"output_rows={pose_jump_stats['output_rows']}, "
            f"dropped_rows={pose_jump_stats['dropped_rows']}"
        )
        print(
            f"  input_episodes={pose_jump_stats['input_episodes']}, "
            f"output_episodes={pose_jump_stats['output_episodes']}, "
            f"jump_count={pose_jump_stats['jump_count']}, "
            f"dropped_short_segments={pose_jump_stats['dropped_short_segments']}"
        )
        print(
            f"  max_pos_delta_m={pose_jump_stats['max_pos_delta_m']:.6f}, "
            f"max_rot_delta_deg={pose_jump_stats['max_rot_delta_deg']:.3f}"
        )
    print("Output:", args.output)
    print("Target replay buffer:", Path(args.output) / "replay_buffer.zarr")
    print("Source keys:")
    for key in sorted(data.keys()):
        arr = data[key]
        print(f"  data/{key}: shape={arr.shape}, dtype={arr.dtype}")
    print("Target keys:")
    if args.preset == "rdp10d":
        print(f"  data/{args.target_image_key}: ({len(indices)}, {args.image_size[0]}, {args.image_size[1]}, 3) uint8")
        print(f"  data/{args.target_pose_key}: ({len(indices)}, 9) float32")
        print(f"  data/{args.target_gripper_key}: ({len(indices)}, 1) float32")
        print(f"  data/action: ({len(indices)}, 10) float32")
    else:
        print(f"  data/{args.target_image_key}: ({len(indices)}, {args.image_size[0]}, {args.image_size[1]}, 3) uint8")
        print(f"  data/{args.target_pose_key}: ({len(indices)}, 3) float32")
        print(f"  data/{args.target_gripper_key}: ({len(indices)}, 1) float32")
        if "tactile" not in magnet_outputs:
            print("  data/left_gripper1_marker_offset_emb: " f"({len(indices)}, 15) float32 zeros")
        print(f"  data/action: ({len(indices)}, 4) float32")
    if "tactile" in magnet_outputs:
        print(f"  data/{args.magnet_tactile_key}: ({len(indices)}, {args.magnet_tactile_dim}) float32 from data/{args.magnet_key}")
    if "wrench" in magnet_outputs:
        print(f"  data/{args.magnet_wrench_key}: ({len(indices)}, 6) float32 from data/{args.magnet_key}")
    if "tactile" in magnet2_outputs:
        magnet2_source = (
            f"data/{args.magnet_key} board slot 1"
            if combined_dual_magnet
            else f"data/{args.magnet2_key}"
        )
        print(
            f"  data/{args.magnet2_tactile_key}: "
            f"({len(indices)}, {args.magnet2_tactile_dim}) float32 "
            f"from {magnet2_source}"
        )
    if "wrench" in magnet2_outputs:
        magnet2_source = (
            f"data/{args.magnet_key} board slot 1"
            if combined_dual_magnet
            else f"data/{args.magnet2_key}"
        )
        print(f"  data/{args.magnet2_wrench_key}: ({len(indices)}, 6) float32 from {magnet2_source}")
    if args.copy_magnet_timestamps and args.magnet_timestamp_key in data:
        shape_tail = data[args.magnet_timestamp_key].shape[1:]
        print(f"  data/{args.magnet_timestamp_key}: ({len(indices)}, {', '.join(map(str, shape_tail))}) {data[args.magnet_timestamp_key].dtype}")
    if args.copy_magnet_timestamps and args.magnet2_timestamp_key in data:
        shape_tail = data[args.magnet2_timestamp_key].shape[1:]
        print(
            f"  data/{args.magnet2_timestamp_key}: "
            f"({len(indices)}, {', '.join(map(str, shape_tail))}) "
            f"{data[args.magnet2_timestamp_key].dtype}"
        )


def convert(args):
    if args.output is None:
        suffix = "rdp10d" if args.preset == "rdp10d" else "rdp_train_dp_default"
        args.output = str(Path("dataset") / f"traj_{suffix}")

    root, input_store = open_zarr(args.input)
    temporary_replay_buffer_path = None
    try:
        image_arr = require_data_key(root, args.image_key)
        pos_arr = require_data_key(root, args.pos_key)
        rotvec_arr = require_data_key(root, args.rotvec_key)
        gripper_arr = require_data_key(root, args.gripper_key)
        action_arr = require_data_key(root, args.action_key)
        src_episode_ends = require_episode_ends(root)
        combined_dual_magnet = has_combined_dual_magnet_input(args, root)
        magnet_outputs = resolve_magnet_outputs(args, root)
        magnet2_outputs = resolve_magnet_outputs(
            args,
            root,
            magnet_key=args.magnet2_key,
            required=args.require_magnet2,
            combined_dual_fallback=combined_dual_magnet,
        )
        validate_magnet_output_keys(args, magnet_outputs, magnet2_outputs)
        magnet_arr = None
        sample_count_arr = None
        magnet_timestamp_arr = None
        magnet2_arr = None
        sample_count2_arr = None
        magnet_timestamp2_arr = None
        if magnet_outputs:
            magnet_arr = require_data_key(root, args.magnet_key)
            validate_magnet_array(magnet_arr, args.magnet_key)
            if args.magnet_sample_count_key in root["data"]:
                sample_count_arr = root["data"][args.magnet_sample_count_key]
            if args.copy_magnet_timestamps:
                magnet_timestamp_arr = require_data_key(root, args.magnet_timestamp_key)
        if magnet2_outputs:
            if combined_dual_magnet:
                magnet2_arr = magnet_arr
                sample_count2_arr = sample_count_arr
                magnet_timestamp2_arr = magnet_timestamp_arr
            else:
                magnet2_arr = require_data_key(root, args.magnet2_key)
                validate_magnet_array(magnet2_arr, args.magnet2_key)
                if args.magnet2_sample_count_key in root["data"]:
                    sample_count2_arr = root["data"][args.magnet2_sample_count_key]
                if args.copy_magnet_timestamps:
                    magnet_timestamp2_arr = require_data_key(root, args.magnet2_timestamp_key)

        total_len = int(image_arr.shape[0])
        for key, arr in [
            (args.pos_key, pos_arr),
            (args.rotvec_key, rotvec_arr),
            (args.gripper_key, gripper_arr),
            (args.action_key, action_arr),
        ]:
            if int(arr.shape[0]) != total_len:
                raise ValueError(
                    f"data/{key} length {arr.shape[0]} does not match "
                    f"data/{args.image_key} length {total_len}"
                )
        for key, arr in [
            (args.magnet_key, magnet_arr),
            (args.magnet_sample_count_key, sample_count_arr),
            (args.magnet_timestamp_key, magnet_timestamp_arr),
            (args.magnet2_key, magnet2_arr),
            (args.magnet2_sample_count_key, sample_count2_arr),
            (args.magnet2_timestamp_key, magnet_timestamp2_arr),
        ]:
            if arr is not None and int(arr.shape[0]) != total_len:
                raise ValueError(
                    f"data/{key} length {arr.shape[0]} does not match "
                    f"data/{args.image_key} length {total_len}"
                )

        indices, episode_ends = build_downsample_indices(
            src_episode_ends,
            total_len=total_len,
            downsample=args.temporal_downsample,
            max_steps=args.max_steps,
        )
        pose_jump_stats = None
        if args.filter_pose_jumps:
            indices, episode_ends, pose_jump_stats = filter_pose_jump_indices(
                indices=indices,
                episode_ends=episode_ends,
                pos_arr=pos_arr,
                rotvec_arr=rotvec_arr,
                pos_threshold_m=args.pose_jump_pos_threshold_m,
                rot_threshold_deg=args.pose_jump_rot_threshold_deg,
                drop_target_frame=args.pose_jump_drop_target_frame,
                min_episode_len=args.pose_jump_min_episode_len,
            )
        action_from_next_obs, action_source_detail = resolve_action_source(
            args,
            action_arr,
            pos_arr,
            rotvec_arr,
            gripper_arr,
            src_episode_ends,
        )
        action_target_indices = build_next_selected_indices(indices, episode_ends)
        print_summary(
            args,
            root,
            indices,
            episode_ends,
            action_from_next_obs=action_from_next_obs,
            action_source_detail=action_source_detail,
            pose_jump_stats=pose_jump_stats,
        )
        if args.dry_run:
            return

        output_dir = Path(args.output).expanduser()
        replay_buffer_path = output_dir / "replay_buffer.zarr"
        if replay_buffer_path.exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"{replay_buffer_path} already exists. Pass --overwrite to replace it."
                )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Keep the previous dataset intact until every output array and the
        # episode metadata have been written successfully. A failed conversion
        # must not leave a zero-initialized Zarr directory at the final path.
        temporary_replay_buffer_path = output_dir / (
            f".replay_buffer.zarr.tmp-{os.getpid()}"
        )
        if temporary_replay_buffer_path.exists():
            shutil.rmtree(temporary_replay_buffer_path)

        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)
        out_store = zarr.DirectoryStore(str(temporary_replay_buffer_path))
        out_root = zarr.group(store=out_store, overwrite=True)
        out_data = out_root.create_group("data")
        out_meta = out_root.create_group("meta")

        n = len(indices)
        height, width = args.image_size
        chunk = max(1, args.chunk_size)

        image_out = create_array(
            out_data,
            args.target_image_key,
            shape=(n, height, width, 3),
            dtype=np.uint8,
            chunks=(min(chunk, n), height, width, 3),
            compressor=compressor,
        )
        gripper_out = create_array(
            out_data,
            args.target_gripper_key,
            shape=(n, 1),
            dtype=np.float32,
            chunks=(min(max(chunk * 8, 1), n), 1),
            compressor=compressor,
        )

        if args.preset == "rdp10d":
            pose_dim = 9
            action_dim = 10
        else:
            pose_dim = 3
            action_dim = 4

        pose_out = create_array(
            out_data,
            args.target_pose_key,
            shape=(n, pose_dim),
            dtype=np.float32,
            chunks=(min(max(chunk * 8, 1), n), pose_dim),
            compressor=compressor,
        )
        action_out = create_array(
            out_data,
            "action",
            shape=(n, action_dim),
            dtype=np.float32,
            chunks=(min(max(chunk * 8, 1), n), action_dim),
            compressor=compressor,
        )

        tactile_out = None
        magnet_tactile_out = None
        magnet_wrench_out = None
        magnet_timestamp_out = None
        magnet2_tactile_out = None
        magnet2_wrench_out = None
        magnet_timestamp2_out = None
        if args.preset == "train_dp_default" and "tactile" not in magnet_outputs:
            tactile_out = create_array(
                out_data,
                "left_gripper1_marker_offset_emb",
                shape=(n, 15),
                dtype=np.float32,
                chunks=(min(max(chunk * 8, 1), n), 15),
                compressor=compressor,
            )
        if "tactile" in magnet_outputs:
            magnet_tactile_out = create_array(
                out_data,
                args.magnet_tactile_key,
                shape=(n, args.magnet_tactile_dim),
                dtype=np.float32,
                chunks=(min(max(chunk * 8, 1), n), args.magnet_tactile_dim),
                compressor=compressor,
            )
        if "wrench" in magnet_outputs:
            magnet_wrench_out = create_array(
                out_data,
                args.magnet_wrench_key,
                shape=(n, 6),
                dtype=np.float32,
                chunks=(min(max(chunk * 8, 1), n), 6),
                compressor=compressor,
            )
        if magnet_timestamp_arr is not None:
            magnet_timestamp_tail = (
                ()
                if combined_dual_magnet
                else tuple(magnet_timestamp_arr.shape[1:])
            )
            magnet_timestamp_out = create_array(
                out_data,
                args.magnet_timestamp_key,
                shape=(n,) + magnet_timestamp_tail,
                dtype=magnet_timestamp_arr.dtype,
                chunks=(min(max(chunk * 8, 1), n),) + magnet_timestamp_tail,
                compressor=compressor,
            )
        if "tactile" in magnet2_outputs:
            magnet2_tactile_out = create_array(
                out_data,
                args.magnet2_tactile_key,
                shape=(n, args.magnet2_tactile_dim),
                dtype=np.float32,
                chunks=(min(max(chunk * 8, 1), n), args.magnet2_tactile_dim),
                compressor=compressor,
            )
        if "wrench" in magnet2_outputs:
            magnet2_wrench_out = create_array(
                out_data,
                args.magnet2_wrench_key,
                shape=(n, 6),
                dtype=np.float32,
                chunks=(min(max(chunk * 8, 1), n), 6),
                compressor=compressor,
            )
        if magnet_timestamp2_arr is not None:
            magnet_timestamp2_tail = (
                ()
                if combined_dual_magnet
                else tuple(magnet_timestamp2_arr.shape[1:])
            )
            magnet_timestamp2_out = create_array(
                out_data,
                args.magnet2_timestamp_key,
                shape=(n,) + magnet_timestamp2_tail,
                dtype=magnet_timestamp2_arr.dtype,
                chunks=(min(max(chunk * 8, 1), n),) + magnet_timestamp2_tail,
                compressor=compressor,
            )

        timestamp_out = None
        if "timestamp" in root["data"]:
            timestamp_out = create_array(
                out_data,
                "timestamp",
                shape=(n,),
                dtype=root["data"]["timestamp"].dtype,
                chunks=(min(max(chunk * 8, 1), n),),
                compressor=compressor,
            )

        for out_start in range(0, n, chunk):
            out_end = min(out_start + chunk, n)
            src_idx = indices[out_start:out_end]

            images = read_rows(image_arr, src_idx)
            image_out[out_start:out_end] = resize_images(images, height, width)

            pos = read_rows(pos_arr, src_idx).astype(np.float32)
            rotvec = read_rows(rotvec_arr, src_idx).astype(np.float32)
            gripper = read_rows(gripper_arr, src_idx).astype(np.float32)
            source_action = read_rows(action_arr, src_idx).astype(np.float32)
            if action_from_next_obs:
                action_src_idx = action_target_indices[out_start:out_end]
                action_pos = read_rows(pos_arr, action_src_idx).astype(np.float32)
                action_rotvec = read_rows(rotvec_arr, action_src_idx).astype(np.float32)
                action_gripper = gripper_from_source_action(source_action, gripper)
                action = action_from_pose_gripper(action_pos, action_rotvec, action_gripper)
            else:
                action = source_action
            magnet = None
            sample_count = None
            magnet2 = None
            sample_count2 = None
            if magnet_outputs:
                magnet = read_rows(magnet_arr, src_idx).astype(np.float32)
                if sample_count_arr is not None:
                    sample_count = read_rows(sample_count_arr, src_idx)
            if magnet2_outputs:
                magnet2 = read_rows(magnet2_arr, src_idx).astype(np.float32)
                if sample_count2_arr is not None:
                    sample_count2 = read_rows(sample_count2_arr, src_idx)
            if combined_dual_magnet:
                if magnet is not None:
                    magnet = magnet[:, 0:1]
                if sample_count is not None:
                    sample_count = sample_count[:, 0:1]
                if magnet2 is not None:
                    magnet2 = magnet2[:, 1:2]
                if sample_count2 is not None:
                    sample_count2 = sample_count2[:, 1:2]

            if gripper.ndim == 1:
                gripper = gripper[:, None]
            gripper = gripper[:, :1]
            gripper_out[out_start:out_end] = gripper

            if args.preset == "rdp10d":
                pose_out[out_start:out_end] = pose9_from_pos_rotvec(pos[:, :3], rotvec[:, :3])
                action_out[out_start:out_end] = action10_from_source(action, gripper)
            else:
                pose_out[out_start:out_end] = pos[:, :3]
                action_out[out_start:out_end] = action4_from_source(action, gripper)
                if tactile_out is not None:
                    tactile_out[out_start:out_end] = np.zeros((out_end - out_start, 15), dtype=np.float32)

            if magnet_tactile_out is not None:
                magnet_tactile_out[out_start:out_end] = magnet_to_tactile_embedding(
                    magnet,
                    sample_count=sample_count,
                    output_dim=args.magnet_tactile_dim,
                )
            if magnet_wrench_out is not None:
                magnet_wrench_out[out_start:out_end] = magnet_to_wrench(
                    magnet,
                    sample_count=sample_count,
                )
            if magnet_timestamp_out is not None:
                magnet_timestamps = read_rows(magnet_timestamp_arr, src_idx)
                if combined_dual_magnet:
                    magnet_timestamps = magnet_timestamps[:, 0]
                magnet_timestamp_out[out_start:out_end] = magnet_timestamps
            if magnet2_tactile_out is not None:
                magnet2_tactile_out[out_start:out_end] = magnet_to_tactile_embedding(
                    magnet2,
                    sample_count=sample_count2,
                    output_dim=args.magnet2_tactile_dim,
                )
            if magnet2_wrench_out is not None:
                magnet2_wrench_out[out_start:out_end] = magnet_to_wrench(
                    magnet2,
                    sample_count=sample_count2,
                )
            if magnet_timestamp2_out is not None:
                magnet_timestamps2 = read_rows(magnet_timestamp2_arr, src_idx)
                if combined_dual_magnet:
                    magnet_timestamps2 = magnet_timestamps2[:, 1]
                magnet_timestamp2_out[out_start:out_end] = magnet_timestamps2

            if timestamp_out is not None:
                timestamp_out[out_start:out_end] = read_rows(root["data"]["timestamp"], src_idx)

            print(f"\rConverted {out_end}/{n} rows", end="", flush=True)

        print()
        out_meta.create_dataset(
            "episode_ends",
            data=episode_ends.astype(np.int64),
            dtype=np.int64,
            chunks=(min(len(episode_ends), 1024),),
            compressor=compressor,
            overwrite=True,
        )

        if args.validate:
            validate_output(
                temporary_replay_buffer_path,
                args,
                n,
                episode_ends,
                magnet_outputs,
                magnet2_outputs,
            )

        if replay_buffer_path.exists():
            shutil.rmtree(replay_buffer_path)
        os.replace(temporary_replay_buffer_path, replay_buffer_path)
        temporary_replay_buffer_path = None

        print(f"Done. RDP dataset directory: {output_dir}")
        print(f"RDP replay buffer: {replay_buffer_path}")
    finally:
        if temporary_replay_buffer_path is not None and temporary_replay_buffer_path.exists():
            shutil.rmtree(temporary_replay_buffer_path)
        close = getattr(input_store, "close", None)
        if close is not None:
            close()


def validate_output(
    replay_buffer_path,
    args,
    n,
    episode_ends,
    magnet_outputs=None,
    magnet2_outputs=None,
):
    root = zarr.open(str(replay_buffer_path), mode="r")
    if magnet_outputs is None:
        magnet_outputs = []
    if magnet2_outputs is None:
        magnet2_outputs = []
    required = [
        args.target_image_key,
        args.target_pose_key,
        args.target_gripper_key,
        "action",
    ]
    if args.preset == "train_dp_default" and "tactile" not in magnet_outputs:
        required.append("left_gripper1_marker_offset_emb")
    if "tactile" in magnet_outputs:
        required.append(args.magnet_tactile_key)
    if "wrench" in magnet_outputs:
        required.append(args.magnet_wrench_key)
    if "tactile" in magnet2_outputs:
        required.append(args.magnet2_tactile_key)
    if "wrench" in magnet2_outputs:
        required.append(args.magnet2_wrench_key)
    if args.copy_magnet_timestamps:
        required.append(args.magnet_timestamp_key)
        if magnet2_outputs:
            required.append(args.magnet2_timestamp_key)
    for key in required:
        if key not in root["data"]:
            raise AssertionError(f"Converted dataset missing data/{key}")
        if root["data"][key].shape[0] != n:
            raise AssertionError(f"data/{key} has wrong length: {root['data'][key].shape[0]} != {n}")
    converted_episode_ends = root["meta"]["episode_ends"][:]
    if not np.array_equal(converted_episode_ends, episode_ends):
        raise AssertionError("meta/episode_ends does not match expected converted episode ends")
    if int(converted_episode_ends[-1]) != n:
        raise AssertionError(f"Last episode end {converted_episode_ends[-1]} != data length {n}")
    print("Validation passed.")


def main():
    args = parse_args()
    convert(args)


if __name__ == "__main__":
    main()
