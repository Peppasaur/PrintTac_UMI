#!/usr/bin/env python3
"""
Replay one Franka trajectory from a zarr dataset through the local
RDP-compatible Polymetis HTTP robot server.

The default dataset path matches this repository:
    dataset/dataset.zarr.zip

By default the script is a dry run. Pass --execute to send TCP waypoints to:
    http://127.0.0.1:8092/move_tcp/left
"""
import argparse
import os
import sys
import threading
import time

import numpy as np
import requests
import scipy.spatial.transform as st
import zarr

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from reactive_diffusion_policy.common.precise_sleep import precise_wait


def wxyz_to_xyzw(quat):
    quat = np.asarray(quat, dtype=np.float64)
    if quat.ndim == 1:
        return np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float64)
    return np.stack([quat[:, 1], quat[:, 2], quat[:, 3], quat[:, 0]], axis=-1)


def xyzw_to_wxyz(quat):
    quat = np.asarray(quat, dtype=np.float64)
    if quat.ndim == 1:
        return np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float64)
    return np.stack([quat[:, 3], quat[:, 0], quat[:, 1], quat[:, 2]], axis=-1)


def normalize_pose7d(poses):
    poses = np.asarray(poses, dtype=np.float64).copy()
    quat_norm = np.linalg.norm(poses[..., 3:7], axis=-1, keepdims=True)
    quat_norm = np.maximum(quat_norm, 1e-12)
    poses[..., 3:7] = poses[..., 3:7] / quat_norm
    return poses


def rotvec_pose_to_pose7d(poses6):
    poses6 = np.asarray(poses6, dtype=np.float64)
    if poses6.ndim != 2 or poses6.shape[1] != 6:
        raise ValueError(f"Expected Nx6 rotvec poses, got {poses6.shape}")
    quat_wxyz = xyzw_to_wxyz(st.Rotation.from_rotvec(poses6[:, 3:6]).as_quat())
    return normalize_pose7d(np.concatenate([poses6[:, :3], quat_wxyz], axis=-1))


def normalize_vectors(v):
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(norm, 1e-12)


def ortho6d_to_rotation_matrix(ortho6d):
    ortho6d = np.asarray(ortho6d, dtype=np.float64)
    x_raw = ortho6d[:, 0:3]
    y_raw = ortho6d[:, 3:6]
    x = normalize_vectors(x_raw)
    z = normalize_vectors(np.cross(x, y_raw))
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=-1)


def ortho9_pose_to_pose7d(poses9):
    poses9 = np.asarray(poses9, dtype=np.float64)
    if poses9.ndim != 2 or poses9.shape[1] != 9:
        raise ValueError(f"Expected Nx9 ortho6d poses, got {poses9.shape}")
    rot_mats = ortho6d_to_rotation_matrix(poses9[:, 3:9])
    quat_wxyz = xyzw_to_wxyz(st.Rotation.from_matrix(rot_mats).as_quat())
    return normalize_pose7d(np.concatenate([poses9[:, :3], quat_wxyz], axis=-1))


def get_zarr_root(input_path):
    input_path = os.path.expanduser(input_path)
    if input_path.endswith(".zip"):
        store = zarr.ZipStore(input_path, mode="r")
        return zarr.group(store), store
    return zarr.open(input_path, mode="r"), None


def get_episode_slice(episode_ends, episode_idx):
    if episode_idx < 0 or episode_idx >= len(episode_ends):
        raise ValueError(
            f"episode_idx must be in [0, {len(episode_ends) - 1}], got {episode_idx}"
        )
    start = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
    end = int(episode_ends[episode_idx])
    return slice(start, end), start, end


def infer_action_gripper_index(action):
    dim = action.shape[1]
    if dim in (4, 7, 10):
        return dim - 1
    if dim > 10:
        return 3
    if dim > 6:
        return 6
    return None


def build_pose7d_from_action_like(data, name, episode_slice, reference_pose7d=None):
    action = data[name][episode_slice].astype(np.float64)
    if action.ndim != 2:
        raise ValueError(f"Expected data/{name} to be 2D, got {action.shape}")

    dim = action.shape[1]
    if dim == 4:
        if reference_pose7d is None:
            raise ValueError(
                f"data/{name} is Nx4 xyz+gripper, but no observation orientation "
                "is available to complete a 7D TCP pose."
            )
        pose7d = reference_pose7d.copy()
        pose7d[:, :3] = action[:, :3]
        return pose7d, f"{name}[:, :3] + observation orientation", action

    if dim == 10:
        pose9 = action[:, :9]
        return ortho9_pose_to_pose7d(pose9), f"{name}[:, :9] xyz+ortho6d", action

    if dim >= 6:
        return rotvec_pose_to_pose7d(action[:, :6]), f"{name}[:, :6] xyz+rotvec", action

    raise ValueError(f"Unsupported data/{name} shape {action.shape}")


def load_observation_pose7d(data, episode_slice):
    if "robot0_eef_pos" in data and "robot0_eef_rot_axis_angle" in data:
        pos = data["robot0_eef_pos"][episode_slice].astype(np.float64)
        rotvec = data["robot0_eef_rot_axis_angle"][episode_slice].astype(np.float64)
        poses6 = np.concatenate([pos, rotvec], axis=-1)
        return rotvec_pose_to_pose7d(poses6), "robot0_eef_pos + robot0_eef_rot_axis_angle"

    if "left_robot_tcp_pose" in data:
        pose = data["left_robot_tcp_pose"][episode_slice].astype(np.float64)
        if pose.ndim != 2:
            raise ValueError(f"Expected data/left_robot_tcp_pose to be 2D, got {pose.shape}")
        if pose.shape[1] == 9:
            return ortho9_pose_to_pose7d(pose), "left_robot_tcp_pose xyz+ortho6d"
        if pose.shape[1] == 6:
            return rotvec_pose_to_pose7d(pose), "left_robot_tcp_pose xyz+rotvec"
        if pose.shape[1] == 7:
            return normalize_pose7d(pose), "left_robot_tcp_pose xyz+quat_wxyz"
        raise ValueError(f"Unsupported data/left_robot_tcp_pose shape {pose.shape}")

    return None, None


def load_gripper_signal(data, episode_slice, action, gripper_source, gripper_action_index):
    if gripper_source in ("auto", "action") and action is not None:
        idx = infer_action_gripper_index(action) if gripper_action_index is None else gripper_action_index
        if idx is not None:
            if idx < 0:
                idx += action.shape[1]
            if idx < 0 or idx >= action.shape[1]:
                raise ValueError(
                    f"--gripper-action-index {gripper_action_index} is out of range "
                    f"for action shape {action.shape}"
                )
            return action[:, idx].astype(np.float64), f"action[:, {idx}]"
        if gripper_source == "action":
            raise ValueError(f"Cannot infer gripper column from action shape {action.shape}")

    if gripper_source in ("auto", "obs"):
        for key in ("robot0_gripper_width", "left_robot_gripper_width"):
            if key in data:
                signal = data[key][episode_slice].astype(np.float64)
                return np.squeeze(signal), key
        if gripper_source == "obs":
            raise KeyError(
                "No observation gripper key found. Expected robot0_gripper_width "
                "or left_robot_gripper_width."
            )

    return None, None


def load_episode(input_path, episode_idx, pose_source, gripper_source,
        fallback_frequency, gripper_action_index):
    root, store = get_zarr_root(input_path)
    try:
        if "data" not in root or "meta" not in root:
            raise KeyError("Dataset must contain data and meta groups")
        data = root["data"]
        episode_ends = root["meta"]["episode_ends"][:].astype(np.int64)
        episode_slice, start, end = get_episode_slice(episode_ends, episode_idx)

        obs_pose7d, obs_pose_source = load_observation_pose7d(data, episode_slice)
        action = None

        if pose_source == "auto":
            if "action" in data:
                pose_source = "action"
            elif obs_pose7d is not None:
                pose_source = "obs"
            elif "target" in data:
                pose_source = "target"
            else:
                raise KeyError("No supported pose source found in dataset")

        if pose_source == "obs":
            if obs_pose7d is None:
                raise KeyError("No supported observation pose keys found in dataset")
            poses7d = obs_pose7d
            actual_pose_source = obs_pose_source
        elif pose_source in ("action", "target"):
            if pose_source not in data:
                raise KeyError(f"Dataset has no data/{pose_source} array")
            poses7d, actual_pose_source, action = build_pose7d_from_action_like(
                data=data,
                name=pose_source,
                episode_slice=episode_slice,
                reference_pose7d=obs_pose7d,
            )
        else:
            raise ValueError(f"Unknown pose source {pose_source}")

        gripper_signal, actual_gripper_source = load_gripper_signal(
            data=data,
            episode_slice=episode_slice,
            action=action,
            gripper_source=gripper_source,
            gripper_action_index=gripper_action_index,
        )

        if "timestamp" in data:
            timestamps = data["timestamp"][episode_slice].astype(np.float64)
            timestamps = timestamps - timestamps[0]
            timestamp_source = "timestamp"
        else:
            timestamps = np.arange(len(poses7d), dtype=np.float64) / fallback_frequency
            timestamp_source = f"fallback {fallback_frequency:g}Hz"

        data_keys = list(data.keys())
    finally:
        if store is not None:
            store.close()

    if poses7d.ndim != 2 or poses7d.shape[1] != 7:
        raise ValueError(f"Expected Nx7 TCP poses, got {poses7d.shape}")
    if len(poses7d) < 2:
        raise ValueError("Episode must contain at least 2 poses")
    if gripper_signal is not None:
        gripper_signal = np.asarray(gripper_signal, dtype=np.float64)
        if gripper_signal.ndim != 1:
            raise ValueError(f"Expected 1D gripper signal, got {gripper_signal.shape}")
        if len(gripper_signal) != len(poses7d):
            raise ValueError(
                f"Gripper signal length {len(gripper_signal)} does not match "
                f"pose length {len(poses7d)}"
            )
    if np.any(~np.isfinite(poses7d)):
        raise ValueError("TCP poses contain NaN or inf")
    if np.any(~np.isfinite(timestamps)):
        raise ValueError("Timestamps contain NaN or inf")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("Episode timestamps must be strictly increasing")

    return {
        "poses7d": poses7d,
        "timestamps": timestamps,
        "gripper_signal": gripper_signal,
        "pose_source": actual_pose_source,
        "gripper_source": actual_gripper_source,
        "timestamp_source": timestamp_source,
        "episode_start": start,
        "episode_end": end,
        "episode_count": len(episode_ends),
        "data_keys": data_keys,
    }


def select_time_window(poses7d, timestamps, gripper_signal, start_index, end_index,
        max_steps):
    n_steps = len(poses7d)
    if start_index < 0 or start_index >= n_steps:
        raise ValueError(f"--start-index must be in [0, {n_steps - 1}], got {start_index}")

    if end_index is None:
        end_index = n_steps
    if end_index <= start_index or end_index > n_steps:
        raise ValueError(f"--end-index must be in ({start_index}, {n_steps}], got {end_index}")

    if max_steps is not None:
        if max_steps < 2:
            raise ValueError("--max-steps must be at least 2")
        end_index = min(end_index, start_index + max_steps)

    if end_index - start_index < 2:
        raise ValueError("Selected replay window must contain at least 2 poses")

    poses7d = poses7d[start_index:end_index].copy()
    timestamps = timestamps[start_index:end_index].copy()
    if gripper_signal is not None:
        gripper_signal = gripper_signal[start_index:end_index].copy()
    timestamps -= timestamps[0]
    return poses7d, timestamps, gripper_signal, start_index, end_index


def infer_gripper_mode(signal, stroke):
    signal = np.asarray(signal, dtype=np.float64)
    sig_min = float(np.nanmin(signal))
    sig_max = float(np.nanmax(signal))
    if sig_min >= -1e-6 and sig_max <= stroke * 1.25:
        return "width"
    if sig_min >= -1e-6 and sig_max <= 1.05:
        return "open_ratio"
    return "width"


def gripper_signal_to_width(signal, mode, stroke):
    signal = np.asarray(signal, dtype=np.float64)
    if not np.all(np.isfinite(signal)):
        raise ValueError("Gripper signal contains NaN or inf")

    actual_mode = infer_gripper_mode(signal, stroke) if mode == "auto" else mode
    if actual_mode == "width":
        width = signal
    elif actual_mode == "open_ratio":
        width = np.clip(signal, 0.0, 1.0) * stroke
    elif actual_mode == "close_ratio":
        width = (1.0 - np.clip(signal, 0.0, 1.0)) * stroke
    else:
        raise ValueError(f"Unknown gripper mode {mode}")

    return np.clip(width, 0.0, stroke), actual_mode


def build_gripper_commands(widths, timestamps, threshold):
    commands = []
    last_width = None
    for width, timestamp in zip(widths, timestamps):
        width = float(width)
        if last_width is None or threshold <= 0 or abs(width - last_width) >= threshold:
            commands.append((float(timestamp), width))
            last_width = width
    return commands


def pose_error(current_pose, target_pose):
    current_pose = normalize_pose7d(np.asarray(current_pose, dtype=np.float64))
    target_pose = normalize_pose7d(np.asarray(target_pose, dtype=np.float64))
    pos_error = float(np.linalg.norm(current_pose[:3] - target_pose[:3]))
    current_rot = st.Rotation.from_quat(wxyz_to_xyzw(current_pose[3:7]))
    target_rot = st.Rotation.from_quat(wxyz_to_xyzw(target_pose[3:7]))
    rot_error = float((target_rot * current_rot.inv()).magnitude())
    return pos_error, rot_error


def trajectory_stats(poses7d, timestamps):
    dt = np.diff(timestamps)
    dpos = np.linalg.norm(np.diff(poses7d[:, :3], axis=0), axis=1)
    rotations = st.Rotation.from_quat(wxyz_to_xyzw(poses7d[:, 3:7]))
    drot = (rotations[1:] * rotations[:-1].inv()).magnitude()
    return {
        "duration": float(timestamps[-1]),
        "frequency": float(1.0 / np.median(dt)),
        "min_dt": float(np.min(dt)),
        "max_dt": float(np.max(dt)),
        "max_pos_step": float(np.max(dpos)),
        "max_rot_step": float(np.max(drot)),
        "max_pos_speed": float(np.max(dpos / dt)),
        "max_rot_speed": float(np.max(drot / dt)),
        "xyz_min": np.min(poses7d[:, :3], axis=0),
        "xyz_max": np.max(poses7d[:, :3], axis=0),
    }


def format_array(arr, precision=4):
    return np.array2string(np.asarray(arr), precision=precision, suppress_small=False)


def get_current_tcp(session, base_url, timeout):
    response = session.get(f"{base_url}/get_current_tcp/left", timeout=timeout)
    response.raise_for_status()
    pose = np.asarray(response.json(), dtype=np.float64)
    if pose.shape != (7,):
        raise ValueError(f"Expected current TCP shape (7,), got {pose.shape}")
    return normalize_pose7d(pose)


def post_tcp(session, base_url, pose7d, timeout):
    payload = {"target_tcp": [float(x) for x in pose7d]}
    response = session.post(f"{base_url}/move_tcp/left", json=payload, timeout=timeout)
    response.raise_for_status()
    return response


def post_gripper(session, base_url, width, velocity, force_limit, timeout):
    payload = {
        "width": float(width),
        "velocity": float(velocity),
        "force_limit": float(force_limit),
    }
    response = session.post(f"{base_url}/move_gripper/left", json=payload, timeout=timeout)
    response.raise_for_status()
    return response


def interpolate_pose7d(start_pose, end_pose, alpha):
    alpha = np.asarray(alpha, dtype=np.float64)
    start_pose = normalize_pose7d(start_pose)
    end_pose = normalize_pose7d(end_pose)
    pos = start_pose[:3][None, :] * (1.0 - alpha[:, None]) + end_pose[:3][None, :] * alpha[:, None]
    slerp = st.Slerp(
        [0.0, 1.0],
        st.Rotation.from_quat([
            wxyz_to_xyzw(start_pose[3:7]),
            wxyz_to_xyzw(end_pose[3:7]),
        ]),
    )
    quat_wxyz = xyzw_to_wxyz(slerp(alpha).as_quat())
    return normalize_pose7d(np.concatenate([pos, quat_wxyz], axis=-1))


def move_to_start(session, base_url, current_pose, start_pose, duration, frequency,
        timeout, verbose):
    if duration <= 0:
        post_tcp(session, base_url, start_pose, timeout)
        return

    n_steps = max(2, int(np.ceil(duration * frequency)))
    alphas = np.linspace(0.0, 1.0, n_steps)
    waypoints = interpolate_pose7d(current_pose, start_pose, alphas)
    t0 = time.monotonic()
    for i, waypoint in enumerate(waypoints[1:], start=1):
        precise_wait(t0 + i * duration / (n_steps - 1), time_func=time.monotonic)
        post_tcp(session, base_url, waypoint, timeout)
        if verbose and (i % max(1, n_steps // 10) == 0 or i == n_steps - 1):
            print(f"Move-to-start command {i}/{n_steps - 1}")


class GripperReplayThread(threading.Thread):
    def __init__(self, base_url, commands, replay_start_time, send_ahead,
            velocity, force_limit, timeout, verbose):
        super().__init__(name="FrankaGripperReplayThread", daemon=True)
        self.base_url = base_url
        self.commands = commands
        self.replay_start_time = replay_start_time
        self.send_ahead = send_ahead
        self.velocity = velocity
        self.force_limit = force_limit
        self.timeout = timeout
        self.verbose = verbose
        self.error = None
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        session = requests.Session()
        try:
            for i, (timestamp, width) in enumerate(self.commands):
                if self._stop_event.is_set():
                    break
                send_time = self.replay_start_time + timestamp - self.send_ahead
                if send_time > time.monotonic():
                    precise_wait(send_time, time_func=time.monotonic)
                if self._stop_event.is_set():
                    break
                post_gripper(
                    session=session,
                    base_url=self.base_url,
                    width=width,
                    velocity=self.velocity,
                    force_limit=self.force_limit,
                    timeout=self.timeout,
                )
                if self.verbose:
                    print(f"Gripper command {i + 1}/{len(self.commands)}: width={width:.4f}m")
        except Exception as exc:
            self.error = exc
        finally:
            session.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay dataset/dataset.zarr.zip on a Franka Polymetis HTTP controller."
    )
    parser.add_argument("--input", "-i", default="dataset/dataset.zarr.zip")
    parser.add_argument("--replay-episode", "-re", type=int, default=0)
    parser.add_argument(
        "--pose-source",
        choices=["auto", "action", "obs", "target"],
        default="auto",
        help="auto prefers data/action; obs supports UMI robot0_* and RDP left_robot_tcp_pose.",
    )
    parser.add_argument(
        "--gripper-source",
        choices=["auto", "action", "obs", "none"],
        default="auto",
    )
    parser.add_argument(
        "--gripper-action-index",
        type=int,
        default=None,
        help="Override the inferred gripper column for data/action.",
    )
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8092)
    parser.add_argument(
        "--fallback-data-frequency",
        type=float,
        default=25.0,
        help="Used only when the dataset has no timestamp array.",
    )
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--z-offset", type=float, default=0.0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--move-to-start-duration", type=float, default=5.0)
    parser.add_argument("--move-to-start-frequency", type=float, default=30.0)
    parser.add_argument("--skip-move-to-start", action="store_true")
    parser.add_argument("--start-delay", type=float, default=1.0)
    parser.add_argument("--hold-time", type=float, default=1.0)
    parser.add_argument(
        "--send-ahead",
        type=float,
        default=0.02,
        help="Seconds to send commands before each dataset timestamp.",
    )
    parser.add_argument("--enable-gripper", action="store_true")
    parser.add_argument(
        "--gripper-signal-mode",
        choices=["auto", "width", "open_ratio", "close_ratio"],
        default="auto",
    )
    parser.add_argument("--gripper-stroke", type=float, default=0.085)
    parser.add_argument("--gripper-velocity", type=float, default=0.08)
    parser.add_argument("--gripper-force-limit", type=float, default=40.0)
    parser.add_argument("--gripper-command-threshold", type=float, default=0.002)
    parser.add_argument("--tcp-request-timeout", type=float, default=0.5)
    parser.add_argument("--gripper-request-timeout", type=float, default=2.0)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually send commands to the robot server. Default is dry run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and print trajectory information without robot commands.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.time_scale <= 0:
        raise ValueError("--time-scale must be positive")
    if args.fallback_data_frequency <= 0:
        raise ValueError("--fallback-data-frequency must be positive")
    if args.move_to_start_frequency <= 0:
        raise ValueError("--move-to-start-frequency must be positive")
    if args.gripper_source == "none":
        args.enable_gripper = False

    episode = load_episode(
        input_path=args.input,
        episode_idx=args.replay_episode,
        pose_source=args.pose_source,
        gripper_source=args.gripper_source,
        fallback_frequency=args.fallback_data_frequency,
        gripper_action_index=args.gripper_action_index,
    )
    full_episode_len = len(episode["poses7d"])
    poses7d, timestamps, gripper_signal, start_index, end_index = select_time_window(
        poses7d=episode["poses7d"],
        timestamps=episode["timestamps"],
        gripper_signal=episode["gripper_signal"],
        start_index=args.start_index,
        end_index=args.end_index,
        max_steps=args.max_steps,
    )
    poses7d[:, 2] += args.z_offset
    timestamps = timestamps * args.time_scale

    gripper_widths = None
    gripper_mode = None
    gripper_commands = []
    if gripper_signal is not None:
        gripper_widths, gripper_mode = gripper_signal_to_width(
            gripper_signal,
            mode=args.gripper_signal_mode,
            stroke=args.gripper_stroke,
        )
        gripper_commands = build_gripper_commands(
            widths=gripper_widths,
            timestamps=timestamps,
            threshold=args.gripper_command_threshold,
        )
    if args.enable_gripper and gripper_widths is None:
        raise ValueError("Gripper replay enabled, but no gripper signal was found")

    stats = trajectory_stats(poses7d, timestamps)
    base_url = f"http://{args.server_host}:{args.server_port}"
    dry_run = args.dry_run or not args.execute

    print(
        f"Loaded episode {args.replay_episode + 1}/{episode['episode_count']}: "
        f"{len(poses7d)}/{full_episode_len} poses, indices=[{start_index}, {end_index}), "
        f"dataset rows=[{episode['episode_start']}, {episode['episode_end']})"
    )
    print(f"Dataset keys: {', '.join(episode['data_keys'])}")
    print(f"Pose source: {episode['pose_source']}")
    print(f"Timestamp source: {episode['timestamp_source']}")
    print(
        f"Duration: {stats['duration']:.3f}s, median frequency: {stats['frequency']:.2f}Hz, "
        f"dt=[{stats['min_dt']:.4f}, {stats['max_dt']:.4f}]s"
    )
    print(
        f"XYZ range: min={format_array(stats['xyz_min'])}, max={format_array(stats['xyz_max'])}, "
        f"z_offset={args.z_offset:g}"
    )
    print(
        f"Max step: {stats['max_pos_step']:.4f}m, {np.rad2deg(stats['max_rot_step']):.2f}deg; "
        f"max speed: {stats['max_pos_speed']:.4f}m/s, "
        f"{np.rad2deg(stats['max_rot_speed']):.2f}deg/s"
    )
    print(f"Start TCP xyz+qwxyz: {format_array(poses7d[0])}")
    print(f"End TCP xyz+qwxyz:   {format_array(poses7d[-1])}")

    if gripper_signal is not None:
        print(
            f"Gripper signal: source={episode['gripper_source']}, mode={gripper_mode}, "
            f"raw=[{np.min(gripper_signal):.4f}, {np.max(gripper_signal):.4f}], "
            f"width=[{np.min(gripper_widths):.4f}, {np.max(gripper_widths):.4f}]m, "
            f"commands={len(gripper_commands)}"
        )
        if args.execute and not args.enable_gripper:
            print(
                "Gripper replay is disabled. Add --enable-gripper to send "
                "these commands to /move_gripper/left."
            )
    else:
        print("Gripper signal: not found")

    if dry_run:
        print("Dry run only; pass --execute to send commands to the robot server.")
        return

    print(f"Robot server: {base_url}")
    session = requests.Session()
    gripper_thread = None
    try:
        current_pose = get_current_tcp(
            session=session,
            base_url=base_url,
            timeout=args.tcp_request_timeout,
        )
        pos_error, rot_error = pose_error(current_pose, poses7d[0])
        print(
            f"Current TCP: {format_array(current_pose)}; "
            f"distance to start={pos_error:.4f}m, {np.rad2deg(rot_error):.2f}deg"
        )

        if args.skip_move_to_start:
            print("Skipping move-to-start.")
        else:
            print(f"Moving to dataset start over {args.move_to_start_duration:.2f}s...")
            move_to_start(
                session=session,
                base_url=base_url,
                current_pose=current_pose,
                start_pose=poses7d[0],
                duration=args.move_to_start_duration,
                frequency=args.move_to_start_frequency,
                timeout=args.tcp_request_timeout,
                verbose=args.verbose,
            )
            if args.hold_time > 0:
                time.sleep(min(args.hold_time, 1.0))

        if args.enable_gripper and gripper_widths is not None:
            print(f"Priming gripper to {gripper_widths[0]:.4f}m...")
            post_gripper(
                session=session,
                base_url=base_url,
                width=gripper_widths[0],
                velocity=args.gripper_velocity,
                force_limit=args.gripper_force_limit,
                timeout=args.gripper_request_timeout,
            )

        replay_start_time = time.monotonic() + args.start_delay
        if args.enable_gripper and gripper_commands:
            gripper_thread = GripperReplayThread(
                base_url=base_url,
                commands=gripper_commands,
                replay_start_time=replay_start_time,
                send_ahead=args.send_ahead,
                velocity=args.gripper_velocity,
                force_limit=args.gripper_force_limit,
                timeout=args.gripper_request_timeout,
                verbose=args.verbose,
            )
            gripper_thread.start()

        print(f"Replaying trajectory in {args.start_delay:.2f}s...")
        for i, (pose, timestamp) in enumerate(zip(poses7d, timestamps)):
            if gripper_thread is not None and gripper_thread.error is not None:
                raise RuntimeError("Gripper replay failed") from gripper_thread.error

            send_time = replay_start_time + float(timestamp) - args.send_ahead
            if send_time > time.monotonic():
                precise_wait(send_time, time_func=time.monotonic)
            post_tcp(
                session=session,
                base_url=base_url,
                pose7d=pose,
                timeout=args.tcp_request_timeout,
            )
            if args.verbose and (i % 50 == 0 or i == len(poses7d) - 1):
                print(f"Sent TCP waypoint {i + 1}/{len(poses7d)}")

        precise_wait(
            replay_start_time + float(timestamps[-1]) + args.hold_time,
            time_func=time.monotonic,
        )
        if gripper_thread is not None:
            gripper_thread.join(timeout=1.0)
            if gripper_thread.error is not None:
                raise RuntimeError("Gripper replay failed") from gripper_thread.error

        final_pose = get_current_tcp(
            session=session,
            base_url=base_url,
            timeout=args.tcp_request_timeout,
        )
        pos_error, rot_error = pose_error(final_pose, poses7d[-1])
        print(
            f"Replay finished. Final TCP error={pos_error:.4f}m, "
            f"{np.rad2deg(rot_error):.2f}deg"
        )
    finally:
        if gripper_thread is not None and gripper_thread.is_alive():
            gripper_thread.stop()
            gripper_thread.join(timeout=1.0)
        session.close()


if __name__ == "__main__":
    main()
