import csv
import glob
import json
import os
import re
import socket
import struct
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from select import select
from types import SimpleNamespace
from typing import Dict, Optional

import cv2
import numpy as np
import requests
import scipy.spatial.transform as st
import zarr
from loguru import logger

try:
    import av
except Exception as exc:  # pragma: no cover - runtime dependency for iPhone UDP video only
    av = None
    AV_IMPORT_ERROR = exc
else:
    AV_IMPORT_ERROR = None

from reactive_diffusion_policy.common.precise_sleep import precise_wait
from reactive_diffusion_policy.common.space_utils import (
    pose_6d_to_pose_7d,
    pose_7d_to_pose_6d,
    pose_6d_to_pose_9d,
)


MAGNET_SENSOR_COUNT = 5
MAGNET_USED_SENSOR_COUNT = 4
MAGNET_VALUES_PER_SENSOR = 4
MAGNET_SAMPLES_PER_FRAME = 8
MAGNET_ABNORMAL_ABS_THRESHOLD = 5000.0
MAGNET_FLOAT_PATTERN = re.compile(r"[+-]?\d+\.\d{2}")
IPHONE_VIDEO_PACKET_HEADER_V1 = struct.Struct("<4sBBHIdHHHH")
IPHONE_VIDEO_PACKET_HEADER_V2 = struct.Struct("<4sBBHIdHHHHffffHH")
IPHONE_VIDEO_PACKET_HEADER = IPHONE_VIDEO_PACKET_HEADER_V1
IPHONE_VIDEO_MAGIC_V1 = b"APV1"
IPHONE_VIDEO_MAGIC_V2 = b"APV2"
IPHONE_VIDEO_MAGIC = IPHONE_VIDEO_MAGIC_V1
IPHONE_VIDEO_VERSION = 1
IPHONE_VIDEO_VERSION_V2 = 2
IPHONE_FRAME_STALE_SECONDS = 0.20
IPHONE_MAX_INFLIGHT_FRAMES = 8


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(value)


def _open_zarr(path):
    path = os.path.expanduser(str(path))
    if path.endswith(".zip"):
        store = zarr.ZipStore(path, mode="r")
        return zarr.group(store), store
    return zarr.open(path, mode="r"), None


def _dataset_candidates(dataset_path):
    if dataset_path is None:
        return []
    dataset_path = os.path.expanduser(str(dataset_path))
    candidates = []
    if os.path.isdir(dataset_path):
        replay_buffer_path = os.path.join(dataset_path, "replay_buffer.zarr")
        if os.path.exists(replay_buffer_path):
            candidates.append(replay_buffer_path)
        if (
            dataset_path.endswith(".zarr")
            or os.path.exists(os.path.join(dataset_path, ".zgroup"))
            or os.path.exists(os.path.join(dataset_path, ".zarray"))
        ):
            candidates.append(dataset_path)
    else:
        candidates.append(dataset_path)
    return candidates


def _get_episode_slice(episode_ends, episode_idx):
    if episode_idx < 0 or episode_idx >= len(episode_ends):
        raise ValueError(
            f"episode_idx must be in [0, {len(episode_ends) - 1}], got {episode_idx}"
        )
    start = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
    end = int(episode_ends[episode_idx])
    return slice(start, end), start, end


def _normalize_pose7d(pose):
    pose = np.asarray(pose, dtype=np.float64).reshape(7)
    quat = pose[3:7]
    norm = np.linalg.norm(quat)
    if norm <= 1e-9:
        raise ValueError(f"Invalid zero-norm quaternion in pose {pose}")
    pose = pose.copy()
    pose[3:7] = quat / norm
    return pose


def _pose9_to_pose7d(pose):
    pose = np.asarray(pose, dtype=np.float64).reshape(9)
    rot = pose[3:9]
    x_raw = rot[:3]
    y_raw = rot[3:6]
    x = x_raw / max(np.linalg.norm(x_raw), 1e-9)
    z = np.cross(x, y_raw)
    z = z / max(np.linalg.norm(z), 1e-9)
    y = np.cross(z, x)
    mat = np.stack([x, y, z], axis=1)
    quat_xyzw = st.Rotation.from_matrix(mat).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return _normalize_pose7d(np.concatenate([pose[:3], quat_wxyz]))


def _matrix_to_pose7d(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (16,):
        matrix = matrix.reshape(4, 4)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 pose matrix, got shape {matrix.shape}")
    quat_xyzw = st.Rotation.from_matrix(matrix[:3, :3]).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return _normalize_pose7d(np.concatenate([matrix[:3, 3], quat_wxyz]))


def _rotvec_pose_to_pose7d(pose):
    pose = np.asarray(pose, dtype=np.float64).reshape(6)
    quat_xyzw = st.Rotation.from_rotvec(pose[3:6]).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return _normalize_pose7d(np.concatenate([pose[:3], quat_wxyz]))


def _rpy_pose_to_pose7d(position, rpy):
    position = np.asarray(position, dtype=np.float64).reshape(3)
    quat_xyzw = st.Rotation.from_euler("xyz", np.asarray(rpy, dtype=np.float64).reshape(3)).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return _normalize_pose7d(np.concatenate([position, quat_wxyz]))


def _load_manual_start_pose(pose_path):
    pose_path = os.path.expanduser(str(pose_path))
    with open(pose_path, "r") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"Manual start pose JSON must contain an object, got {type(payload).__name__}")

    source = None
    if "O_T_EE" in payload:
        pose7d = _matrix_to_pose7d(payload["O_T_EE"])
        source = "O_T_EE 4x4 matrix"
    elif "pose7d" in payload:
        pose7d = _normalize_pose7d(payload["pose7d"])
        source = "pose7d xyz+quat_wxyz"
    elif "tcp_pose" in payload:
        tcp_pose = np.asarray(payload["tcp_pose"], dtype=np.float64).reshape(-1)
        if tcp_pose.shape[0] == 7:
            pose7d = _normalize_pose7d(tcp_pose)
            source = "tcp_pose xyz+quat_wxyz"
        elif tcp_pose.shape[0] == 6:
            pose7d = _rotvec_pose_to_pose7d(tcp_pose)
            source = "tcp_pose xyz+rotvec"
        elif tcp_pose.shape[0] == 9:
            pose7d = _pose9_to_pose7d(tcp_pose)
            source = "tcp_pose xyz+ortho6d"
        else:
            raise ValueError(f"Unsupported tcp_pose length {tcp_pose.shape[0]} in {pose_path}")
    elif "position_m" in payload and "quat_wxyz" in payload:
        pose7d = _normalize_pose7d(np.concatenate([
            np.asarray(payload["position_m"], dtype=np.float64).reshape(3),
            np.asarray(payload["quat_wxyz"], dtype=np.float64).reshape(4),
        ]))
        source = "position_m + quat_wxyz"
    elif "position_m" in payload and "quat_xyzw" in payload:
        quat_xyzw = np.asarray(payload["quat_xyzw"], dtype=np.float64).reshape(4)
        pose7d = _normalize_pose7d(np.concatenate([
            np.asarray(payload["position_m"], dtype=np.float64).reshape(3),
            np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]),
        ]))
        source = "position_m + quat_xyzw"
    elif "position_m" in payload and "rotvec" in payload:
        pose7d = _rotvec_pose_to_pose7d(np.concatenate([
            np.asarray(payload["position_m"], dtype=np.float64).reshape(3),
            np.asarray(payload["rotvec"], dtype=np.float64).reshape(3),
        ]))
        source = "position_m + rotvec"
    elif "position_m" in payload and "rpy_rad" in payload:
        pose7d = _rpy_pose_to_pose7d(payload["position_m"], payload["rpy_rad"])
        source = "position_m + rpy_rad xyz"
    else:
        raise ValueError(
            f"Could not find a supported pose in {pose_path}. "
            "Expected one of O_T_EE, pose7d, tcp_pose, position_m+quat_wxyz, "
            "position_m+quat_xyzw, position_m+rotvec, or position_m+rpy_rad."
        )

    return {
        "pose7d": pose7d,
        "path": pose_path,
        "episode_idx": None,
        "episode_count": None,
        "row_start": None,
        "row_end": None,
        "source": source,
        "manual": True,
        "timestamp_unix_s": payload.get("timestamp_unix_s"),
    }


def _interpolate_pose7d(start_pose, end_pose, alpha):
    start_pose = _normalize_pose7d(start_pose)
    end_pose = _normalize_pose7d(end_pose)
    alpha = np.asarray(alpha, dtype=np.float64)
    pos = start_pose[:3][None, :] * (1.0 - alpha[:, None]) + end_pose[:3][None, :] * alpha[:, None]
    rots = st.Rotation.from_quat([
        [start_pose[4], start_pose[5], start_pose[6], start_pose[3]],
        [end_pose[4], end_pose[5], end_pose[6], end_pose[3]],
    ])
    quat_xyzw = st.Slerp([0.0, 1.0], rots)(alpha).as_quat()
    quat_wxyz = np.concatenate([quat_xyzw[:, 3:4], quat_xyzw[:, :3]], axis=1)
    return np.concatenate([pos, quat_wxyz], axis=1)


def _load_dataset_start_pose(dataset_path, episode_idx):
    for candidate in _dataset_candidates(dataset_path):
        if not os.path.exists(candidate):
            continue
        root, store = _open_zarr(candidate)
        try:
            if "data" not in root or "meta" not in root or "episode_ends" not in root["meta"]:
                continue
            data = root["data"]
            episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
            episode_slice, start, end = _get_episode_slice(episode_ends, int(episode_idx))

            if "robot0_eef_pos" in data and "robot0_eef_rot_axis_angle" in data:
                pos = np.asarray(data["robot0_eef_pos"][episode_slice][0], dtype=np.float64)
                rotvec = np.asarray(data["robot0_eef_rot_axis_angle"][episode_slice][0], dtype=np.float64)
                quat_xyzw = st.Rotation.from_rotvec(rotvec).as_quat()
                quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
                pose7d = _normalize_pose7d(np.concatenate([pos, quat_wxyz]))
                source = "robot0_eef_pos + robot0_eef_rot_axis_angle"
            elif "left_robot_tcp_pose" in data:
                pose = np.asarray(data["left_robot_tcp_pose"][episode_slice][0], dtype=np.float64)
                if pose.shape[0] == 9:
                    pose7d = _pose9_to_pose7d(pose)
                    source = "left_robot_tcp_pose xyz+ortho6d"
                elif pose.shape[0] == 7:
                    pose7d = _normalize_pose7d(pose)
                    source = "left_robot_tcp_pose xyz+quat_wxyz"
                elif pose.shape[0] == 6:
                    quat_xyzw = st.Rotation.from_rotvec(pose[3:6]).as_quat()
                    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
                    pose7d = _normalize_pose7d(np.concatenate([pose[:3], quat_wxyz]))
                    source = "left_robot_tcp_pose xyz+rotvec"
                else:
                    raise ValueError(f"Unsupported left_robot_tcp_pose shape {pose.shape}")
            else:
                continue

            return {
                "pose7d": pose7d,
                "path": candidate,
                "episode_idx": int(episode_idx),
                "episode_count": len(episode_ends),
                "row_start": start,
                "row_end": end,
                "source": source,
            }
        finally:
            if store is not None:
                store.close()
    raise FileNotFoundError(f"Could not load reset start pose from dataset_path={dataset_path}")


def _load_dataset_start_image_stats(dataset_path, episode_idx, image_shape):
    out_h, out_w = int(image_shape[1]), int(image_shape[2])
    for candidate in _dataset_candidates(dataset_path):
        if not os.path.exists(candidate):
            continue
        root, store = _open_zarr(candidate)
        try:
            if "data" not in root or "meta" not in root or "episode_ends" not in root["meta"]:
                continue
            data = root["data"]
            image_key = "left_wrist_img" if "left_wrist_img" in data else "camera0_rgb"
            if image_key not in data:
                continue
            episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
            _, start, _ = _get_episode_slice(episode_ends, int(episode_idx))
            image = np.asarray(data[image_key][start], dtype=np.uint8)
            if image.shape[:2] != (out_h, out_w):
                interpolation = cv2.INTER_AREA
                if out_h > image.shape[0] or out_w > image.shape[1]:
                    interpolation = cv2.INTER_LINEAR
                image = cv2.resize(image, (out_w, out_h), interpolation=interpolation)
            image_float = image.astype(np.float32)
            return {
                "mean": image_float.mean(axis=(0, 1)),
                "std": image_float.std(axis=(0, 1)),
                "path": candidate,
                "image_key": image_key,
                "episode_idx": int(episode_idx),
                "row": int(start),
            }
        finally:
            if store is not None:
                store.close()
    raise FileNotFoundError(
        f"Could not load a start image from dataset_path={dataset_path}, episode={episode_idx}"
    )


def _infer_gripper_raw_range(
        dataset_path,
        open_quantile=0.99,
        range_mode="dataset_observed_range",
        calibration_scale=1.0,
        calibration_offset=0.0,
        gripper_stroke=None):
    if dataset_path is None:
        return None
    range_mode = str(range_mode)
    valid_range_modes = (
        "dataset_observed_range",
        "trajectory_zero_to_max",
        "calibrated_marker_trajectory",
        "calibrated_marker_width",
    )
    if range_mode not in valid_range_modes:
        raise ValueError(
            "gripper_raw_range_mode must be one of "
            f"{valid_range_modes}, got {range_mode!r}"
        )
    calibration_scale = float(calibration_scale)
    calibration_offset = float(calibration_offset)
    if range_mode in ("calibrated_marker_trajectory", "calibrated_marker_width") and (
            not np.isfinite(calibration_scale) or calibration_scale <= 0.0
            or not np.isfinite(calibration_offset)):
        raise ValueError(
            "calibrated marker modes require a finite positive "
            "gripper_raw_calibration_scale and finite gripper_raw_calibration_offset"
        )
    if gripper_stroke is not None:
        gripper_stroke = float(gripper_stroke)
        if not np.isfinite(gripper_stroke) or gripper_stroke <= 0.0:
            raise ValueError("gripper_stroke must be finite and positive")
    open_quantile = float(open_quantile)
    if not np.isfinite(open_quantile):
        open_quantile = 0.99
    open_quantile = float(np.clip(open_quantile, 0.5, 1.0))
    for candidate in _dataset_candidates(dataset_path):
        if not os.path.exists(candidate):
            continue
        root, store = _open_zarr(candidate)
        try:
            data = root["data"] if "data" in root else root
            if range_mode in (
                    "trajectory_zero_to_max",
                    "calibrated_marker_trajectory",
                    "calibrated_marker_width"):
                if "action" not in data:
                    continue
                values = np.asarray(data["action"][:, -1], dtype=np.float64).reshape(-1)
            else:
                value_arrays = []
                if "action" in data:
                    value_arrays.append(np.asarray(data["action"][:, -1], dtype=np.float64).reshape(-1))
                if "left_robot_gripper_width" in data:
                    value_arrays.append(
                        np.asarray(data["left_robot_gripper_width"][:], dtype=np.float64).reshape(-1)
                    )
                if not value_arrays:
                    continue
                values = np.concatenate(value_arrays, axis=0)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            raw_abs_max = float(values.max())
            if range_mode == "trajectory_zero_to_max":
                raw_min = 0.0
                raw_max = raw_abs_max
            elif range_mode == "calibrated_marker_trajectory":
                # The model is trained on marker distance, but physical closure is
                # defined by the calibrated gap reaching zero.
                raw_min = max(0.0, -calibration_offset / calibration_scale)
                raw_max = raw_abs_max
            elif range_mode == "calibrated_marker_width":
                raw_min = max(0.0, -calibration_offset / calibration_scale)
                raw_max = (
                    raw_abs_max
                    if gripper_stroke is None
                    else (gripper_stroke - calibration_offset) / calibration_scale
                )
            else:
                raw_min = float(values.min())
                raw_max = raw_abs_max if open_quantile >= 1.0 else float(np.quantile(values, open_quantile))
            if raw_max > raw_min:
                return raw_min, raw_max, raw_abs_max, open_quantile, candidate
        finally:
            if store is not None:
                store.close()
    return None


def _infer_magnet_used_sensor_count(dataset_path, tactile_key, tactile_dim):
    if dataset_path is None:
        return None
    tactile_dim = int(tactile_dim)
    max_groups = min(MAGNET_SENSOR_COUNT, max(0, tactile_dim // 3))
    if max_groups <= 0:
        return None
    for candidate in _dataset_candidates(dataset_path):
        if not os.path.exists(candidate):
            continue
        root, store = _open_zarr(candidate)
        try:
            data = root["data"] if "data" in root else root
            if tactile_key not in data:
                continue
            arr = data[tactile_key]
            if arr.ndim != 2 or arr.shape[1] < 3:
                continue
            n = min(int(arr.shape[0]), 4096)
            values = np.asarray(arr[:n, :max_groups * 3], dtype=np.float32)
            if values.size == 0:
                continue
            grouped = values.reshape(n, max_groups, 3)
            finite = np.isfinite(grouped)
            if not np.any(finite):
                continue
            magnitudes = np.nanmax(np.where(finite, np.abs(grouped), np.nan), axis=(0, 2))
            active = np.where(np.nan_to_num(magnitudes, nan=0.0) > 1e-6)[0]
            if active.size > 0:
                return int(active[-1] + 1), candidate
        finally:
            if store is not None:
                store.close()
    return None


def _parse_magnet_sensor_order(sensor_order, used_sensor_count):
    used_sensor_count = int(used_sensor_count)
    if sensor_order is None:
        return np.arange(used_sensor_count, dtype=np.int64)
    if isinstance(sensor_order, str):
        value = sensor_order.strip()
        if not value or value.lower() in ("identity", "none"):
            return np.arange(used_sensor_count, dtype=np.int64)
        value = value.strip("[]")
        sensor_order = [part.strip() for part in value.split(",")]

    try:
        one_based_order = [int(sensor_idx) for sensor_idx in sensor_order]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "magnet_sensor_order must be a one-based sensor permutation, "
            "for example [2, 1, 4, 3]"
        ) from exc

    expected = list(range(1, used_sensor_count + 1))
    if len(one_based_order) != used_sensor_count or sorted(one_based_order) != expected:
        raise ValueError(
            "magnet_sensor_order must contain each used sensor exactly once: "
            f"expected a permutation of {expected}, got {one_based_order}"
        )
    return np.asarray(one_based_order, dtype=np.int64) - 1


def _parse_magnet_zero_channels(channels, tactile_dim):
    tactile_dim = int(tactile_dim)
    if channels is None:
        return np.zeros((0,), dtype=np.int64)
    if isinstance(channels, str):
        value = channels.strip()
        if not value:
            return np.zeros((0,), dtype=np.int64)
        value = value.strip("[]")
        channels = [part.strip() for part in value.split(",")]

    try:
        one_based_channels = [int(channel) for channel in channels]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "magnet_zero_channels must be a comma-separated list of one-based "
            "tactile embedding channels, for example [1, 2, 6]"
        ) from exc

    zero_based_channels = [channel - 1 for channel in one_based_channels]
    invalid = [
        channel
        for channel, zero_based in zip(one_based_channels, zero_based_channels)
        if zero_based < 0 or zero_based >= tactile_dim
    ]
    if invalid:
        raise ValueError(
            "magnet_zero_channels entries must be in "
            f"[1, {tactile_dim}], got {invalid}"
        )
    return np.asarray(sorted(set(zero_based_channels)), dtype=np.int64)


def _remap_magnet_sensors(magnet_xyz, sensor_order):
    magnet_xyz = np.asarray(magnet_xyz, dtype=np.float32)
    if magnet_xyz.ndim != 3 or magnet_xyz.shape[-1] != 3:
        raise ValueError(f"Expected magnetic frame [S, N, 3], got {magnet_xyz.shape}")
    sensor_order = np.asarray(sensor_order, dtype=np.int64).reshape(-1)
    if magnet_xyz.shape[1] != sensor_order.size:
        raise ValueError(
            "Magnet sensor order length does not match magnetic frame: "
            f"{sensor_order.size} != {magnet_xyz.shape[1]}"
        )
    return magnet_xyz[:, sensor_order, :].copy()


class _MagnetometerReader:
    """Background serial reader matching gello_software/experiments/collect_fr3_data.py."""

    def __init__(
            self,
            port: str,
            baudrate: int,
            samples_per_frame: int = MAGNET_SAMPLES_PER_FRAME,
            used_sensor_count: int = MAGNET_USED_SENSOR_COUNT,
            buffer_size: int = 1024,
            idle_sleep: float = 0.0005,
            subtract_baseline: bool = True):
        self.port = port
        self.baudrate = int(baudrate)
        self.samples_per_frame = int(samples_per_frame)
        self.used_sensor_count = int(used_sensor_count)
        if self.used_sensor_count < 1 or self.used_sensor_count > MAGNET_SENSOR_COUNT:
            raise ValueError(
                f"used_sensor_count must be in [1, {MAGNET_SENSOR_COUNT}], "
                f"got {self.used_sensor_count}"
            )
        self.buffer_size = int(buffer_size)
        self.idle_sleep = float(idle_sleep)
        self.subtract_baseline = bool(subtract_baseline)
        self._ser = None
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._buffer = bytearray()
        self._samples = deque(maxlen=self.buffer_size)
        self._sample_count = 0
        self._baseline_xyz = None

    def start(self):
        import serial

        self._ser = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=0)
        self._ser.reset_input_buffer()
        time.sleep(0.2)
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._ser is not None:
            self._ser.close()
            self._ser = None

    def reset_baseline(self):
        with self._lock:
            self._baseline_xyz = None
            self._samples.clear()

    @property
    def sample_count(self):
        with self._lock:
            return self._sample_count

    def _read_loop(self):
        while self._running:
            try:
                waiting = self._ser.in_waiting
                chunk = self._ser.read(waiting if waiting > 0 else 1)
                if not chunk:
                    time.sleep(self.idle_sleep)
                    continue

                self._buffer.extend(chunk)
                while b"\n" in self._buffer:
                    line, _, remainder = self._buffer.partition(b"\n")
                    self._buffer = bytearray(remainder)
                    self._handle_line(line.decode("utf-8", errors="ignore").strip())
            except Exception as exc:
                logger.warning(f"Magnetometer read error: {exc}")
                self._running = False

    def _handle_line(self, line: str):
        values = self._parse_sensor_values(line)
        if len(values) != MAGNET_SENSOR_COUNT * MAGNET_VALUES_PER_SENSOR:
            return

        xyz_values = []
        for sensor_idx in range(self.used_sensor_count):
            start = sensor_idx * MAGNET_VALUES_PER_SENSOR
            _, x, y, z = values[start:start + MAGNET_VALUES_PER_SENSOR]
            xyz_values.append([x, y, z])
        xyz = np.asarray(xyz_values, dtype=np.float32)
        with self._lock:
            if self.subtract_baseline:
                if self._baseline_xyz is None:
                    self._baseline_xyz = xyz.copy()
                    logger.info("Magnetometer reader baseline initialized from first valid sample.")
                xyz = xyz - self._baseline_xyz
            sample = {
                "timestamp_ns": time.time_ns(),
                "xyz": xyz.astype(np.float32),
            }
            self._samples.append(sample)
            self._sample_count += 1

    @staticmethod
    def _parse_sensor_values(line: str):
        parts = line.split()
        if len(parts) == MAGNET_SENSOR_COUNT * MAGNET_VALUES_PER_SENSOR:
            try:
                return [float(part) for part in parts]
            except ValueError:
                pass

        matches = MAGNET_FLOAT_PATTERN.findall(line)
        if len(matches) != MAGNET_SENSOR_COUNT * MAGNET_VALUES_PER_SENSOR:
            return []
        return [float(match) for match in matches]

    def get_recent_samples(self):
        with self._lock:
            samples = list(self._samples)[-self.samples_per_frame:]

        xyz = np.full(
            (self.samples_per_frame, self.used_sensor_count, 3),
            np.nan,
            dtype=np.float32,
        )
        timestamps = np.zeros(self.samples_per_frame, dtype=np.int64)
        start = self.samples_per_frame - len(samples)
        for idx, sample in enumerate(samples, start=start):
            xyz[idx] = sample["xyz"]
            timestamps[idx] = sample["timestamp_ns"]

        return {
            "magnet_xyz": xyz,
            "magnet_timestamp_ns": timestamps,
            "magnet_sample_count": np.array([len(samples)], dtype=np.int32),
        }


def _magnetic_time_mean(magnet, sample_count=None):
    magnet = np.nan_to_num(np.asarray(magnet, dtype=np.float32), nan=0.0)
    if magnet.ndim != 4 or magnet.shape[-1] != 3:
        raise ValueError(f"Expected magnetic batch [B, S, N, 3], got {magnet.shape}")

    batch_size, max_samples = magnet.shape[:2]
    if sample_count is None:
        return magnet.mean(axis=1)

    counts = np.asarray(sample_count, dtype=np.int64).reshape(batch_size)
    counts = np.clip(counts, 0, max_samples)
    sample_indices = np.arange(max_samples, dtype=np.int64)[None, :]
    # get_recent_samples right-aligns valid samples and leaves leading NaNs
    # when fewer than max_samples are available.
    mask = sample_indices >= (max_samples - counts[:, None])
    weighted = magnet * mask[:, :, None, None].astype(np.float32)
    denom = np.maximum(counts, 1).astype(np.float32)[:, None, None]
    mean = weighted.sum(axis=1) / denom
    mean[counts == 0] = 0
    return mean.astype(np.float32)


def _filter_abnormal_magnet_readings_live(
        magnet_xyz,
        last_valid_xyz=None,
        threshold: float = MAGNET_ABNORMAL_ABS_THRESHOLD):
    cleaned = np.asarray(magnet_xyz, dtype=np.float32).copy()
    if cleaned.ndim != 3 or cleaned.shape[-1] != 3:
        raise ValueError(f"Expected magnetic frame [S, N, 3], got {cleaned.shape}")

    if last_valid_xyz is None or np.shape(last_valid_xyz) != cleaned.shape[1:]:
        last_valid = np.full(cleaned.shape[1:], np.nan, dtype=np.float32)
    else:
        last_valid = np.asarray(last_valid_xyz, dtype=np.float32).copy()

    replaced_count = 0
    threshold = float(threshold)
    for sample_idx in range(cleaned.shape[0]):
        values = cleaned[sample_idx]
        finite_mask = np.isfinite(values)
        abnormal_mask = finite_mask & (np.abs(values) > threshold)
        normal_mask = finite_mask & ~abnormal_mask

        replaceable_mask = abnormal_mask & np.isfinite(last_valid)
        if np.any(replaceable_mask):
            values[replaceable_mask] = last_valid[replaceable_mask]
            replaced_count += int(np.count_nonzero(replaceable_mask))

        missing_history_mask = abnormal_mask & ~np.isfinite(last_valid)
        if np.any(missing_history_mask):
            values[missing_history_mask] = 0.0
            replaced_count += int(np.count_nonzero(missing_history_mask))

        if np.any(normal_mask):
            last_valid[normal_mask] = values[normal_mask]

    return cleaned, last_valid, replaced_count


def _magnet_to_tactile_embedding(magnet, sample_count=None, output_dim=15):
    if output_dim <= 0:
        raise ValueError("magnet_tactile_dim must be positive")
    mean_by_sensor = _magnetic_time_mean(magnet, sample_count=sample_count)
    flat = mean_by_sensor.reshape(mean_by_sensor.shape[0], -1)
    if flat.shape[1] == output_dim:
        return flat.astype(np.float32)
    out = np.zeros((flat.shape[0], output_dim), dtype=np.float32)
    copy_dim = min(flat.shape[1], output_dim)
    out[:, :copy_dim] = flat[:, :copy_dim]
    return out


def _camera_candidates(source, backend="auto"):
    if source != "auto":
        try:
            return [int(source)]
        except ValueError:
            return [source]
    candidates = []
    backend = str(backend).lower()
    if backend in ("zed", "zed_v4l"):
        for pattern in (
            "/dev/v4l/by-id/*ZED*video-index0",
            "/dev/v4l/by-id/*ZED*video-index*",
            "/dev/v4l/by-path/*video-index0",
            "/dev/video*",
        ):
            candidates.extend(sorted(glob.glob(pattern)))
    else:
        for pattern in (
            "/dev/v4l/by-path/*:1.3-video-index0",
            "/dev/v4l/by-path/*:1.3-video-index1",
            "/dev/v4l/by-id/*RealSense*video-index2",
            "/dev/v4l/by-id/*RealSense*video-index0",
            "/dev/v4l/by-path/*:1.0-video-index2",
            "/dev/v4l/by-path/*:1.0-video-index0",
            "/dev/v4l/by-path/*video-index2",
            "/dev/v4l/by-path/*video-index0",
            "/dev/video*",
        ):
            candidates.extend(sorted(glob.glob(pattern)))
    result = []
    seen = set()
    for candidate in candidates:
        if str(candidate) not in seen:
            seen.add(str(candidate))
            result.append(candidate)
    return result or [0]


def _bgr_color_score(frame):
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        return 0.0
    frame_f = frame.astype(np.float32)
    return float(max(
        np.mean(np.abs(frame_f[..., 0] - frame_f[..., 1])),
        np.mean(np.abs(frame_f[..., 1] - frame_f[..., 2])),
        np.mean(np.abs(frame_f[..., 0] - frame_f[..., 2])),
    ))


@dataclass
class _IPhoneNALAssembly:
    total_fragments: int
    fragments: Dict[int, bytes] = field(default_factory=dict)

    def is_complete(self):
        return len(self.fragments) == self.total_fragments


@dataclass
class _IPhoneFrameAssembly:
    frame_id: int
    capture_timestamp: float
    nalu_count: int
    is_keyframe: bool
    created_at: float
    last_update_at: float
    nalus: Dict[int, _IPhoneNALAssembly] = field(default_factory=dict)

    def is_complete(self):
        if len(self.nalus) != self.nalu_count:
            return False
        return all(nalu.is_complete() for nalu in self.nalus.values())

    def to_annexb(self):
        chunks = []
        for nalu_index in range(self.nalu_count):
            assembly = self.nalus.get(nalu_index)
            if assembly is None or not assembly.is_complete():
                raise ValueError(f"iPhone frame {self.frame_id} is incomplete")
            payload = b"".join(
                assembly.fragments[index] for index in range(assembly.total_fragments)
            )
            chunks.append(b"\x00\x00\x00\x01" + payload)
        return b"".join(chunks)


class _IPhoneUDPCamera:
    """Receive ARPoseStreamer APV1/APV2 UDP H.264 video and expose BGR frames."""

    def __init__(
            self,
            source="auto",
            bind_host="0.0.0.0",
            video_port=5560,
            combined_port=5558,
            phone_ip="",
            registration_port=5559,
            startup_timeout=5.0,
            read_timeout=1.0,
            hello_interval=2.0,
            require_color=True,
            color_threshold=1.5):
        self.source = str(source)
        self.bind_host = str(bind_host)
        self.video_port = int(video_port)
        self.combined_port = int(combined_port)
        self.phone_ip = str(phone_ip or "")
        if not self.phone_ip and self.source not in ("", "auto", "none", "None"):
            self.phone_ip = self.source
        self.registration_port = int(registration_port)
        self.startup_timeout = float(startup_timeout)
        self.read_timeout = float(read_timeout)
        self.hello_interval = float(hello_interval)
        self.require_color = bool(require_color)
        self.color_threshold = float(color_threshold)
        self.video_socket = None
        self.hello_socket = None
        self.thread = None
        self.stop_event = threading.Event()
        self.condition = threading.Condition()
        self.latest_frame = None
        self.latest_frame_id = None
        self.latest_frame_time = None
        self.receiver_error = None
        self.decoder = None
        self.frames = {}
        self.waiting_for_keyframe = True
        self.decoded_frames = 0
        self.dropped_frames = 0
        self.decode_errors = 0
        self.received_packets = 0
        self.unsupported_packets = 0

    @staticmethod
    def _create_decoder():
        if av is None:
            return None
        return av.CodecContext.create("h264", "r")

    def open(self):
        if av is None:
            raise RuntimeError(
                "CAMERA_BACKEND=iphone requires PyAV to decode the ARPoseStreamer "
                f"H.264 UDP stream. PyAV import failed: {AV_IMPORT_ERROR}. "
                "Install it in the eval environment, e.g. `conda install -n umi -c conda-forge av`."
            )

        self.close()
        self.stop_event.clear()
        self.receiver_error = None
        self.latest_frame = None
        self.latest_frame_id = None
        self.frames = {}
        self.waiting_for_keyframe = True
        self.received_packets = 0
        self.unsupported_packets = 0
        self.decoded_frames = 0
        self.dropped_frames = 0
        self.decode_errors = 0
        self.decoder = self._create_decoder()

        self.video_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.video_socket.bind((self.bind_host, self.video_port))
        except OSError as exc:
            self.video_socket.close()
            self.video_socket = None
            raise RuntimeError(
                "Could not bind the iPhone video UDP port "
                f"{self.bind_host}:{self.video_port}. Another iPhone receiver "
                "such as experiment_replay_ui.py may already be running; stop it "
                "before starting eval."
            ) from exc
        self.video_socket.setblocking(False)

        if self.phone_ip:
            self.hello_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.hello_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                self.hello_socket.bind((self.bind_host, self.combined_port))
            except OSError as exc:
                logger.warning(
                    "Could not bind iPhone combined/registration socket on "
                    f"{self.bind_host}:{self.combined_port}: {exc}. "
                    "Falling back to an ephemeral UDP source port for PC_HELLO."
                )
                self.hello_socket.bind((self.bind_host, 0))
            logger.info(
                "Registering iPhone stream with PC_HELLO: "
                f"phone={self.phone_ip}:{self.registration_port}, "
                f"combined_port={self.combined_port}, video_port={self.video_port}"
            )
        else:
            logger.warning(
                "iPhone phone IP is empty; eval will only passively listen on "
                f"{self.bind_host}:{self.video_port} and cannot send PC_HELLO."
            )

        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()
        self._send_hello()

        deadline = time.monotonic() + self.startup_timeout
        with self.condition:
            while (
                    self.latest_frame is None
                    and self.receiver_error is None
                    and time.monotonic() < deadline):
                self.condition.wait(timeout=0.05)
            if self.receiver_error is not None:
                error = self.receiver_error
                self.close()
                raise RuntimeError(f"iPhone camera receiver failed: {error}")
            if self.latest_frame is None:
                self.close()
                hint = (
                    f" Set CAMERA_SOURCE or CAMERA_IPHONE_PHONE_IP to the iPhone IP "
                    f"so eval can send PC_HELLO to UDP {self.registration_port}."
                    if not self.phone_ip else ""
                )
                stats = (
                    f" received_packets={self.received_packets}, "
                    f"unsupported_packets={self.unsupported_packets}, "
                    f"decode_errors={self.decode_errors}, "
                    f"dropped_frames={self.dropped_frames}"
                )
                raise RuntimeError(
                    "Timed out waiting for iPhone video frame on "
                    f"{self.bind_host}:{self.video_port}.{stats}.{hint}"
                )
            first_frame = self.latest_frame.copy()

        color_score = _bgr_color_score(first_frame)
        if self.require_color and color_score < self.color_threshold:
            self.close()
            raise RuntimeError(
                f"iPhone frame color_score={color_score:.3f} is below "
                f"threshold={self.color_threshold:.3f}"
            )
        logger.info(
            "iPhone UDP camera opened: "
            f"bind={self.bind_host}:{self.video_port}, "
            f"phone_ip={self.phone_ip or 'not registered'}, "
            f"combined_port={self.combined_port}, "
            f"registration_port={self.registration_port}, "
            f"shape={first_frame.shape}, color_score={color_score:.3f}"
        )

    def read(self):
        deadline = time.monotonic() + self.read_timeout
        with self.condition:
            while self.latest_frame is None and self.receiver_error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self.condition.wait(timeout=min(0.05, remaining))
            if self.latest_frame is None:
                if self.receiver_error is not None:
                    raise RuntimeError(f"iPhone camera receiver failed: {self.receiver_error}")
                raise RuntimeError(
                    f"Timed out reading iPhone frame after {self.read_timeout:.3f}s"
                )
            return self.latest_frame.copy()

    def close(self):
        self.stop_event.set()
        for sock in (self.video_socket, self.hello_socket):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        self.video_socket = None
        self.hello_socket = None
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.thread = None
        self.decoder = None

    def _receive_loop(self):
        next_hello = 0.0
        try:
            while not self.stop_event.is_set():
                now = time.monotonic()
                if self.phone_ip and now >= next_hello:
                    self._send_hello()
                    next_hello = now + self.hello_interval

                try:
                    readable, _, _ = select([self.video_socket], [], [], 0.05)
                except (OSError, TypeError, ValueError):
                    break
                for current_socket in readable:
                    try:
                        packet, _address = current_socket.recvfrom(65535)
                    except OSError:
                        continue
                    self._handle_video_packet(packet)
                self._prune_stale_frames(time.monotonic())
        except Exception as exc:
            with self.condition:
                self.receiver_error = exc
                self.condition.notify_all()

    def _send_hello(self):
        if not self.phone_ip or self.hello_socket is None:
            return
        hello = f"PC_HELLO,1,{self.combined_port},{self.video_port}\n".encode("ascii")
        try:
            self.hello_socket.sendto(hello, (self.phone_ip, self.registration_port))
        except OSError as exc:
            logger.warning(f"Failed to send iPhone PC_HELLO: {exc}")

    def _parse_video_packet(self, packet):
        if len(packet) < 6:
            return None
        magic = packet[:4]
        version = packet[4]
        if magic == IPHONE_VIDEO_MAGIC_V1 and version == IPHONE_VIDEO_VERSION:
            header = IPHONE_VIDEO_PACKET_HEADER_V1
        elif magic == IPHONE_VIDEO_MAGIC_V2 and version == IPHONE_VIDEO_VERSION_V2:
            header = IPHONE_VIDEO_PACKET_HEADER_V2
        else:
            self.unsupported_packets += 1
            if self.unsupported_packets <= 3:
                logger.warning(
                    "Ignoring unsupported iPhone video packet: "
                    f"magic={magic!r}, version={version}, bytes={len(packet)}"
                )
            return None
        if len(packet) < header.size:
            return None
        try:
            values = header.unpack_from(packet)
        except struct.error:
            return None

        if header is IPHONE_VIDEO_PACKET_HEADER_V1:
            (
                magic,
                version,
                flags,
                _reserved,
                frame_id,
                capture_timestamp,
                nalu_index,
                nalu_count,
                fragment_index,
                fragment_count,
            ) = values
        else:
            (
                magic,
                version,
                flags,
                _reserved,
                frame_id,
                capture_timestamp,
                nalu_index,
                nalu_count,
                fragment_index,
                fragment_count,
                _fx,
                _fy,
                _cx,
                _cy,
                _image_width,
                _image_height,
            ) = values

        if fragment_count <= 0 or nalu_count <= 0:
            return None
        if not (0 <= nalu_index < nalu_count) or not (0 <= fragment_index < fragment_count):
            return None
        return (
            flags,
            frame_id,
            capture_timestamp,
            nalu_index,
            nalu_count,
            fragment_index,
            fragment_count,
            packet[header.size:],
        )

    def _handle_video_packet(self, packet):
        self.received_packets += 1
        parsed = self._parse_video_packet(packet)
        if parsed is None:
            return
        (
            flags,
            frame_id,
            capture_timestamp,
            nalu_index,
            nalu_count,
            fragment_index,
            fragment_count,
            payload,
        ) = parsed
        if self.latest_frame_id is not None and frame_id <= self.latest_frame_id:
            return

        now = time.monotonic()
        frame = self.frames.get(frame_id)
        if frame is None:
            frame = _IPhoneFrameAssembly(
                frame_id=frame_id,
                capture_timestamp=capture_timestamp,
                nalu_count=nalu_count,
                is_keyframe=bool(flags & 0x01),
                created_at=now,
                last_update_at=now,
            )
            self.frames[frame_id] = frame
        else:
            frame.last_update_at = now

        nalu = frame.nalus.get(nalu_index)
        if nalu is None:
            nalu = _IPhoneNALAssembly(total_fragments=fragment_count)
            frame.nalus[nalu_index] = nalu
        elif nalu.total_fragments != fragment_count:
            nalu.total_fragments = max(nalu.total_fragments, fragment_count)
        if fragment_index not in nalu.fragments:
            nalu.fragments[fragment_index] = payload

        if frame.is_complete():
            self.frames.pop(frame_id, None)
            self._decode_frame(frame)
        self._trim_inflight_frames()

    def _decode_frame(self, frame):
        if self.decoder is None:
            return
        if self.waiting_for_keyframe and not frame.is_keyframe:
            return
        try:
            annexb = frame.to_annexb()
            if frame.is_keyframe:
                self.decoder = self._create_decoder()
                self.waiting_for_keyframe = False
            decoded_frames = self._decode_annexb_packet(annexb)
            decoded_any = False
            for decoded_frame in decoded_frames:
                decoded_any = True
                rgb = decoded_frame.to_ndarray(format="rgb24")
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                with self.condition:
                    self.latest_frame = bgr
                    self.latest_frame_id = frame.frame_id
                    self.latest_frame_time = time.time()
                    self.decoded_frames += 1
                    self.condition.notify_all()
            if not decoded_any and frame.is_keyframe:
                self.waiting_for_keyframe = True
            else:
                self.waiting_for_keyframe = False
        except Exception as exc:
            self.decode_errors += 1
            self.waiting_for_keyframe = True
            self.decoder = self._create_decoder()
            logger.warning(f"iPhone H.264 decode error on frame {frame.frame_id}: {exc}")

    def _decode_annexb_packet(self, annexb):
        packet = av.Packet(annexb)
        decoded_frames = self.decoder.decode(packet)
        if decoded_frames:
            return decoded_frames
        fallback_frames = []
        for parsed_packet in self.decoder.parse(annexb):
            fallback_frames.extend(self.decoder.decode(parsed_packet))
        return fallback_frames

    def _trim_inflight_frames(self):
        if len(self.frames) <= IPHONE_MAX_INFLIGHT_FRAMES:
            return
        stale = sorted(
            self.frames.items(),
            key=lambda item: item[1].created_at,
        )[:-IPHONE_MAX_INFLIGHT_FRAMES]
        for frame_id, _frame in stale:
            self.frames.pop(frame_id, None)
            self.dropped_frames += 1

    def _prune_stale_frames(self, now):
        stale_ids = [
            frame_id
            for frame_id, frame in self.frames.items()
            if now - frame.last_update_at >= IPHONE_FRAME_STALE_SECONDS
        ]
        for frame_id in stale_ids:
            self.frames.pop(frame_id, None)
            self.dropped_frames += 1


def _import_gello_zed_camera():
    try:
        from gello.cameras.zed_camera import ZEDCamera
        return ZEDCamera
    except ImportError as first_error:
        candidate_paths = []
        env_path = os.environ.get("GELLO_SOFTWARE_PATH")
        if env_path:
            candidate_paths.append(os.path.expanduser(env_path))
        candidate_paths.append(
            os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../../../../gello_software")
            )
        )
        for path in candidate_paths:
            if path and os.path.isdir(path) and path not in sys.path:
                sys.path.insert(0, path)
                try:
                    from gello.cameras.zed_camera import ZEDCamera
                    return ZEDCamera
                except ImportError:
                    continue
        raise RuntimeError(
            "Could not import gello.cameras.zed_camera for CAMERA_BACKEND=zed. "
            "Install gello_software, add it to PYTHONPATH, or set GELLO_SOFTWARE_PATH."
        ) from first_error


class _ZEDSDKCamera:
    """ZED SDK camera wrapper matching collect_fr3_data.py VIEW.LEFT output."""

    def __init__(
            self,
            source="auto",
            fps=30.0,
            require_color=True,
            color_threshold=1.5,
            view="left",
            resolution="HD720",
            depth_mode="NEURAL"):
        self.source = source
        self.fps = fps
        self.require_color = require_color
        self.color_threshold = color_threshold
        self.view = str(view).lower()
        self.resolution = str(resolution)
        self.depth_mode = str(depth_mode)
        self.camera = None
        self.device_id = self._device_id_from_source(source)
        if self.view != "left":
            raise ValueError(
                "CAMERA_BACKEND=zed uses the ZED SDK path from collect_fr3_data.py, "
                "which currently reads only VIEW.LEFT. Use CAMERA_ZED_VIEW=left."
            )

    @staticmethod
    def _device_id_from_source(source):
        if source is None:
            return None
        source = str(source)
        if source in ("", "auto"):
            return None
        if source.startswith("/dev/"):
            logger.warning(
                "Ignoring V4L camera_source for ZED SDK backend: "
                f"{source}. Use CAMERA_SOURCE=auto or a ZED serial number."
            )
            return None
        return source

    def open(self):
        ZEDCamera = _import_gello_zed_camera()
        try:
            self.camera = ZEDCamera(
                device_id=self.device_id,
                flip=False,
                resolution=self.resolution,
                depth_mode=self.depth_mode,
                fps=int(round(float(self.fps))),
            )
            rgb, _, _ = self.camera.read_with_timestamp()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            color_score = _bgr_color_score(bgr)
            if self.require_color and color_score < self.color_threshold:
                raise RuntimeError(
                    f"ZED SDK frame color_score={color_score:.3f} is below "
                    f"threshold={self.color_threshold:.3f}"
                )
            logger.info(
                "ZED SDK camera opened: "
                f"device_id={self.device_id or 'default'}, "
                f"view={self.view}, resolution={self.resolution}, "
                f"fps={self.fps}, shape={rgb.shape}, color_score={color_score:.3f}"
            )
        except ModuleNotFoundError as exc:
            if exc.name == "pyzed":
                raise RuntimeError(
                    "CAMERA_BACKEND=zed now uses the ZED SDK, but the current Python "
                    "environment does not have the ZED Python API package 'pyzed'. "
                    "Install pyzed into the same environment used by eval.sh "
                    "(usually conda env 'umi'), for example: "
                    "conda run -n umi python /usr/local/zed/get_python_api.py"
                ) from exc
            self.close()
            raise
        except Exception:
            self.close()
            raise

    def read(self):
        if self.camera is None:
            raise RuntimeError("ZED SDK camera is not open")
        rgb, _, _ = self.camera.read_with_timestamp()
        if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise RuntimeError(f"Unexpected ZED SDK RGB frame shape: {None if rgb is None else rgb.shape}")
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def close(self):
        if self.camera is not None:
            self.camera.close()
            self.camera = None


class _OpenCVCamera:
    def __init__(self, source="auto", width=1280, height=720, fps=30.0,
            require_color=True, color_threshold=1.5, backend="auto"):
        self.candidates = _camera_candidates(source, backend=backend)
        self.width = width
        self.height = height
        self.fps = fps
        self.require_color = require_color
        self.color_threshold = color_threshold
        self.cap = None
        self.source = None

    def open(self):
        errors = []
        for source in self.candidates:
            cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.height))
            cap.set(cv2.CAP_PROP_FPS, float(self.fps))
            if not cap.isOpened():
                errors.append(f"{source}: open failed")
                cap.release()
                continue
            last_shape = None
            last_score = None
            for _ in range(10):
                ok, frame = cap.read()
                if ok and frame is not None:
                    last_shape = frame.shape
                    last_score = _bgr_color_score(frame)
                    if (
                        frame.ndim == 3
                        and frame.shape[2] == 3
                        and (not self.require_color or last_score >= self.color_threshold)
                    ):
                        self.cap = cap
                        self.source = source
                        logger.info(
                            f"Camera opened: {source}, shape={frame.shape}, "
                            f"color_score={last_score:.3f}"
                        )
                        return
                time.sleep(0.05)
            cap.release()
            errors.append(f"{source}: shape={last_shape}, color_score={last_score}")
        raise RuntimeError("Could not open RGB camera. " + "; ".join(errors))

    def read(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame from camera source: {self.source}")
        return frame

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class FrankaPolymetisEnv:
    """RealRunner-compatible environment for a single FR3 via the local HTTP server."""

    uses_ros_executor = False

    def __init__(
            self,
            robot_server_ip: str,
            robot_server_port: int,
            transforms,
            data_processing_params=None,
            max_fps: int = 12,
            camera_source: str = "auto",
            camera_width: int = 1280,
            camera_height: int = 720,
            camera_fps: float = 30.0,
            require_color_camera: bool = True,
            camera_color_threshold: float = 1.5,
            dataset_path: Optional[str] = None,
            image_shape=(3, 240, 320),
            gripper_stroke: float = 0.080,
            gripper_raw_min: Optional[float] = None,
            gripper_raw_max: Optional[float] = None,
            gripper_raw_open_quantile: float = 0.99,
            gripper_raw_range_mode: str = "dataset_observed_range",
            gripper_raw_calibration_scale: float = 1.0,
            gripper_raw_calibration_offset: float = 0.0,
            gripper_command_width_offset: float = 0.0,
            fixed_gripper_width_mm: Optional[float] = None,
            max_gripper_width: Optional[float] = None,
            min_gripper_width: Optional[float] = None,
            grasp_force: float = 20.0,
            gripper_velocity: float = 0.08,
            gripper_control_width_precision: float = 0.001,
            gripper_width_threshold: Optional[float] = None,
            gripper_width_threshold_m: float = 0.046,
            enable_gripper_width_clipping: bool = True,
            use_force_control_for_gripper: bool = False,
            ignore_gripper_commands: bool = False,
            ignore_policy_tcp_commands: bool = False,
            ignore_policy_gripper_commands: bool = False,
            gripper_action_mode: str = "continuous",
            gripper_binary_threshold: float = 0.5,
            gripper_binary_hysteresis: float = 0.0,
            gripper_binary_open_threshold: Optional[float] = None,
            gripper_obs_mode: str = "auto",
            gripper_obs_raw_offset: float = 0.0,
            enable_exp_recording: bool = False,
            output_dir: Optional[str] = None,
            move_to_start_on_reset: bool = True,
            move_to_start_pose_path: Optional[str] = None,
            reset_episode: int = 0,
            move_to_start_duration: float = 5.0,
            move_to_start_frequency: float = 30.0,
            move_to_start_settle: bool = True,
            move_to_start_settle_timeout: float = 3.0,
            move_to_start_settle_frequency: float = 20.0,
            move_to_start_pos_tolerance: float = 0.002,
            move_to_start_rot_tolerance_deg: float = 3.0,
            move_to_start_strict: bool = True,
            save_processed_image: bool = False,
            processed_image_output_dir: Optional[str] = None,
            enable_policy_recording: bool = False,
            policy_recording_output_dir: Optional[str] = None,
            policy_recording_fps: Optional[float] = None,
            policy_recording_image_width: Optional[int] = None,
            policy_recording_image_height: Optional[int] = None,
            policy_recording_plot_width: int = 360,
            policy_recording_plot_window_sec: float = 10.0,
            policy_recording_codec: str = "mp4v",
            tcp_target_duration: Optional[float] = None,
            tcp_move_timeout: float = 0.001,
            http_timeout: float = 1.0,
            gripper_http_timeout: float = 5.0,
            enable_magnet: bool = False,
            magnet_required: bool = True,
            magnet_port: str = "/dev/ttyACM0",
            magnet_baudrate: int = 115200,
            magnet_samples_per_frame: int = MAGNET_SAMPLES_PER_FRAME,
            magnet_warmup_timeout: float = 2.0,
            magnet_tactile_key: str = "left_gripper1_marker_offset_emb",
            magnet_tactile_dim: int = 15,
            magnet_used_sensor_count: Optional[int] = None,
            magnet_sensor_order=None,
            magnet_zero_channels=None,
            magnet2_port: Optional[str] = None,
            magnet2_required: Optional[bool] = None,
            magnet2_tactile_key: str = "left_gripper2_marker_offset_emb",
            magnet2_tactile_dim: int = 15,
            magnet2_used_sensor_count: Optional[int] = None,
            magnet2_sensor_order=None,
            magnet2_zero_channels=None,
            magnet_normalize_to_first_frame: bool = False,
            magnet_rezero_after_policy_start_sec: Optional[float] = None,
            magnet_reader_subtract_baseline: bool = True,
            magnet_filter_abnormal_readings: bool = True,
            magnet_abnormal_abs_threshold: float = MAGNET_ABNORMAL_ABS_THRESHOLD,
            camera_backend: str = "opencv",
            camera_preprocess_mode: str = "square_crop",
            camera_square_crop_bottom_rows: int = 0,
            camera_color_match_dataset_start: bool = False,
            camera_color_match_dataset_episode: Optional[int] = None,
            camera_color_match_timeout: float = 2.0,
            camera_zed_view: str = "left",
            camera_zed_resolution: str = "HD720",
            camera_zed_depth_mode: str = "NEURAL",
            camera_iphone_bind_host: str = "0.0.0.0",
            camera_iphone_video_port: int = 5560,
            camera_iphone_combined_port: int = 5562,
            camera_iphone_phone_ip: str = "",
            camera_iphone_registration_port: int = 5559,
            camera_iphone_startup_timeout: float = 5.0,
            camera_iphone_read_timeout: float = 1.0,
            camera_iphone_hello_interval: float = 2.0,
            camera_flip: bool = False,
            **kwargs):
        self.robot_server_ip = robot_server_ip
        self.robot_server_port = robot_server_port
        self.base_url = f"http://{robot_server_ip}:{robot_server_port}"
        self.transforms = transforms
        self.max_fps = max_fps
        self.image_shape = tuple(image_shape)
        self.gripper_stroke = float(gripper_stroke)
        self.gripper_command_width_offset = float(gripper_command_width_offset)
        if not np.isfinite(self.gripper_command_width_offset):
            raise ValueError("gripper_command_width_offset must be finite")
        self.fixed_gripper_width_m = (
            None
            if fixed_gripper_width_mm is None
            else float(fixed_gripper_width_mm) / 1000.0
        )
        if self.fixed_gripper_width_m is not None and (
                not np.isfinite(self.fixed_gripper_width_m)
                or self.fixed_gripper_width_m < 0.0
                or self.fixed_gripper_width_m > self.gripper_stroke):
            raise ValueError(
                "fixed_gripper_width_mm must be within the physical gripper stroke: "
                f"0..{self.gripper_stroke * 1000.0:.3f} mm, "
                f"got {float(fixed_gripper_width_mm):.3f} mm"
            )
        self.last_fixed_gripper_width_command = None
        self.grasp_force = float(grasp_force)
        self.gripper_velocity = float(gripper_velocity)
        self.gripper_control_width_precision = float(gripper_control_width_precision)
        self.gripper_width_threshold_config = gripper_width_threshold
        self.gripper_width_threshold_m = float(gripper_width_threshold_m)
        self.enable_gripper_width_clipping = bool(enable_gripper_width_clipping)
        self.use_force_control_for_gripper = bool(use_force_control_for_gripper)
        self.ignore_gripper_commands = _as_bool(ignore_gripper_commands)
        self.ignore_policy_tcp_commands = _as_bool(ignore_policy_tcp_commands)
        self.ignore_policy_gripper_commands = _as_bool(ignore_policy_gripper_commands)
        self.gripper_action_mode = str(gripper_action_mode)
        self.gripper_binary_threshold = float(gripper_binary_threshold)
        self.gripper_binary_hysteresis = max(0.0, float(gripper_binary_hysteresis))
        if gripper_binary_open_threshold is None:
            self.gripper_binary_open_threshold = (
                self.gripper_binary_threshold + self.gripper_binary_hysteresis
            )
        else:
            self.gripper_binary_open_threshold = float(gripper_binary_open_threshold)
        if self.gripper_binary_open_threshold < self.gripper_binary_threshold:
            raise ValueError(
                "gripper_binary_open_threshold must be greater than or equal to "
                "gripper_binary_threshold"
            )
        if self.gripper_action_mode not in ("continuous", "binary"):
            raise ValueError(
                "gripper_action_mode must be 'continuous' or 'binary', "
                f"got {self.gripper_action_mode!r}"
            )
        self.gripper_obs_mode = str(gripper_obs_mode)
        if self.gripper_obs_mode not in ("measured", "commanded", "auto"):
            raise ValueError(
                "gripper_obs_mode must be 'measured', 'commanded', or 'auto', "
                f"got {self.gripper_obs_mode!r}"
            )
        self.gripper_obs_raw_offset = float(gripper_obs_raw_offset)
        self.move_to_start_on_reset = bool(move_to_start_on_reset)
        self.move_to_start_pose_path = (
            None if move_to_start_pose_path is None or str(move_to_start_pose_path).strip() == ""
            else str(move_to_start_pose_path)
        )
        self.reset_episode = int(reset_episode)
        self.move_to_start_duration = float(move_to_start_duration)
        self.move_to_start_frequency = float(move_to_start_frequency)
        self.move_to_start_settle = bool(move_to_start_settle)
        self.move_to_start_settle_timeout = float(move_to_start_settle_timeout)
        self.move_to_start_settle_frequency = float(move_to_start_settle_frequency)
        self.move_to_start_pos_tolerance = float(move_to_start_pos_tolerance)
        self.move_to_start_rot_tolerance = np.deg2rad(float(move_to_start_rot_tolerance_deg))
        self.move_to_start_strict = bool(move_to_start_strict)
        self.save_processed_image = bool(save_processed_image)
        self.processed_image_output_dir = (
            processed_image_output_dir
            if processed_image_output_dir is not None
            else (
                os.path.join(output_dir, "debug_images")
                if output_dir is not None
                else os.path.join("data", "eval_outputs", "franka_polymetis", "debug_images")
            )
        )
        self.processed_image_saved = False
        self.processed_image_capture_enabled = False
        self.processed_image_count = 0
        self.processed_image_lock = threading.Lock()
        self.enable_policy_recording = bool(enable_policy_recording)
        self.policy_recording_output_dir = (
            policy_recording_output_dir
            if policy_recording_output_dir is not None
            else (
                os.path.join(output_dir, "policy_recordings")
                if output_dir is not None
                else os.path.join("data", "eval_outputs", "franka_polymetis", "policy_recordings")
            )
        )
        self.policy_recording_fps = float(policy_recording_fps) if policy_recording_fps is not None else float(max_fps)
        self.policy_recording_image_width = (
            None if policy_recording_image_width is None else int(policy_recording_image_width)
        )
        self.policy_recording_image_height = (
            None if policy_recording_image_height is None else int(policy_recording_image_height)
        )
        if self.policy_recording_image_width is not None and self.policy_recording_image_width <= 0:
            raise ValueError("policy_recording_image_width must be positive when set")
        if self.policy_recording_image_height is not None and self.policy_recording_image_height <= 0:
            raise ValueError("policy_recording_image_height must be positive when set")
        self.policy_recording_plot_width = int(policy_recording_plot_width)
        self.policy_recording_plot_window_sec = float(policy_recording_plot_window_sec)
        self.policy_recording_codec = str(policy_recording_codec)
        self.policy_recording_lock = threading.Lock()
        self.policy_recording_active = False
        self.policy_recording_started_at = None
        self.policy_recording_writer = None
        self.policy_recording_video_path = None
        self.policy_recording_npz_path = None
        self.policy_recording_csv_path = None
        self.policy_recording_records = []
        self.policy_recording_magnet_normalizer = None
        self.last_policy_action_command = np.full((16,), np.nan, dtype=np.float32)
        self.predicted_magnet_lock = threading.Lock()
        self.latest_predicted_tactile_emb = None
        self.latest_predicted_normalized_tactile_emb = None
        self.tcp_target_duration = (
            1.0 / max_fps if tcp_target_duration is None else float(tcp_target_duration)
        )
        self.tcp_move_timeout = float(tcp_move_timeout)
        self.http_timeout = float(http_timeout)
        self.gripper_http_timeout = float(gripper_http_timeout)
        self.enable_exp_recording = bool(enable_exp_recording)
        self.camera_backend = str(camera_backend).lower()
        self.camera_preprocess_mode = str(camera_preprocess_mode).lower()
        if self.camera_preprocess_mode not in ("square_crop", "resize"):
            raise ValueError(
                "camera_preprocess_mode must be one of square_crop/resize, "
                f"got {self.camera_preprocess_mode!r}"
            )
        self.camera_square_crop_bottom_rows = int(camera_square_crop_bottom_rows)
        if not 0 <= self.camera_square_crop_bottom_rows < 224:
            raise ValueError(
                "camera_square_crop_bottom_rows must be in [0, 223], "
                f"got {self.camera_square_crop_bottom_rows}"
            )
        self.camera_color_match_dataset_start = _as_bool(camera_color_match_dataset_start)
        self.camera_color_match_dataset_episode = (
            self.reset_episode
            if camera_color_match_dataset_episode is None
            else int(camera_color_match_dataset_episode)
        )
        self.camera_color_match_timeout = max(0.0, float(camera_color_match_timeout))
        self.camera_color_match_target = None
        self.camera_color_match_gain = None
        self.camera_color_match_offset = None
        self.camera_color_match_pending = False
        self.camera_color_match_lock = threading.Lock()
        self.camera_color_match_event = threading.Event()
        if self.camera_color_match_dataset_start:
            self.camera_color_match_target = _load_dataset_start_image_stats(
                dataset_path,
                self.camera_color_match_dataset_episode,
                self.image_shape,
            )
            logger.info(
                "Will photometrically align live camera frames to dataset start: "
                f"path={self.camera_color_match_target['path']}, "
                f"key={self.camera_color_match_target['image_key']}, "
                f"episode={self.camera_color_match_target['episode_idx']}, "
                f"row={self.camera_color_match_target['row']}, "
                f"target_mean={np.round(self.camera_color_match_target['mean'], 2).tolist()}, "
                f"target_std={np.round(self.camera_color_match_target['std'], 2).tolist()}"
            )
        self.camera_zed_view = str(camera_zed_view).lower()
        self.camera_zed_resolution = str(camera_zed_resolution)
        self.camera_zed_depth_mode = str(camera_zed_depth_mode)
        self.camera_iphone_bind_host = str(camera_iphone_bind_host)
        self.camera_iphone_video_port = int(camera_iphone_video_port)
        self.camera_iphone_combined_port = int(camera_iphone_combined_port)
        self.camera_iphone_phone_ip = str(camera_iphone_phone_ip or "")
        if (
                self.camera_backend == "iphone"
                and not self.camera_iphone_phone_ip
                and str(camera_source) in ("", "auto", "none", "None")):
            self.camera_iphone_phone_ip = "172.20.10.1"
            logger.info(
                "CAMERA_BACKEND=iphone with CAMERA_SOURCE=auto; "
                "using ARPoseStreamer UI default phone IP 172.20.10.1. "
                "Set CAMERA_SOURCE or CAMERA_IPHONE_PHONE_IP to override."
            )
        self.camera_iphone_registration_port = int(camera_iphone_registration_port)
        self.camera_iphone_startup_timeout = float(camera_iphone_startup_timeout)
        self.camera_iphone_read_timeout = float(camera_iphone_read_timeout)
        self.camera_iphone_hello_interval = float(camera_iphone_hello_interval)
        self.camera_flip = bool(camera_flip)
        if self.camera_backend not in ("auto", "opencv", "realsense", "zed", "zed_v4l", "iphone"):
            raise ValueError(
                "camera_backend must be one of auto/opencv/realsense/zed/zed_v4l/iphone, "
                f"got {self.camera_backend!r}"
            )
        if self.camera_zed_view not in ("left", "right", "full"):
            raise ValueError(
                "camera_zed_view must be one of left/right/full, "
                f"got {self.camera_zed_view!r}"
            )
        self.enable_magnet = bool(enable_magnet)
        self.magnet_required = bool(magnet_required)
        self.magnet_tactile_key = str(magnet_tactile_key)
        self.magnet_tactile_dim = int(magnet_tactile_dim)
        if magnet_used_sensor_count is None:
            inferred_magnet = _infer_magnet_used_sensor_count(
                dataset_path,
                self.magnet_tactile_key,
                self.magnet_tactile_dim,
            )
            if inferred_magnet is None:
                self.magnet_used_sensor_count = MAGNET_USED_SENSOR_COUNT
            else:
                self.magnet_used_sensor_count, inferred_path = inferred_magnet
                logger.info(
                    "Using magnet sensor count inferred from "
                    f"{inferred_path}: {self.magnet_used_sensor_count}"
                )
        else:
            self.magnet_used_sensor_count = int(magnet_used_sensor_count)
        if self.magnet_used_sensor_count < 1 or self.magnet_used_sensor_count > MAGNET_SENSOR_COUNT:
            raise ValueError(
                f"magnet_used_sensor_count must be in [1, {MAGNET_SENSOR_COUNT}], "
                f"got {self.magnet_used_sensor_count}"
            )
        if self.magnet_tactile_dim < self.magnet_used_sensor_count * 3:
            raise ValueError(
                "magnet_tactile_dim is too small for magnet_used_sensor_count: "
                f"{self.magnet_tactile_dim} < {self.magnet_used_sensor_count * 3}"
            )
        self.magnet_sensor_order = _parse_magnet_sensor_order(
            magnet_sensor_order,
            self.magnet_used_sensor_count,
        )
        self.magnet_zero_channels = _parse_magnet_zero_channels(
            magnet_zero_channels,
            self.magnet_tactile_dim,
        )
        identity_sensor_order = np.arange(self.magnet_used_sensor_count, dtype=np.int64)
        if not np.array_equal(self.magnet_sensor_order, identity_sensor_order):
            mapping = ", ".join(
                f"policy S{policy_idx + 1} <- live S{live_idx + 1}"
                for policy_idx, live_idx in enumerate(self.magnet_sensor_order)
            )
            logger.info(f"Remapping magnet sensors before policy obs: {mapping}")
        if self.magnet_zero_channels.size > 0:
            logger.info(
                "Zeroing magnet tactile channels before policy obs: "
                f"{(self.magnet_zero_channels + 1).tolist()}"
            )
        self.magnet_normalize_to_first_frame = bool(magnet_normalize_to_first_frame)
        self.magnet_rezero_after_policy_start_sec = (
            None
            if magnet_rezero_after_policy_start_sec is None
            else float(magnet_rezero_after_policy_start_sec)
        )
        if self.magnet_rezero_after_policy_start_sec is not None and (
                not np.isfinite(self.magnet_rezero_after_policy_start_sec)
                or self.magnet_rezero_after_policy_start_sec < 0.0):
            raise ValueError(
                "magnet_rezero_after_policy_start_sec must be finite and non-negative, "
                f"got {magnet_rezero_after_policy_start_sec}"
            )
        self.magnet_reader_subtract_baseline = _as_bool(magnet_reader_subtract_baseline)
        self.magnet_filter_abnormal_readings = _as_bool(magnet_filter_abnormal_readings)
        self.magnet_abnormal_abs_threshold = float(magnet_abnormal_abs_threshold)
        self.magnet_last_valid_xyz = None
        self.magnet_abnormal_replaced_count = 0
        self.magnet_baseline = None
        self.magnet_baseline_lock = threading.Lock()
        self.magnet_rezero_lock = threading.Lock()
        self.magnet_rezero_deadline = None
        self.magnet_rezero_pending = False
        self.magnet_rezero_baseline = None
        self.magnet_reader = None
        if self.enable_magnet:
            self.magnet_reader = _MagnetometerReader(
                port=magnet_port,
                baudrate=magnet_baudrate,
                samples_per_frame=magnet_samples_per_frame,
                used_sensor_count=self.magnet_used_sensor_count,
                subtract_baseline=self.magnet_reader_subtract_baseline,
            )
            try:
                self.magnet_reader.start()
                warmup_deadline = time.monotonic() + float(magnet_warmup_timeout)
                min_warmup_samples = max(1, int(magnet_samples_per_frame))
                while self.magnet_reader.sample_count < min_warmup_samples and time.monotonic() < warmup_deadline:
                    time.sleep(0.02)
                if self.magnet_reader.sample_count < min_warmup_samples:
                    message = (
                        f"Magnetometer opened on {magnet_port} but produced only "
                        f"{self.magnet_reader.sample_count}/{min_warmup_samples} warmup samples "
                        f"within {float(magnet_warmup_timeout):.2f}s"
                    )
                    if self.magnet_required:
                        raise RuntimeError(message)
                    logger.warning(message)
                else:
                    logger.info(
                        f"Magnetometer reading from {magnet_port} at {int(magnet_baudrate)} baud, "
                        f"samples={self.magnet_reader.sample_count}"
                    )
                    if self.magnet_normalize_to_first_frame:
                        logger.info(
                            "Magnet normalization enabled: subtracting first frame "
                            "baseline for policy obs and recording"
                        )
                    if self.magnet_filter_abnormal_readings:
                        logger.info(
                            "Magnet abnormal reading filter enabled before policy obs: "
                            f"abs_threshold={self.magnet_abnormal_abs_threshold:g}"
                        )
            except Exception:
                self.magnet_reader.stop()
                self.magnet_reader = None
                if self.magnet_required:
                    raise
                logger.exception("Failed to initialize optional magnetometer")

        self.magnet2_port = None if magnet2_port is None else str(magnet2_port).strip()
        self.magnet2_required = (
            bool(self.magnet2_port) and self.magnet_required
            if magnet2_required is None
            else _as_bool(magnet2_required)
        )
        self.magnet2_tactile_key = str(magnet2_tactile_key)
        self.magnet2_tactile_dim = int(magnet2_tactile_dim)
        if magnet2_used_sensor_count is None:
            inferred_magnet2 = _infer_magnet_used_sensor_count(
                dataset_path,
                self.magnet2_tactile_key,
                self.magnet2_tactile_dim,
            )
            if inferred_magnet2 is None:
                self.magnet2_used_sensor_count = MAGNET_USED_SENSOR_COUNT
            else:
                self.magnet2_used_sensor_count, inferred_path = inferred_magnet2
                logger.info(
                    "Using second magnet sensor count inferred from "
                    f"{inferred_path}: {self.magnet2_used_sensor_count}"
                )
        else:
            self.magnet2_used_sensor_count = int(magnet2_used_sensor_count)
        if self.magnet2_used_sensor_count < 1 or self.magnet2_used_sensor_count > MAGNET_SENSOR_COUNT:
            raise ValueError(
                f"magnet2_used_sensor_count must be in [1, {MAGNET_SENSOR_COUNT}], "
                f"got {self.magnet2_used_sensor_count}"
            )
        if self.magnet2_tactile_dim < self.magnet2_used_sensor_count * 3:
            raise ValueError(
                "magnet2_tactile_dim is too small for magnet2_used_sensor_count: "
                f"{self.magnet2_tactile_dim} < {self.magnet2_used_sensor_count * 3}"
            )
        self.magnet2_sensor_order = _parse_magnet_sensor_order(
            magnet2_sensor_order,
            self.magnet2_used_sensor_count,
        )
        self.magnet2_zero_channels = _parse_magnet_zero_channels(
            magnet2_zero_channels,
            self.magnet2_tactile_dim,
        )
        self.magnet2_normalize_to_first_frame = self.magnet_normalize_to_first_frame
        self.magnet2_reader_subtract_baseline = self.magnet_reader_subtract_baseline
        self.magnet2_filter_abnormal_readings = self.magnet_filter_abnormal_readings
        self.magnet2_abnormal_abs_threshold = self.magnet_abnormal_abs_threshold
        self.magnet2_last_valid_xyz = None
        self.magnet2_abnormal_replaced_count = 0
        self.magnet2_baseline = None
        self.magnet2_baseline_lock = threading.Lock()
        self.magnet2_rezero_lock = threading.Lock()
        self.magnet2_rezero_deadline = None
        self.magnet2_rezero_pending = False
        self.magnet2_rezero_baseline = None
        self.magnet2_reader = None
        if self.enable_magnet and self.magnet2_required and not self.magnet2_port:
            if self.magnet_reader is not None:
                self.magnet_reader.stop()
                self.magnet_reader = None
            raise ValueError("magnet2_port is required when magnet2_required=True")
        if self.enable_magnet and self.magnet2_port:
            if str(magnet_port).strip() == self.magnet2_port:
                if self.magnet_reader is not None:
                    self.magnet_reader.stop()
                    self.magnet_reader = None
                raise ValueError("magnet_port and magnet2_port must be different")
            self.magnet2_reader = _MagnetometerReader(
                port=self.magnet2_port,
                baudrate=magnet_baudrate,
                samples_per_frame=magnet_samples_per_frame,
                used_sensor_count=self.magnet2_used_sensor_count,
                subtract_baseline=self.magnet2_reader_subtract_baseline,
            )
            try:
                self.magnet2_reader.start()
                warmup_deadline = time.monotonic() + float(magnet_warmup_timeout)
                min_warmup_samples = max(1, int(magnet_samples_per_frame))
                while (
                    self.magnet2_reader.sample_count < min_warmup_samples
                    and time.monotonic() < warmup_deadline
                ):
                    time.sleep(0.02)
                if self.magnet2_reader.sample_count < min_warmup_samples:
                    message = (
                        f"Second magnetometer opened on {self.magnet2_port} but produced only "
                        f"{self.magnet2_reader.sample_count}/{min_warmup_samples} warmup samples "
                        f"within {float(magnet_warmup_timeout):.2f}s"
                    )
                    if self.magnet2_required:
                        raise RuntimeError(message)
                    logger.warning(message)
                else:
                    logger.info(
                        f"Second magnetometer reading from {self.magnet2_port} at "
                        f"{int(magnet_baudrate)} baud, samples={self.magnet2_reader.sample_count}"
                    )
            except Exception:
                self.magnet2_reader.stop()
                self.magnet2_reader = None
                if self.magnet2_required:
                    if self.magnet_reader is not None:
                        self.magnet_reader.stop()
                        self.magnet_reader = None
                    raise
                logger.exception("Failed to initialize optional second magnetometer")

        self.data_processing_manager = SimpleNamespace(use_6d_rotation=True)
        self.reset_start = None
        if self.move_to_start_on_reset:
            if self.move_to_start_pose_path is not None:
                self.reset_start = _load_manual_start_pose(self.move_to_start_pose_path)
                logger.info(
                    "Will move to manual start pose before policy execution: "
                    f"path={self.reset_start['path']}, "
                    f"source={self.reset_start['source']}, "
                    f"target_xyz={self.reset_start['pose7d'][:3].round(4).tolist()}, "
                    f"duration={self.move_to_start_duration:.2f}s, "
                    f"settle={self.move_to_start_settle}, "
                    f"settle_timeout={self.move_to_start_settle_timeout:.2f}s, "
                    f"pos_tol={self.move_to_start_pos_tolerance:.4f}m, "
                    f"rot_tol={np.rad2deg(self.move_to_start_rot_tolerance):.2f}deg, "
                    f"strict={self.move_to_start_strict}"
                )
            else:
                self.reset_start = _load_dataset_start_pose(dataset_path, self.reset_episode)
                logger.info(
                    "Will move to dataset start before policy execution: "
                    f"path={self.reset_start['path']}, "
                    f"episode={self.reset_start['episode_idx']}/{self.reset_start['episode_count'] - 1}, "
                    f"rows=[{self.reset_start['row_start']}, {self.reset_start['row_end']}), "
                    f"source={self.reset_start['source']}, "
                    f"duration={self.move_to_start_duration:.2f}s, "
                    f"settle={self.move_to_start_settle}, "
                    f"settle_timeout={self.move_to_start_settle_timeout:.2f}s, "
                    f"pos_tol={self.move_to_start_pos_tolerance:.4f}m, "
                    f"rot_tol={np.rad2deg(self.move_to_start_rot_tolerance):.2f}deg, "
                    f"strict={self.move_to_start_strict}"
                )
        self.gripper_raw_range_mode = str(gripper_raw_range_mode)
        self.gripper_raw_calibration_scale = float(gripper_raw_calibration_scale)
        self.gripper_raw_calibration_offset = float(gripper_raw_calibration_offset)
        inferred = _infer_gripper_raw_range(
            dataset_path,
            gripper_raw_open_quantile,
            self.gripper_raw_range_mode,
            self.gripper_raw_calibration_scale,
            self.gripper_raw_calibration_offset,
            self.gripper_stroke,
        )
        if inferred is not None:
            inferred_min, inferred_max, inferred_abs_max, inferred_open_quantile, inferred_path = inferred
            if self.gripper_raw_range_mode == "trajectory_zero_to_max":
                full_open_source = "action absolute maximum"
            elif self.gripper_raw_range_mode == "calibrated_marker_trajectory":
                full_open_source = "calibrated action absolute maximum"
            elif self.gripper_raw_range_mode == "calibrated_marker_width":
                full_open_source = "calibrated physical gripper stroke"
            else:
                full_open_source = f"dataset quantile {inferred_open_quantile:.3f}"
            logger.info(
                "Using gripper raw range from "
                f"{inferred_path}: "
                f"[{inferred_min:.4f}, {inferred_max:.4f}], "
                f"mode={self.gripper_raw_range_mode}, "
                f"full_open_source={full_open_source}, "
                f"absolute_max={inferred_abs_max:.4f}"
            )
            if gripper_raw_min is None:
                gripper_raw_min = inferred_min
            if gripper_raw_max is None:
                gripper_raw_max = inferred_max
        self.gripper_raw_min = 0.0 if gripper_raw_min is None else float(gripper_raw_min)
        self.gripper_raw_max = 1.0 if gripper_raw_max is None else float(gripper_raw_max)
        self.max_gripper_width = (
            self.gripper_raw_max if max_gripper_width is None else float(max_gripper_width)
        )
        self.min_gripper_width = (
            self.gripper_raw_min if min_gripper_width is None else float(min_gripper_width)
        )
        self.gripper_calibrated_full_open = None
        if self.gripper_raw_range_mode in (
                "calibrated_marker_trajectory",
                "calibrated_marker_width"):
            if not np.isfinite(self.gripper_raw_calibration_scale) or self.gripper_raw_calibration_scale <= 0.0:
                raise ValueError("gripper_raw_calibration_scale must be finite and positive")
            if not np.isfinite(self.gripper_raw_calibration_offset):
                raise ValueError("gripper_raw_calibration_offset must be finite")
        if self.gripper_raw_range_mode == "calibrated_marker_trajectory":
            self.gripper_calibrated_full_open = (
                self.gripper_raw_calibration_scale * self.gripper_raw_max
                + self.gripper_raw_calibration_offset
            )
            if self.gripper_calibrated_full_open <= 0.0:
                raise ValueError(
                    "Calibrated full-open gripper gap must be positive; "
                    f"got {self.gripper_calibrated_full_open:.6f}m"
                )
        if self.gripper_width_threshold_config is None:
            self.gripper_width_threshold = self._width_to_raw_gripper(self.gripper_width_threshold_m)
            threshold_source = f"{self.gripper_width_threshold_m:.4f}m mapped to raw"
        else:
            self.gripper_width_threshold = float(self.gripper_width_threshold_config)
            threshold_source = "raw config"
        self.last_gripper_width_target = [self.max_gripper_width, self.max_gripper_width]
        self.last_gripper_binary_open = None
        self.gripper_clip_log_count = 0
        self.gripper_obs_auto_fallback_log_count = 0
        auto_obs_zero_fallback = (
            "width0_force0_to_commanded"
            if self.gripper_obs_mode == "auto"
            else "disabled"
        )
        logger.info(
            "FR3 gripper setup: "
            f"action_mode={self.gripper_action_mode}, "
            f"obs_mode={self.gripper_obs_mode}, "
            f"obs_raw_offset={self.gripper_obs_raw_offset:+.4f}, "
            f"stroke={self.gripper_stroke:.4f}m, "
            f"binary_close_threshold={self.gripper_binary_threshold:.4f}, "
            f"binary_open_threshold={self.gripper_binary_open_threshold:.4f}, "
            f"raw_range=[{self.gripper_raw_min:.4f}, {self.gripper_raw_max:.4f}], "
            f"raw_to_width=[{self.gripper_raw_min:.4f}->0.0000m, "
            f"{self.gripper_raw_max:.4f}->{self.gripper_stroke:.4f}m], "
            f"calibration=(scale={self.gripper_raw_calibration_scale:.8f}, "
            f"offset={self.gripper_raw_calibration_offset:+.8f}m, "
            f"full_open={self.gripper_calibrated_full_open}), "
            f"command_width_offset={self.gripper_command_width_offset:+.4f}m, "
            f"fixed_physical_width={self.fixed_gripper_width_m}, "
            f"width_threshold={self.gripper_width_threshold:.4f} ({threshold_source}), "
            f"command_deadband={self.gripper_control_width_precision:.4f}, "
            f"ignore_commands={self.ignore_gripper_commands}, "
            f"ignore_policy_commands={self.ignore_policy_gripper_commands}, "
            f"auto_obs_zero_fallback={auto_obs_zero_fallback}"
        )

        self.state_session = requests.Session()
        self.command_session = requests.Session()
        self.obs_buffer = deque(maxlen=1024)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        if self.camera_backend == "zed":
            self.camera = _ZEDSDKCamera(
                source=camera_source,
                fps=camera_fps,
                require_color=require_color_camera,
                color_threshold=camera_color_threshold,
                view=self.camera_zed_view,
                resolution=self.camera_zed_resolution,
                depth_mode=self.camera_zed_depth_mode,
            )
        elif self.camera_backend == "iphone":
            self.camera = _IPhoneUDPCamera(
                source=camera_source,
                bind_host=self.camera_iphone_bind_host,
                video_port=self.camera_iphone_video_port,
                combined_port=self.camera_iphone_combined_port,
                phone_ip=self.camera_iphone_phone_ip,
                registration_port=self.camera_iphone_registration_port,
                startup_timeout=self.camera_iphone_startup_timeout,
                read_timeout=self.camera_iphone_read_timeout,
                hello_interval=self.camera_iphone_hello_interval,
                require_color=require_color_camera,
                color_threshold=camera_color_threshold,
            )
        else:
            self.camera = _OpenCVCamera(
                source=camera_source,
                width=camera_width,
                height=camera_height,
                fps=camera_fps,
                require_color=require_color_camera,
                color_threshold=camera_color_threshold,
                backend=self.camera_backend,
            )
        self.camera.open()
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.thread.start()

    def _clip_gripper_raw(self, raw_width, *, log=False, context="gripper raw"):
        raw_width = float(raw_width)
        if not np.isfinite(raw_width):
            clipped = float(np.clip(
                self.last_gripper_width_target[0],
                self.gripper_raw_min,
                self.gripper_raw_max,
            ))
            logger.warning(
                f"Received non-finite {context}={raw_width}; "
                f"using previous clipped raw={clipped:.4f}"
            )
            return clipped
        clipped = float(np.clip(raw_width, self.gripper_raw_min, self.gripper_raw_max))
        if log and abs(clipped - raw_width) > 1e-6:
            self.gripper_clip_log_count += 1
            if self.gripper_clip_log_count <= 10 or self.gripper_clip_log_count % 50 == 0:
                logger.warning(
                    f"Clipped {context} from {raw_width:.4f} to {clipped:.4f} "
                    f"to match configured raw range "
                    f"[{self.gripper_raw_min:.4f}, {self.gripper_raw_max:.4f}]"
                )
        return clipped

    def _raw_gripper_to_width(self, value):
        value = self._clip_gripper_raw(value)
        if self.gripper_raw_range_mode == "calibrated_marker_width":
            return float(np.clip(
                self.gripper_raw_calibration_scale * float(value)
                + self.gripper_raw_calibration_offset,
                0.0,
                self.gripper_stroke,
            ))
        if self.gripper_raw_range_mode == "calibrated_marker_trajectory":
            calibrated_gap = (
                self.gripper_raw_calibration_scale * float(value)
                + self.gripper_raw_calibration_offset
            )
            open_ratio = np.clip(
                calibrated_gap / self.gripper_calibrated_full_open,
                0.0,
                1.0,
            )
            return float(open_ratio * self.gripper_stroke)
        denom = max(self.gripper_raw_max - self.gripper_raw_min, 1e-6)
        open_ratio = np.clip((float(value) - self.gripper_raw_min) / denom, 0.0, 1.0)
        return float(open_ratio * self.gripper_stroke)

    def _width_to_raw_gripper(self, width):
        if self.gripper_raw_range_mode == "calibrated_marker_width":
            width = float(np.clip(width, 0.0, self.gripper_stroke))
            raw_width = (
                width - self.gripper_raw_calibration_offset
            ) / self.gripper_raw_calibration_scale
            return self._clip_gripper_raw(raw_width)
        open_ratio = np.clip(float(width) / max(self.gripper_stroke, 1e-6), 0.0, 1.0)
        if self.gripper_raw_range_mode == "calibrated_marker_trajectory":
            calibrated_gap = open_ratio * self.gripper_calibrated_full_open
            raw_width = (
                calibrated_gap - self.gripper_raw_calibration_offset
            ) / self.gripper_raw_calibration_scale
            return self._clip_gripper_raw(raw_width)
        return float(self.gripper_raw_min + open_ratio * (self.gripper_raw_max - self.gripper_raw_min))

    def _raw_gripper_to_command_width(self, value):
        nominal_width = self._raw_gripper_to_width(value)
        return float(np.clip(
            nominal_width + self.gripper_command_width_offset,
            0.0,
            self.gripper_stroke,
        ))

    def _physical_gripper_width_to_nominal_width(self, width):
        return float(np.clip(
            float(width) - self.gripper_command_width_offset,
            0.0,
            self.gripper_stroke,
        ))

    def _width_to_obs_raw_gripper(self, width):
        nominal_width = self._physical_gripper_width_to_nominal_width(width)
        return float(
            self._width_to_raw_gripper(nominal_width)
            + self.gripper_obs_raw_offset
        )

    def _commanded_gripper_obs_raw(self):
        commanded_raw = self._clip_gripper_raw(self.last_gripper_width_target[0])
        return self._width_to_obs_raw_gripper(
            self._raw_gripper_to_command_width(commanded_raw)
        )

    def _preprocess_bgr(self, frame_bgr):
        frame_bgr = self._select_camera_view_bgr(frame_bgr)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.camera_flip:
            rgb = cv2.rotate(rgb, cv2.ROTATE_180)
        c, out_h, out_w = self.image_shape
        if c != 3:
            raise ValueError(f"Expected RGB image shape [3,H,W], got {self.image_shape}")
        if self.camera_preprocess_mode == "resize":
            interpolation = cv2.INTER_AREA
            if out_h > rgb.shape[0] or out_w > rgb.shape[1]:
                interpolation = cv2.INTER_LINEAR
            rgb = cv2.resize(rgb, (out_w, out_h), interpolation=interpolation)
        else:
            h, w = rgb.shape[:2]
            sq = min(h, w)
            y0 = (h - sq) // 2
            x0 = (w - sq) // 2
            rgb = rgb[y0:y0 + sq, x0:x0 + sq]
            rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
            if self.camera_square_crop_bottom_rows > 0:
                keep_rows = 224 - self.camera_square_crop_bottom_rows
                rgb = rgb[:keep_rows, :]
                rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
            if (out_h, out_w) != (224, 224):
                rgb = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
        return self._apply_camera_color_match(rgb.astype(np.uint8))

    def _apply_camera_color_match(self, rgb):
        if not self.camera_color_match_dataset_start:
            return rgb
        with self.camera_color_match_lock:
            if self.camera_color_match_pending and self.camera_color_match_gain is None:
                image_float = rgb.astype(np.float32)
                live_mean = image_float.mean(axis=(0, 1))
                live_std = image_float.std(axis=(0, 1))
                target_mean = self.camera_color_match_target["mean"]
                target_std = self.camera_color_match_target["std"]
                self.camera_color_match_gain = target_std / np.maximum(live_std, 1e-6)
                self.camera_color_match_offset = (
                    target_mean - self.camera_color_match_gain * live_mean
                )
                self.camera_color_match_pending = False
                self.camera_color_match_event.set()
                logger.info(
                    "Calibrated live camera color to dataset start: "
                    f"live_mean={np.round(live_mean, 2).tolist()}, "
                    f"live_std={np.round(live_std, 2).tolist()}, "
                    f"gain={np.round(self.camera_color_match_gain, 4).tolist()}, "
                    f"offset={np.round(self.camera_color_match_offset, 2).tolist()}"
                )
            gain = self.camera_color_match_gain
            offset = self.camera_color_match_offset
        if gain is None or offset is None:
            return rgb
        corrected = rgb.astype(np.float32) * gain + offset
        return np.clip(corrected, 0.0, 255.0).astype(np.uint8)

    def _select_camera_view_bgr(self, frame_bgr):
        if self.camera_backend != "zed_v4l" or self.camera_zed_view == "full":
            return frame_bgr
        h, w = frame_bgr.shape[:2]
        if w < int(2.5 * h):
            return frame_bgr
        half_w = w // 2
        if self.camera_zed_view == "left":
            return frame_bgr[:, :half_w]
        if self.camera_zed_view == "right":
            return frame_bgr[:, half_w:2 * half_w]
        return frame_bgr

    def _maybe_save_processed_image(self, image):
        if not self.save_processed_image:
            return
        with self.processed_image_lock:
            if not self.processed_image_capture_enabled or self.processed_image_saved:
                return
            os.makedirs(self.processed_image_output_dir, exist_ok=True)
            path = os.path.join(
                self.processed_image_output_dir,
                (
                    f"processed_left_wrist_img_{self.processed_image_count:03d}_"
                    f"{time.strftime('%Y%m%d_%H%M%S')}.png"
                ),
            )
            cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            self.processed_image_saved = True
            self.processed_image_count += 1
        logger.info(f"Saved processed policy image to {path}, shape={image.shape}, dtype={image.dtype}")

    def _get_current_tcp(self):
        response = self.state_session.get(f"{self.base_url}/get_current_tcp/left", timeout=self.http_timeout)
        response.raise_for_status()
        pose7d = np.asarray(response.json(), dtype=np.float64)
        return pose7d

    def _get_gripper_state(self):
        response = self.state_session.get(f"{self.base_url}/get_current_robot_states", timeout=self.http_timeout)
        response.raise_for_status()
        state = response.json()
        gripper_state = state.get("leftGripperState", [self.gripper_stroke, 0.0])
        width = float(gripper_state[0])
        force = float(gripper_state[1]) if len(gripper_state) > 1 else 0.0
        return width, force

    def _get_gripper_width(self):
        width, _ = self._get_gripper_state()
        return width

    def _is_invalid_zero_gripper_measurement(self, measured_width, measured_force):
        commanded_raw = self._clip_gripper_raw(self.last_gripper_width_target[0])
        commanded_width = self._raw_gripper_to_command_width(commanded_raw)
        return (
            float(measured_width) <= 1e-6
            and abs(float(measured_force)) <= 1e-6
            and commanded_width >= max(0.01, 0.5 * self.gripper_stroke)
        )

    def _get_obs_gripper_raw(self):
        if self.gripper_obs_mode == "commanded":
            return self._commanded_gripper_obs_raw()
        measured_width, measured_force = self._get_gripper_state()
        if (
            self.gripper_obs_mode == "auto"
            and self._is_invalid_zero_gripper_measurement(measured_width, measured_force)
        ):
            fallback_raw = self._commanded_gripper_obs_raw()
            self.gripper_obs_auto_fallback_log_count += 1
            if self.gripper_obs_auto_fallback_log_count == 1:
                logger.warning(
                    "Robot server reported gripper width=0 and force=0 while "
                    "the commanded/open state is nonzero; treating measured "
                    f"width as invalid and using commanded raw={fallback_raw:.4f} "
                    "for policy observation. Set GRIPPER_OBS_MODE=measured to "
                    "force raw measured readings. If you need Robotiq measured "
                    "state, make sure the Robotiq Polymetis gripper server is "
                    "running and restart the Franka robot server."
                )
            elif self.gripper_obs_auto_fallback_log_count % 300 == 0:
                logger.info(
                    "Still using commanded gripper raw for policy observation "
                    f"because measured gripper state remains width=0, force=0 "
                    f"(count={self.gripper_obs_auto_fallback_log_count}, "
                    f"raw={fallback_raw:.4f})."
                )
            return fallback_raw
        return self._width_to_obs_raw_gripper(measured_width)

    def sync_commanded_gripper_state_to_measured(self):
        try:
            measured_width = self._get_gripper_width()
            nominal_width = self._physical_gripper_width_to_nominal_width(
                measured_width
            )
            measured_raw = self._width_to_raw_gripper(nominal_width)
        except Exception as exc:
            logger.warning(f"Failed to sync commanded gripper state to measured state: {exc}")
            return
        self.last_gripper_width_target[0] = float(measured_raw)
        logger.info(
            "Synced commanded gripper observation to current measured width: "
            f"raw={measured_raw:.4f}"
        )

    @staticmethod
    def _magnet_attr_name(name, input_index):
        if input_index == 0:
            return name
        if input_index == 1 and name.startswith("magnet"):
            return f"magnet2{name[len('magnet'):]}"
        raise ValueError(f"Unsupported magnetometer input index {input_index}")

    def _get_magnet_attr(self, name, input_index):
        return getattr(self, self._magnet_attr_name(name, input_index))

    def _set_magnet_attr(self, name, input_index, value):
        setattr(self, self._magnet_attr_name(name, input_index), value)

    def _magnet_input_indices(self):
        return [
            input_index
            for input_index in range(2)
            if getattr(
                self,
                self._magnet_attr_name("magnet_reader", input_index),
                None,
            ) is not None
        ]

    def reset_magnet_baseline(self, input_index=0):
        baseline_lock = self._get_magnet_attr("magnet_baseline_lock", input_index)
        with baseline_lock:
            self._set_magnet_attr("magnet_baseline", input_index, None)
        reader = self._get_magnet_attr("magnet_reader", input_index)
        subtract_baseline = self._get_magnet_attr(
            "magnet_reader_subtract_baseline", input_index
        )
        if reader is not None and subtract_baseline:
            reader.reset_baseline()

    def reset_magnet_filter(self, input_index=0):
        self._set_magnet_attr("magnet_last_valid_xyz", input_index, None)
        self._set_magnet_attr("magnet_abnormal_replaced_count", input_index, 0)

    def _reset_magnet_policy_state(self, reason: str):
        clear_obs = False
        for input_index in self._magnet_input_indices():
            if self._get_magnet_attr("magnet_filter_abnormal_readings", input_index):
                self.reset_magnet_filter(input_index)
                logger.info(
                    f"Reset magnetometer {input_index + 1} abnormal reading filter "
                    f"at {reason}"
                )
            if (
                self._get_magnet_attr("magnet_reader_subtract_baseline", input_index)
                or self._get_magnet_attr("magnet_normalize_to_first_frame", input_index)
            ):
                self.reset_magnet_baseline(input_index)
                clear_obs = True
                logger.info(
                    f"Reset magnetometer {input_index + 1} baseline at {reason}"
                )
        if clear_obs:
            with self.lock:
                self.obs_buffer.clear()

    def prepare_policy_start(self):
        for input_index in self._magnet_input_indices():
            rezero_lock = self._get_magnet_attr("magnet_rezero_lock", input_index)
            with rezero_lock:
                self._set_magnet_attr("magnet_rezero_deadline", input_index, None)
                self._set_magnet_attr("magnet_rezero_pending", input_index, False)
                self._set_magnet_attr("magnet_rezero_baseline", input_index, None)
        self._reset_magnet_policy_state("policy start")
        if not self.camera_color_match_dataset_start:
            return
        with self.camera_color_match_lock:
            self.camera_color_match_gain = None
            self.camera_color_match_offset = None
            self.camera_color_match_pending = True
            self.camera_color_match_event.clear()
        if not self.camera_color_match_event.wait(timeout=self.camera_color_match_timeout):
            raise RuntimeError(
                "Timed out waiting for a live frame to calibrate camera color matching"
            )
        with self.lock:
            self.obs_buffer.clear()
        with self.processed_image_lock:
            self.processed_image_saved = False
            self.processed_image_capture_enabled = True

    def notify_policy_started(self):
        delay = self.magnet_rezero_after_policy_start_sec
        input_indices = self._magnet_input_indices()
        if delay is None or not input_indices:
            return
        deadline = time.monotonic() + delay
        for input_index in input_indices:
            rezero_lock = self._get_magnet_attr("magnet_rezero_lock", input_index)
            with rezero_lock:
                self._set_magnet_attr(
                    "magnet_rezero_deadline", input_index, deadline
                )
                self._set_magnet_attr("magnet_rezero_pending", input_index, True)
                self._set_magnet_attr("magnet_rezero_baseline", input_index, None)
        logger.info(
            f"Scheduled {len(input_indices)} magnetometer input(s) for ignore-and-rezero "
            f"{delay:.3f}s after policy start: tactile obs will be zero "
            "until the baseline is captured"
        )

    def _apply_scheduled_magnet_rezero(
        self, magnet_xyz, sample_count, input_index=0
    ):
        magnet_xyz = np.asarray(magnet_xyz, dtype=np.float32)
        sample_count_value = int(np.asarray(sample_count).reshape(-1)[0])
        captured_baseline = None
        ignore_until_baseline = False
        rezero_lock = self._get_magnet_attr("magnet_rezero_lock", input_index)
        with rezero_lock:
            if (
                    self._get_magnet_attr("magnet_rezero_pending", input_index)
                    and self._get_magnet_attr("magnet_rezero_deadline", input_index) is not None
            ):
                deadline = self._get_magnet_attr(
                    "magnet_rezero_deadline", input_index
                )
                if time.monotonic() >= deadline and sample_count_value > 0:
                    captured_baseline = _magnetic_time_mean(
                        magnet_xyz[None, ...],
                        sample_count=sample_count,
                    )[0]
                    self._set_magnet_attr(
                        "magnet_rezero_baseline",
                        input_index,
                        captured_baseline.copy(),
                    )
                    self._set_magnet_attr("magnet_rezero_pending", input_index, False)
                    self._set_magnet_attr("magnet_rezero_deadline", input_index, None)
                else:
                    ignore_until_baseline = True
            rezero_baseline = self._get_magnet_attr(
                "magnet_rezero_baseline", input_index
            )
            baseline = None if rezero_baseline is None else rezero_baseline.copy()

        if captured_baseline is not None:
            baseline_lock = self._get_magnet_attr(
                "magnet_baseline_lock", input_index
            )
            with baseline_lock:
                self._set_magnet_attr("magnet_baseline", input_index, None)
            with self.lock:
                self.obs_buffer.clear()
            logger.info(
                f"Applied one-time magnetometer {input_index + 1} rezero after policy start: "
                f"baseline={np.round(captured_baseline, 3).tolist()}"
            )
        if ignore_until_baseline:
            return np.zeros_like(magnet_xyz, dtype=np.float32)
        if baseline is None:
            return magnet_xyz
        return (magnet_xyz - baseline[None, :, :]).astype(np.float32)

    def _filter_magnet_for_policy(self, magnet_xyz, input_index=0):
        if not self._get_magnet_attr("magnet_filter_abnormal_readings", input_index):
            return np.asarray(magnet_xyz, dtype=np.float32)

        filtered_xyz, last_valid_xyz, replaced_count = (
            _filter_abnormal_magnet_readings_live(
                magnet_xyz,
                last_valid_xyz=self._get_magnet_attr(
                    "magnet_last_valid_xyz", input_index
                ),
                threshold=self._get_magnet_attr(
                    "magnet_abnormal_abs_threshold", input_index
                ),
            )
        )
        self._set_magnet_attr("magnet_last_valid_xyz", input_index, last_valid_xyz)
        if replaced_count > 0:
            total = self._get_magnet_attr(
                "magnet_abnormal_replaced_count", input_index
            ) + replaced_count
            self._set_magnet_attr("magnet_abnormal_replaced_count", input_index, total)
            if (
                total <= 20
                or total % 100 == 0
            ):
                logger.warning(
                    f"Filtered abnormal magnetometer {input_index + 1} readings "
                    "before policy obs: "
                    f"replaced={replaced_count}, "
                    f"total={total}, "
                    f"abs_threshold={self._get_magnet_attr('magnet_abnormal_abs_threshold', input_index):g}"
                )
        return filtered_xyz

    def _apply_magnet_first_frame_normalization(
        self, magnet_xyz, sample_count, input_index=0
    ):
        magnet_xyz = np.asarray(magnet_xyz, dtype=np.float32)
        magnet_mean = _magnetic_time_mean(
            magnet_xyz[None, ...],
            sample_count=sample_count,
        )[0]
        if not self._get_magnet_attr(
            "magnet_normalize_to_first_frame", input_index
        ):
            return magnet_xyz, magnet_mean
        baseline_lock = self._get_magnet_attr("magnet_baseline_lock", input_index)
        with baseline_lock:
            baseline = self._get_magnet_attr("magnet_baseline", input_index)
            if baseline is None:
                baseline = magnet_mean.copy()
                self._set_magnet_attr("magnet_baseline", input_index, baseline)
                logger.info(
                    f"Captured magnetometer {input_index + 1} baseline for "
                    f"first-frame normalization: {np.round(baseline, 3).tolist()}"
                )
            baseline = baseline.copy()
        normalized_xyz = magnet_xyz - baseline[None, :, :]
        normalized_mean = magnet_mean - baseline
        return normalized_xyz.astype(np.float32), normalized_mean.astype(np.float32)

    def _sample_magnet_input(self, input_index):
        reader = self._get_magnet_attr("magnet_reader", input_index)
        magnet_obs = reader.get_recent_samples()
        policy_magnet_xyz = _remap_magnet_sensors(
            magnet_obs["magnet_xyz"],
            self._get_magnet_attr("magnet_sensor_order", input_index),
        )
        filtered_magnet_xyz = self._filter_magnet_for_policy(
            policy_magnet_xyz,
            input_index=input_index,
        )
        filtered_magnet_xyz = self._apply_scheduled_magnet_rezero(
            filtered_magnet_xyz,
            magnet_obs["magnet_sample_count"],
            input_index=input_index,
        )
        magnet_xyz, magnet_mean = self._apply_magnet_first_frame_normalization(
            filtered_magnet_xyz,
            magnet_obs["magnet_sample_count"],
            input_index=input_index,
        )
        zero_channels = self._get_magnet_attr("magnet_zero_channels", input_index)
        if zero_channels.size > 0:
            magnet_xyz = magnet_xyz.copy()
            magnet_mean = magnet_mean.copy()
            for channel_idx in zero_channels:
                sensor_idx = int(channel_idx) // 3
                axis_idx = int(channel_idx) % 3
                if sensor_idx < magnet_xyz.shape[1]:
                    magnet_xyz[:, sensor_idx, axis_idx] = 0.0
                if sensor_idx < magnet_mean.shape[0]:
                    magnet_mean[sensor_idx, axis_idx] = 0.0
        tactile_emb = _magnet_to_tactile_embedding(
            magnet_xyz[None, ...],
            sample_count=magnet_obs["magnet_sample_count"],
            output_dim=self._get_magnet_attr("magnet_tactile_dim", input_index),
        )[0]
        if zero_channels.size > 0:
            tactile_emb = tactile_emb.copy()
            tactile_emb[zero_channels] = 0.0
        prefix = "magnet" if input_index == 0 else f"magnet{input_index + 1}"
        return {
            "tactile_key": self._get_magnet_attr("magnet_tactile_key", input_index),
            "tactile_emb": tactile_emb.astype(np.float32),
            "sample_count_key": f"{prefix}_sample_count",
            "sample_count": magnet_obs["magnet_sample_count"].astype(np.int32),
            "timestamp_key": f"{prefix}_timestamp_ns",
            "timestamp_ns": magnet_obs["magnet_timestamp_ns"].astype(np.int64),
            "magnet_mean": magnet_mean,
        }

    def _sample_once(self):
        frame = self.camera.read()
        image = self._preprocess_bgr(frame)
        self._maybe_save_processed_image(image)
        pose6d = pose_7d_to_pose_6d(self._get_current_tcp())
        gripper_raw = self._get_obs_gripper_raw()
        timestamp = time.time()
        obs = {
            "left_wrist_img": image,
            "left_robot_tcp_pose": pose_6d_to_pose_9d(pose6d).astype(np.float32),
            "left_robot_gripper_width": np.array([gripper_raw], dtype=np.float32),
            "timestamp": np.array([timestamp], dtype=np.float64),
        }
        magnet_means = {}
        for input_index in self._magnet_input_indices():
            magnet_input = self._sample_magnet_input(input_index)
            obs[magnet_input["tactile_key"]] = magnet_input["tactile_emb"]
            obs[magnet_input["sample_count_key"]] = magnet_input["sample_count"]
            obs[magnet_input["timestamp_key"]] = magnet_input["timestamp_ns"]
            magnet_means[input_index] = magnet_input["magnet_mean"]
        try:
            self._record_policy_frame(image, obs, magnet_means)
        except Exception as exc:
            logger.warning(f"Policy recording frame failed: {exc}")
        return obs

    def _extract_policy_recording_magnet_normalizer(self, policy_normalizer):
        if policy_normalizer is None:
            return None
        try:
            params_dict = getattr(policy_normalizer, "params_dict", None)
            if params_dict is None:
                return None
            try:
                params = params_dict[self.magnet_tactile_key]
            except Exception:
                return None

            def to_numpy(value):
                if hasattr(value, "detach"):
                    value = value.detach().cpu().numpy()
                return np.asarray(value, dtype=np.float32).reshape(-1)

            scale = to_numpy(params["scale"])
            offset = to_numpy(params["offset"])
            if scale.shape != offset.shape:
                raise ValueError(
                    f"scale shape {scale.shape} does not match offset shape {offset.shape}"
                )
            return {"scale": scale, "offset": offset}
        except Exception as exc:
            logger.warning(
                "Could not extract policy normalizer for magnet recording "
                f"key={self.magnet_tactile_key!r}: {exc}"
            )
            return None

    def _normalize_policy_recording_tactile(self, tactile_emb):
        params = self.policy_recording_magnet_normalizer
        normalized = np.full_like(tactile_emb, np.nan, dtype=np.float32)
        if params is None:
            return normalized
        dim = min(tactile_emb.shape[0], params["scale"].shape[0])
        if dim <= 0:
            return normalized
        normalized[:dim] = tactile_emb[:dim] * params["scale"][:dim] + params["offset"][:dim]
        return normalized

    def set_predicted_magnet_future(
            self,
            tactile_emb=None,
            normalized_tactile_emb=None):
        with self.predicted_magnet_lock:
            self.latest_predicted_tactile_emb = (
                None
                if tactile_emb is None
                else np.asarray(tactile_emb, dtype=np.float32).copy()
            )
            self.latest_predicted_normalized_tactile_emb = (
                None
                if normalized_tactile_emb is None
                else np.asarray(normalized_tactile_emb, dtype=np.float32).copy()
            )

    def start_policy_recording(self, episode_idx, policy_normalizer=None):
        if not self.enable_policy_recording:
            return None
        magnet_normalizer = self._extract_policy_recording_magnet_normalizer(policy_normalizer)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        stem = f"episode_{int(episode_idx):03d}_{timestamp}"
        os.makedirs(self.policy_recording_output_dir, exist_ok=True)
        with self.policy_recording_lock:
            if self.policy_recording_active:
                self._stop_policy_recording_locked()
            self.policy_recording_active = True
            self.policy_recording_started_at = time.time()
            self.policy_recording_writer = None
            self.policy_recording_video_path = os.path.join(
                self.policy_recording_output_dir,
                f"{stem}_magnet_overlay.mp4",
            )
            self.policy_recording_npz_path = os.path.join(
                self.policy_recording_output_dir,
                f"{stem}_magnet_trace.npz",
            )
            self.policy_recording_csv_path = os.path.join(
                self.policy_recording_output_dir,
                f"{stem}_magnet_trace.csv",
            )
            self.policy_recording_records = []
            self.policy_recording_magnet_normalizer = magnet_normalizer
        if magnet_normalizer is None:
            logger.warning(
                "Policy recording will not include normalized magnet values; "
                f"normalizer key {self.magnet_tactile_key!r} was not available."
            )
        else:
            logger.info(
                "Policy recording will include normalized magnet values for "
                f"{self.magnet_tactile_key!r}"
            )
        logger.info(f"Started policy recording to {self.policy_recording_video_path}")
        return self.policy_recording_video_path

    def stop_policy_recording(self):
        with self.policy_recording_lock:
            return self._stop_policy_recording_locked()

    def _stop_policy_recording_locked(self):
        if not self.policy_recording_active:
            return None
        self.policy_recording_active = False
        writer = self.policy_recording_writer
        records = list(self.policy_recording_records)
        video_path = self.policy_recording_video_path
        npz_path = self.policy_recording_npz_path
        csv_path = self.policy_recording_csv_path
        self.policy_recording_writer = None
        self.policy_recording_records = []
        self.policy_recording_video_path = None
        self.policy_recording_npz_path = None
        self.policy_recording_csv_path = None
        self.policy_recording_started_at = None
        self.policy_recording_magnet_normalizer = None
        if writer is not None:
            writer.release()
        if records:
            self._save_policy_recording_trace(records, npz_path, csv_path)
        logger.info(
            f"Stopped policy recording: video={video_path}, "
            f"trace_npz={npz_path}, trace_csv={csv_path}, frames={len(records)}"
        )
        return video_path

    @staticmethod
    def _coerce_recording_magnet_mean(value, sensor_count):
        result = np.full((sensor_count, 3), np.nan, dtype=np.float32)
        if value is None:
            return result
        rows = np.asarray(value, dtype=np.float32).reshape(-1, 3)
        copy_rows = min(sensor_count, rows.shape[0])
        result[:copy_rows] = rows[:copy_rows]
        return result

    def _record_policy_frame(self, image, obs, magnet_means):
        with self.policy_recording_lock:
            if not self.policy_recording_active:
                return
            if not isinstance(magnet_means, dict):
                magnet_means = {0: magnet_means}
            timestamp = float(obs["timestamp"][0])
            elapsed = timestamp - float(self.policy_recording_started_at)
            tactile_emb = obs.get(self.magnet_tactile_key)
            if tactile_emb is None:
                tactile_emb = np.full((self.magnet_tactile_dim,), np.nan, dtype=np.float32)
            else:
                tactile_emb = np.asarray(tactile_emb, dtype=np.float32).reshape(-1)
            normalized_tactile_emb = self._normalize_policy_recording_tactile(tactile_emb)
            magnet_mean = self._coerce_recording_magnet_mean(
                magnet_means.get(0),
                self.magnet_used_sensor_count,
            )
            magnet_magnitude = np.linalg.norm(np.nan_to_num(magnet_mean, nan=0.0), axis=1).astype(np.float32)
            sample_count = int(np.asarray(obs.get("magnet_sample_count", [0])).reshape(-1)[0])
            tcp_pose_obs = np.asarray(
                obs.get("left_robot_tcp_pose", np.full((9,), np.nan, dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1)
            gripper_obs = np.asarray(
                obs.get("left_robot_gripper_width", np.full((1,), np.nan, dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1)
            last_action_command = np.asarray(
                getattr(self, "last_policy_action_command", np.full((16,), np.nan, dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1)
            with self.predicted_magnet_lock:
                predicted_tactile_emb = (
                    None
                    if self.latest_predicted_tactile_emb is None
                    else self.latest_predicted_tactile_emb.copy()
                )
                predicted_normalized_tactile_emb = (
                    None
                    if self.latest_predicted_normalized_tactile_emb is None
                    else self.latest_predicted_normalized_tactile_emb.copy()
                )
            record = {
                "timestamp": timestamp,
                "elapsed": elapsed,
                "magnet_sample_count": sample_count,
                "magnet_mean": magnet_mean.copy(),
                "magnet_magnitude": magnet_magnitude.copy(),
                "tactile_emb": tactile_emb.copy(),
                "normalized_tactile_emb": normalized_tactile_emb.copy(),
                "predicted_tactile_emb": predicted_tactile_emb,
                "predicted_normalized_tactile_emb": predicted_normalized_tactile_emb,
                "tcp_pose_obs": tcp_pose_obs.copy(),
                "gripper_obs": gripper_obs.copy(),
                "last_action_command": last_action_command.copy(),
            }
            magnet2_mean = magnet_means.get(1)
            tactile2_emb = obs.get(getattr(self, "magnet2_tactile_key", ""))
            if magnet2_mean is not None or tactile2_emb is not None:
                sensor_count2 = int(
                    getattr(self, "magnet2_used_sensor_count", self.magnet_used_sensor_count)
                )
                tactile_dim2 = int(
                    getattr(self, "magnet2_tactile_dim", self.magnet_tactile_dim)
                )
                magnet2_mean = self._coerce_recording_magnet_mean(
                    magnet2_mean,
                    sensor_count2,
                )
                if tactile2_emb is None:
                    tactile2_emb = np.full((tactile_dim2,), np.nan, dtype=np.float32)
                else:
                    tactile2_emb = np.asarray(tactile2_emb, dtype=np.float32).reshape(-1)
                record.update(
                    {
                        "magnet2_sample_count": int(
                            np.asarray(obs.get("magnet2_sample_count", [0])).reshape(-1)[0]
                        ),
                        "magnet2_mean": magnet2_mean.copy(),
                        "magnet2_magnitude": np.linalg.norm(
                            np.nan_to_num(magnet2_mean, nan=0.0), axis=1
                        ).astype(np.float32),
                        "tactile2_emb": tactile2_emb.copy(),
                    }
                )
            self.policy_recording_records.append(record)
            frame = self._render_policy_recording_frame(image, self.policy_recording_records)
            if self.policy_recording_writer is None:
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*self.policy_recording_codec[:4])
                writer = cv2.VideoWriter(
                    self.policy_recording_video_path,
                    fourcc,
                    self.policy_recording_fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open video writer: {self.policy_recording_video_path}")
                self.policy_recording_writer = writer
            self.policy_recording_writer.write(frame)

    def _render_policy_recording_frame(self, image, records):
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        target_width, target_height = self._policy_recording_image_size(image_bgr)
        if (image_bgr.shape[1], image_bgr.shape[0]) != (target_width, target_height):
            interpolation = (
                cv2.INTER_AREA
                if target_width < image_bgr.shape[1] or target_height < image_bgr.shape[0]
                else cv2.INTER_LINEAR
            )
            image_bgr = cv2.resize(
                image_bgr,
                (target_width, target_height),
                interpolation=interpolation,
            )
        height = image_bgr.shape[0]
        panel_width = max(620, self.policy_recording_plot_width)
        panel = np.full((height, panel_width, 3), 250, dtype=np.uint8)
        gripper_panel_height = min(104, max(76, height // 4))
        gripper_panel_top = max(0, height - gripper_panel_height)
        axis_colors = {
            "X": (220, 60, 60),
            "Y": (60, 170, 60),
            "Z": (60, 100, 220),
        }
        dual_magnet = bool(
            records and any(record.get("magnet2_mean") is not None for record in records)
        )
        cv2.putText(
            panel,
            "Magnet",
            (12, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        raw_left = 58
        raw_right = panel_width // 2 - 14
        norm_left = panel_width // 2 + 50
        norm_right = panel_width - 12
        raw_label_x = raw_left + max(0, (raw_right - raw_left) // 2 - 32)
        norm_label_x = norm_left + max(0, (norm_right - norm_left) // 2 - 38)
        left_label = "Magnet 1 delta" if dual_magnet else "Raw delta"
        right_label = "Magnet 2 delta" if dual_magnet else "Policy norm"
        cv2.putText(panel, left_label, (raw_label_x, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1, cv2.LINE_AA)
        cv2.putText(panel, right_label, (norm_label_x, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1, cv2.LINE_AA)
        if not dual_magnet:
            legend_x = panel_width - 112
            for axis_name, color in axis_colors.items():
                cv2.putText(
                    panel,
                    axis_name,
                    (legend_x, 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    color,
                    1,
                    cv2.LINE_AA,
                )
                legend_x += 34
        if records:
            now_elapsed = records[-1]["elapsed"]
            start_elapsed = max(0.0, now_elapsed - self.policy_recording_plot_window_sec)
            recent = [record for record in records if record["elapsed"] >= start_elapsed]
            times = np.array([record["elapsed"] for record in recent], dtype=np.float32)
            raw_values = np.stack([record["magnet_mean"] for record in records], axis=0).astype(np.float32)
            recent_values = np.stack([record["magnet_mean"] for record in recent], axis=0).astype(np.float32)

            def first_valid_baseline(values, sensor_count):
                baseline = np.full((sensor_count, 3), np.nan, dtype=np.float32)
                for sensor_idx in range(sensor_count):
                    for axis_idx in range(3):
                        channel = values[:, sensor_idx, axis_idx]
                        valid = np.flatnonzero(np.isfinite(channel))
                        if valid.size > 0:
                            baseline[sensor_idx, axis_idx] = channel[valid[0]]
                return baseline

            baseline = first_valid_baseline(raw_values, self.magnet_used_sensor_count)
            raw_delta_values = recent_values - baseline[None, :, :]
            raw_finite = np.isfinite(raw_delta_values)
            predicted_normalized_values = None
            plot_sensor_count = self.magnet_used_sensor_count

            if dual_magnet:
                magnet2_sensor_count = int(
                    getattr(self, "magnet2_used_sensor_count", self.magnet_used_sensor_count)
                )
                magnet2_values = np.stack(
                    [
                        self._coerce_recording_magnet_mean(
                            record.get("magnet2_mean"), magnet2_sensor_count
                        )
                        for record in records
                    ],
                    axis=0,
                )
                magnet2_recent_values = np.stack(
                    [
                        self._coerce_recording_magnet_mean(
                            record.get("magnet2_mean"), magnet2_sensor_count
                        )
                        for record in recent
                    ],
                    axis=0,
                )
                magnet2_baseline = first_valid_baseline(
                    magnet2_values, magnet2_sensor_count
                )
                right_plot_values = (
                    magnet2_recent_values - magnet2_baseline[None, :, :]
                )
                norm_finite_flat = right_plot_values[np.isfinite(right_plot_values)]
                plot_sensor_count = max(plot_sensor_count, magnet2_sensor_count)
            else:
                normalized_emb = []
                for record in recent:
                    value = record.get("normalized_tactile_emb")
                    if value is None:
                        value = np.full((self.magnet_tactile_dim,), np.nan, dtype=np.float32)
                    normalized_emb.append(np.asarray(value, dtype=np.float32).reshape(-1))
                normalized_emb = np.stack(normalized_emb, axis=0)
                right_plot_values = np.full(
                    (len(recent), self.magnet_used_sensor_count, 3),
                    np.nan,
                    dtype=np.float32,
                )
                norm_dim = min(normalized_emb.shape[1], self.magnet_used_sensor_count * 3)
                if norm_dim > 0:
                    right_plot_values.reshape(len(recent), -1)[:, :norm_dim] = normalized_emb[:, :norm_dim]
                predicted_normalized_emb = recent[-1].get("predicted_normalized_tactile_emb")
                if predicted_normalized_emb is not None:
                    predicted_normalized_emb = np.asarray(predicted_normalized_emb, dtype=np.float32)
                    if predicted_normalized_emb.ndim == 2 and predicted_normalized_emb.shape[0] > 0:
                        predicted_normalized_values = np.full(
                            (predicted_normalized_emb.shape[0], self.magnet_used_sensor_count, 3),
                            np.nan,
                            dtype=np.float32,
                        )
                        pred_dim = min(
                            predicted_normalized_emb.shape[1],
                            self.magnet_used_sensor_count * 3,
                        )
                        predicted_normalized_values.reshape(predicted_normalized_emb.shape[0], -1)[:, :pred_dim] = (
                            predicted_normalized_emb[:, :pred_dim]
                        )
                norm_finite_values = [right_plot_values[np.isfinite(right_plot_values)]]
                if predicted_normalized_values is not None:
                    norm_finite_values.append(predicted_normalized_values[np.isfinite(predicted_normalized_values)])
                norm_finite_flat = (
                    np.concatenate([value.reshape(-1) for value in norm_finite_values if value.size > 0])
                    if any(value.size > 0 for value in norm_finite_values)
                    else np.asarray([], dtype=np.float32)
                )
            header_height = 24
            plot_height = max(1, gripper_panel_top - header_height - 2)
            row_height = max(1, plot_height // plot_sensor_count)
            raw_limit = max(
                1.0,
                float(np.percentile(np.abs(raw_delta_values[raw_finite]), 99.0)) if np.any(raw_finite) else 1.0,
            )
            norm_limit = max(
                1.0,
                float(np.percentile(np.abs(norm_finite_flat), 99.0)) if norm_finite_flat.size > 0 else 1.0,
            )
            x_span = max(1e-3, float(times[-1] - times[0]))

            def draw_plot(plot_values, sensor_idx, graph_left, graph_right, center_y, amplitude, value_limit):
                if sensor_idx >= plot_values.shape[1]:
                    return [np.nan, np.nan, np.nan]
                latest_values = []
                for axis_idx, axis_name in enumerate(("X", "Y", "Z")):
                    axis_values = plot_values[:, sensor_idx, axis_idx]
                    valid = np.flatnonzero(np.isfinite(axis_values))
                    latest_values.append(float(axis_values[valid[-1]]) if valid.size > 0 else np.nan)
                    points = []
                    for idx in valid:
                        t = times[idx]
                        value = axis_values[idx]
                        x = int(graph_left + (float(t - times[0]) / x_span) * (graph_right - graph_left))
                        y = int(center_y - np.clip(float(value) / value_limit, -1.0, 1.0) * amplitude)
                        points.append((x, y))
                    if len(points) >= 2:
                        cv2.polylines(
                            panel,
                            [np.asarray(points, dtype=np.int32)],
                            False,
                            axis_colors[axis_name],
                            1,
                            cv2.LINE_AA,
                        )
                    elif len(points) == 1:
                        cv2.circle(panel, points[0], 2, axis_colors[axis_name], -1, cv2.LINE_AA)
                return latest_values

            def draw_prediction(plot_values, sensor_idx, graph_left, graph_right, center_y, amplitude, value_limit):
                if plot_values is None or plot_values.shape[0] <= 0:
                    return [np.nan, np.nan, np.nan]
                latest_values = []
                pred_span = max(1, plot_values.shape[0] - 1)
                for axis_idx, axis_name in enumerate(("X", "Y", "Z")):
                    axis_values = plot_values[:, sensor_idx, axis_idx]
                    valid = np.flatnonzero(np.isfinite(axis_values))
                    latest_values.append(float(axis_values[valid[-1]]) if valid.size > 0 else np.nan)
                    points = []
                    for idx in valid:
                        value = axis_values[idx]
                        x = int(graph_left + (float(idx) / pred_span) * (graph_right - graph_left))
                        y = int(center_y - np.clip(float(value) / value_limit, -1.0, 1.0) * amplitude)
                        points.append((x, y))
                    if len(points) >= 2:
                        cv2.polylines(
                            panel,
                            [np.asarray(points, dtype=np.int32)],
                            False,
                            axis_colors[axis_name],
                            2,
                            cv2.LINE_AA,
                        )
                    for point in points[::max(1, len(points) // 8)]:
                        cv2.circle(panel, point, 2, axis_colors[axis_name], -1, cv2.LINE_AA)
                return latest_values

            cv2.line(panel, (0, gripper_panel_top), (panel_width, gripper_panel_top), (190, 190, 190), 1)
            cv2.putText(
                panel,
                "Gripper obs",
                (12, gripper_panel_top + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
            gripper_top = gripper_panel_top + 24
            gripper_bottom = height - 10
            gripper_left = 58
            gripper_right = panel_width - 12
            gripper_track_height = max(1, gripper_bottom - gripper_top)
            gripper_limit_mm = max(1.0, float(self.gripper_stroke) * 1000.0)
            gripper_raw_values = np.stack(
                [
                    np.asarray(record.get("gripper_obs", np.full((1,), np.nan, dtype=np.float32)), dtype=np.float32).reshape(-1)[0:1]
                    for record in recent
                ],
                axis=0,
            ).astype(np.float32).reshape(-1)
            gripper_mm_values = np.asarray(
                [
                    self._raw_gripper_to_width(value) * 1000.0 if np.isfinite(value) else np.nan
                    for value in gripper_raw_values
                ],
                dtype=np.float32,
            )
            gripper_valid = np.flatnonzero(np.isfinite(gripper_mm_values))
            latest_gripper_mm = (
                float(gripper_mm_values[gripper_valid[-1]]) if gripper_valid.size > 0 else np.nan
            )
            cv2.line(panel, (gripper_left, gripper_top), (gripper_left, gripper_bottom), (210, 210, 210), 1)
            cv2.line(panel, (gripper_left, gripper_bottom), (gripper_right, gripper_bottom), (210, 210, 210), 1)
            if gripper_valid.size > 0:
                gripper_points = []
                for idx in gripper_valid:
                    t = times[idx]
                    value = float(gripper_mm_values[idx])
                    x = int(gripper_left + (float(t - times[0]) / x_span) * (gripper_right - gripper_left))
                    y = int(
                        gripper_bottom
                        - np.clip(value / gripper_limit_mm, 0.0, 1.0) * gripper_track_height
                    )
                    gripper_points.append((x, y))
                if len(gripper_points) >= 2:
                    cv2.polylines(
                        panel,
                        [np.asarray(gripper_points, dtype=np.int32)],
                        False,
                        (140, 80, 180),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    cv2.circle(panel, gripper_points[0], 2, (140, 80, 180), -1, cv2.LINE_AA)
            gripper_text = (
                f"obs={latest_gripper_mm:.1f} mm"
                if np.isfinite(latest_gripper_mm)
                else "obs=nan"
            )
            cv2.putText(
                panel,
                gripper_text,
                (gripper_left, gripper_top + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (45, 45, 45),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                panel,
                f"0",
                (gripper_left - 16, gripper_bottom + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (90, 90, 90),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                panel,
                f"{gripper_limit_mm:.0f}",
                (gripper_right - 24, gripper_top + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (90, 90, 90),
                1,
                cv2.LINE_AA,
            )
            for sensor_idx in range(plot_sensor_count):
                y0 = header_height + sensor_idx * row_height
                y1 = (
                    gripper_panel_top - 2
                    if sensor_idx == plot_sensor_count - 1
                    else header_height + (sensor_idx + 1) * row_height
                )
                top = y0 + 18
                bottom = max(top + 4, y1 - 8)
                center_y = (top + bottom) // 2
                amplitude = max((bottom - top) * 0.45, 1.0)
                cv2.line(panel, (0, y0), (panel_width, y0), (210, 210, 210), 1)
                cv2.line(panel, (raw_left, center_y), (raw_right, center_y), (200, 200, 200), 1)
                cv2.line(panel, (norm_left, center_y), (norm_right, center_y), (200, 200, 200), 1)
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
                raw_latest = draw_plot(raw_delta_values, sensor_idx, raw_left, raw_right, center_y, amplitude, raw_limit)
                norm_latest = draw_plot(right_plot_values, sensor_idx, norm_left, norm_right, center_y, amplitude, norm_limit)
                pred_latest = (
                    [np.nan, np.nan, np.nan]
                    if dual_magnet
                    else draw_prediction(
                        predicted_normalized_values,
                        sensor_idx,
                        norm_left,
                        norm_right,
                        center_y,
                        amplitude,
                        norm_limit,
                    )
                )
                raw_text = " ".join(
                    f"d{axis}={value:.0f}" if np.isfinite(value) else f"d{axis}=nan"
                    for axis, value in zip(("X", "Y", "Z"), raw_latest)
                )
                if dual_magnet:
                    norm_text = " ".join(
                        f"d{axis}={value:.0f}" if np.isfinite(value) else f"d{axis}=nan"
                        for axis, value in zip(("X", "Y", "Z"), norm_latest)
                    )
                else:
                    norm_text = " ".join(
                        f"n{axis}={value:.2f}" if np.isfinite(value) else f"n{axis}=nan"
                        for axis, value in zip(("X", "Y", "Z"), norm_latest)
                    )
                cv2.putText(panel, raw_text, (raw_left, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.31, (45, 45, 45), 1, cv2.LINE_AA)
                cv2.putText(panel, norm_text, (norm_left, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.31, (45, 45, 45), 1, cv2.LINE_AA)
                if np.any(np.isfinite(pred_latest)):
                    pred_text = " ".join(
                        f"p{axis}={value:.2f}" if np.isfinite(value) else f"p{axis}=nan"
                        for axis, value in zip(("X", "Y", "Z"), pred_latest)
                    )
                    cv2.putText(panel, pred_text, (norm_left, y0 + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.31, (25, 25, 25), 1, cv2.LINE_AA)
            left_limit_text = f"M1 +/-{raw_limit:.0f}" if dual_magnet else f"raw +/-{raw_limit:.0f}"
            right_limit_text = f"M2 +/-{norm_limit:.0f}" if dual_magnet else f"norm/pred +/-{norm_limit:.1f}"
            cv2.putText(panel, left_limit_text, (12, height - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1, cv2.LINE_AA)
            cv2.putText(panel, right_limit_text, (panel_width // 2 + 2, height - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1, cv2.LINE_AA)
            cv2.putText(panel, f"t={now_elapsed:5.2f}s", (panel_width - 88, height - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (40, 40, 40), 1, cv2.LINE_AA)
        else:
            cv2.putText(panel, "No magnet samples", (12, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)
        return np.concatenate([image_bgr, panel], axis=1)

    def _policy_recording_image_size(self, image_bgr):
        height, width = image_bgr.shape[:2]
        target_width = self.policy_recording_image_width
        target_height = self.policy_recording_image_height
        if target_width is None and target_height is None:
            return width, height
        if target_width is None:
            target_width = int(round(width * float(target_height) / float(height)))
        if target_height is None:
            target_height = int(round(height * float(target_width) / float(width)))
        return max(1, int(target_width)), max(1, int(target_height))

    def _save_policy_recording_trace(self, records, npz_path, csv_path):
        timestamps = np.asarray([record["timestamp"] for record in records], dtype=np.float64)
        elapsed = np.asarray([record["elapsed"] for record in records], dtype=np.float64)
        sample_counts = np.asarray([record["magnet_sample_count"] for record in records], dtype=np.int32)
        magnet_mean = np.stack([record["magnet_mean"] for record in records], axis=0).astype(np.float32)
        magnet_magnitude = np.stack([record["magnet_magnitude"] for record in records], axis=0).astype(np.float32)
        tactile_emb = np.stack([record["tactile_emb"] for record in records], axis=0).astype(np.float32)
        normalized_tactile_emb = np.stack(
            [
                record.get(
                    "normalized_tactile_emb",
                    np.full_like(record["tactile_emb"], np.nan, dtype=np.float32),
                )
                for record in records
            ],
            axis=0,
        ).astype(np.float32)
        has_magnet2 = any(record.get("magnet2_mean") is not None for record in records)
        if has_magnet2:
            magnet2_sensor_count = int(
                getattr(self, "magnet2_used_sensor_count", self.magnet_used_sensor_count)
            )
            magnet2_tactile_dim = int(
                getattr(self, "magnet2_tactile_dim", self.magnet_tactile_dim)
            )
            magnet2_sample_counts = np.asarray(
                [record.get("magnet2_sample_count", 0) for record in records],
                dtype=np.int32,
            )
            magnet2_mean = np.stack(
                [
                    self._coerce_recording_magnet_mean(
                        record.get("magnet2_mean"), magnet2_sensor_count
                    )
                    for record in records
                ],
                axis=0,
            )
            magnet2_magnitude = np.linalg.norm(
                np.nan_to_num(magnet2_mean, nan=0.0), axis=2
            ).astype(np.float32)
            tactile2_emb = np.stack(
                [
                    np.asarray(
                        record.get(
                            "tactile2_emb",
                            np.full((magnet2_tactile_dim,), np.nan, dtype=np.float32),
                        ),
                        dtype=np.float32,
                    ).reshape(-1)
                    for record in records
                ],
                axis=0,
            )
        tcp_pose_obs = np.stack(
            [
                record.get("tcp_pose_obs", np.full((9,), np.nan, dtype=np.float32))
                for record in records
            ],
            axis=0,
        ).astype(np.float32)
        gripper_obs = np.stack(
            [
                record.get("gripper_obs", np.full((1,), np.nan, dtype=np.float32))
                for record in records
            ],
            axis=0,
        ).astype(np.float32)
        last_action_command = np.stack(
            [
                record.get("last_action_command", np.full((16,), np.nan, dtype=np.float32))
                for record in records
            ],
            axis=0,
        ).astype(np.float32)

        def stack_variable_prediction(key):
            values = [record.get(key) for record in records]
            max_steps = 0
            dim = self.magnet_tactile_dim
            for value in values:
                if value is None:
                    continue
                arr = np.asarray(value, dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr[None, :]
                if arr.ndim != 2:
                    continue
                max_steps = max(max_steps, arr.shape[0])
                dim = max(dim, arr.shape[1])
            if max_steps <= 0:
                return np.full((len(records), 0, dim), np.nan, dtype=np.float32)
            out = np.full((len(records), max_steps, dim), np.nan, dtype=np.float32)
            for idx, value in enumerate(values):
                if value is None:
                    continue
                arr = np.asarray(value, dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr[None, :]
                if arr.ndim != 2:
                    continue
                steps = min(max_steps, arr.shape[0])
                width = min(dim, arr.shape[1])
                out[idx, :steps, :width] = arr[:steps, :width]
            return out

        predicted_tactile_emb = stack_variable_prediction("predicted_tactile_emb")
        predicted_normalized_tactile_emb = stack_variable_prediction("predicted_normalized_tactile_emb")
        trace = {
            "timestamp": timestamps,
            "elapsed": elapsed,
            "magnet_sample_count": sample_counts,
            "magnet_mean": magnet_mean,
            "magnet_magnitude": magnet_magnitude,
            "tactile_emb": tactile_emb,
            "normalized_tactile_emb": normalized_tactile_emb,
            "predicted_tactile_emb": predicted_tactile_emb,
            "predicted_normalized_tactile_emb": predicted_normalized_tactile_emb,
            "tcp_pose_obs": tcp_pose_obs,
            "gripper_obs": gripper_obs,
            "last_action_command": last_action_command,
        }
        if has_magnet2:
            trace.update(
                {
                    "magnet2_sample_count": magnet2_sample_counts,
                    "magnet2_mean": magnet2_mean,
                    "magnet2_magnitude": magnet2_magnitude,
                    "tactile2_emb": tactile2_emb,
                }
            )
        np.savez_compressed(npz_path, **trace)
        header = ["timestamp", "elapsed", "magnet_sample_count"]
        for sensor_idx in range(self.magnet_used_sensor_count):
            for axis in ("x", "y", "z"):
                header.append(f"magnet_s{sensor_idx + 1}_{axis}")
            header.append(f"magnet_s{sensor_idx + 1}_magnitude")
        header.extend([f"tactile_emb_{idx}" for idx in range(tactile_emb.shape[1])])
        if has_magnet2:
            header.append("magnet2_sample_count")
            for sensor_idx in range(magnet2_sensor_count):
                for axis in ("x", "y", "z"):
                    header.append(f"magnet2_s{sensor_idx + 1}_{axis}")
                header.append(f"magnet2_s{sensor_idx + 1}_magnitude")
            header.extend(
                [f"tactile2_emb_{idx}" for idx in range(tactile2_emb.shape[1])]
            )
        header.extend([f"normalized_tactile_emb_{idx}" for idx in range(normalized_tactile_emb.shape[1])])
        header.extend([f"tcp_pose_obs_{idx}" for idx in range(tcp_pose_obs.shape[1])])
        header.extend([f"gripper_obs_{idx}" for idx in range(gripper_obs.shape[1])])
        header.extend([f"last_action_command_{idx}" for idx in range(last_action_command.shape[1])])
        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header)
            for row_idx in range(len(records)):
                row = [timestamps[row_idx], elapsed[row_idx], sample_counts[row_idx]]
                for sensor_idx in range(self.magnet_used_sensor_count):
                    row.extend(magnet_mean[row_idx, sensor_idx].tolist())
                    row.append(float(magnet_magnitude[row_idx, sensor_idx]))
                row.extend(tactile_emb[row_idx].tolist())
                if has_magnet2:
                    row.append(magnet2_sample_counts[row_idx])
                    for sensor_idx in range(magnet2_sensor_count):
                        row.extend(magnet2_mean[row_idx, sensor_idx].tolist())
                        row.append(float(magnet2_magnitude[row_idx, sensor_idx]))
                    row.extend(tactile2_emb[row_idx].tolist())
                row.extend(normalized_tactile_emb[row_idx].tolist())
                row.extend(tcp_pose_obs[row_idx].tolist())
                row.extend(gripper_obs[row_idx].tolist())
                row.extend(last_action_command[row_idx].tolist())
                writer.writerow(row)

    def _sample_loop(self):
        next_time = time.monotonic()
        period = 1.0 / self.max_fps
        while not self.stop_event.is_set():
            try:
                obs = self._sample_once()
                with self.lock:
                    self.obs_buffer.append(obs)
            except Exception as exc:
                logger.warning(f"FR3 observation sample failed: {exc}")
            next_time += period
            time.sleep(max(0.0, next_time - time.monotonic()))

    def reset(self):
        with self.lock:
            self.obs_buffer.clear()
        if self.move_to_start_on_reset:
            self.move_to_dataset_start()
            with self.lock:
                self.obs_buffer.clear()
        self._reset_magnet_policy_state("environment reset")
        with self.processed_image_lock:
            self.processed_image_saved = False
            self.processed_image_capture_enabled = not self.camera_color_match_dataset_start

    def move_to_dataset_start(self):
        if self.reset_start is None:
            raise RuntimeError("move_to_start_on_reset=True but no reset start pose was loaded")
        target_pose = self.reset_start["pose7d"]
        duration = self.move_to_start_duration
        start_pose_label = "manual start pose" if self.reset_start.get("manual") else "dataset start pose"
        episode_info = (
            f"episode={self.reset_start['episode_idx']}, "
            if self.reset_start.get("episode_idx") is not None
            else ""
        )
        logger.info(
            f"Moving FR3 to {start_pose_label} before policy execution: "
            f"{episode_info}"
            f"target_xyz={target_pose[:3].round(4).tolist()}, "
            f"duration={duration:.2f}s"
        )
        if duration <= 0:
            self._post_tcp_pose(target_pose, target_duration=self.tcp_target_duration)
        else:
            try:
                current_pose = self._get_current_tcp()
                frequency = max(self.move_to_start_frequency, 1.0)
                n_steps = max(2, int(np.ceil(duration * frequency)))
                step_duration = duration / float(n_steps - 1)
                waypoints = _interpolate_pose7d(
                    current_pose,
                    target_pose,
                    np.linspace(0.0, 1.0, n_steps),
                )
                t0 = time.monotonic()
                for i, waypoint in enumerate(waypoints[1:], start=1):
                    precise_wait(t0 + i * step_duration, time_func=time.monotonic)
                    self._post_tcp_pose(
                        waypoint,
                        target_duration=step_duration,
                        timeout=self.tcp_move_timeout,
                    )
                time.sleep(step_duration)
            except Exception as exc:
                logger.warning(
                    "Continuous move-to-start command failed; falling back to one waypoint: "
                    f"{exc}"
                )
                self._post_tcp_pose(target_pose, target_duration=duration)
                time.sleep(duration)
        try:
            pos_error_vec, pos_error, rot_error = self._settle_at_dataset_start(target_pose)
            logger.info(
                f"{start_pose_label.capitalize()} command finished: "
                f"position_error={pos_error:.4f}m, "
                f"xyz_error={np.round(pos_error_vec, 4).tolist()}, "
                f"rot_error={np.rad2deg(rot_error):.2f}deg"
            )
        except RuntimeError:
            raise
        except Exception as exc:
            if self.move_to_start_strict:
                raise RuntimeError(f"Could not verify dataset start pose after move: {exc}") from exc
            logger.warning(f"Could not verify dataset start pose after move: {exc}")

    def _get_pose_error(self, target_pose):
        reached_pose = self._get_current_tcp()
        pos_error_vec = reached_pose[:3] - target_pose[:3]
        pos_error = float(np.linalg.norm(pos_error_vec))
        target_quat_xyzw = np.array([target_pose[4], target_pose[5], target_pose[6], target_pose[3]])
        reached_quat_xyzw = np.array([reached_pose[4], reached_pose[5], reached_pose[6], reached_pose[3]])
        rot_error = float(
            (
                st.Rotation.from_quat(target_quat_xyzw).inv()
                * st.Rotation.from_quat(reached_quat_xyzw)
            ).magnitude()
        )
        return pos_error_vec, pos_error, rot_error

    def _start_pose_is_reached(self, pos_error, rot_error):
        pos_ok = pos_error <= self.move_to_start_pos_tolerance
        rot_ok = (
            self.move_to_start_rot_tolerance <= 0
            or rot_error <= self.move_to_start_rot_tolerance
        )
        return pos_ok and rot_ok

    def _settle_at_dataset_start(self, target_pose):
        pos_error_vec, pos_error, rot_error = self._get_pose_error(target_pose)
        if self._start_pose_is_reached(pos_error, rot_error):
            return pos_error_vec, pos_error, rot_error
        if not self.move_to_start_settle or self.move_to_start_settle_timeout <= 0:
            message = (
                "Dataset start pose is outside tolerance and settling is disabled: "
                f"position_error={pos_error:.4f}m "
                f"(tol={self.move_to_start_pos_tolerance:.4f}m), "
                f"xyz_error={np.round(pos_error_vec, 4).tolist()}, "
                f"rot_error={np.rad2deg(rot_error):.2f}deg "
                f"(tol={np.rad2deg(self.move_to_start_rot_tolerance):.2f}deg)"
            )
            if self.move_to_start_strict:
                raise RuntimeError(message)
            logger.warning(message)
            return pos_error_vec, pos_error, rot_error

        logger.info(
            "Settling at dataset start pose: "
            f"initial_position_error={pos_error:.4f}m, "
            f"initial_xyz_error={np.round(pos_error_vec, 4).tolist()}, "
            f"initial_rot_error={np.rad2deg(rot_error):.2f}deg, "
            f"timeout={self.move_to_start_settle_timeout:.2f}s"
        )
        deadline = time.monotonic() + self.move_to_start_settle_timeout
        frequency = max(self.move_to_start_settle_frequency, 1.0)
        period = 1.0 / frequency
        while time.monotonic() < deadline:
            self._post_tcp_pose(
                target_pose,
                target_duration=period,
                timeout=self.tcp_move_timeout,
            )
            precise_wait(
                min(time.monotonic() + period, deadline),
                time_func=time.monotonic,
            )
            pos_error_vec, pos_error, rot_error = self._get_pose_error(target_pose)
            if self._start_pose_is_reached(pos_error, rot_error):
                logger.info(
                    "Dataset start pose settled: "
                    f"position_error={pos_error:.4f}m, "
                    f"xyz_error={np.round(pos_error_vec, 4).tolist()}, "
                    f"rot_error={np.rad2deg(rot_error):.2f}deg"
                )
                return pos_error_vec, pos_error, rot_error

        message = (
            "Dataset start pose did not settle within tolerance: "
            f"position_error={pos_error:.4f}m "
            f"(tol={self.move_to_start_pos_tolerance:.4f}m), "
            f"xyz_error={np.round(pos_error_vec, 4).tolist()}, "
            f"rot_error={np.rad2deg(rot_error):.2f}deg "
            f"(tol={np.rad2deg(self.move_to_start_rot_tolerance):.2f}deg)"
        )
        if self.move_to_start_strict:
            raise RuntimeError(message)
        logger.warning(message)
        return pos_error_vec, pos_error, rot_error

    def get_obs(self, obs_steps: int = 2, temporal_downsample_ratio: int = 1) -> Dict[str, np.ndarray]:
        with self.lock:
            obs_list = list(self.obs_buffer)
        if not obs_list:
            return {}
        selected = obs_list[::-max(1, temporal_downsample_ratio)][:obs_steps][::-1]
        if not selected:
            return {}
        result = {}
        for key in selected[0].keys():
            values = [obs[key] for obs in selected]
            if len(values) < obs_steps:
                values = [values[0]] * (obs_steps - len(values)) + values
            result[key] = np.stack(values, axis=0)
        return result

    def send_command(self, endpoint: str, data: dict = None, timeout: Optional[float] = None):
        url = f"{self.base_url}{endpoint}"
        timeout = self.http_timeout if timeout is None else timeout
        if "get" in endpoint:
            response = self.state_session.get(url, timeout=timeout)
        else:
            try:
                response = self.command_session.post(url, json=data or {}, timeout=timeout)
            except requests.exceptions.ReadTimeout:
                if "move" in endpoint:
                    return {}
                raise
        response.raise_for_status()
        return response.json()

    def send_gripper_command_direct(self, left_gripper_width_target: float, right_gripper_width_target: float):
        if self.ignore_gripper_commands:
            logger.debug("Ignoring direct gripper command because ignore_gripper_commands=True")
            self.sync_commanded_gripper_state_to_measured()
            return
        if getattr(self, "fixed_gripper_width_m", None) is not None:
            self._send_fixed_gripper_width()
            return
        if self.gripper_action_mode == "binary":
            self._send_gripper_binary(
                float(left_gripper_width_target) >= self.gripper_binary_threshold
            )
        else:
            self._send_gripper_raw(left_gripper_width_target)

    def send_gripper_width_m_direct(self, width_m: float):
        if self.ignore_gripper_commands:
            logger.debug("Ignoring physical gripper command because ignore_gripper_commands=True")
            self.sync_commanded_gripper_state_to_measured()
            return
        if getattr(self, "fixed_gripper_width_m", None) is not None:
            self._send_fixed_gripper_width()
            return
        width_m = float(width_m)
        if not np.isfinite(width_m) or width_m < 0.0 or width_m > self.gripper_stroke:
            raise ValueError(
                "Physical gripper width must be within the gripper stroke: "
                f"0..{self.gripper_stroke:.6f} m, got {width_m}"
            )
        logger.info(f"Commanding startup gripper width={width_m * 1000.0:.3f} mm")
        self._send_gripper_width_m(width_m)

    def _send_gripper_width_m(self, width_m: float):
        width_m = float(width_m)
        self.send_command(
            "/move_gripper/left",
            {
                "width": width_m,
                "velocity": self.gripper_velocity,
                "force_limit": self.grasp_force,
            },
            timeout=self.gripper_http_timeout,
        )
        nominal_width = self._physical_gripper_width_to_nominal_width(width_m)
        self.last_gripper_width_target[0] = self._width_to_raw_gripper(nominal_width)

    def _send_fixed_gripper_width(self):
        width_m = float(self.fixed_gripper_width_m)
        if self.last_fixed_gripper_width_command is not None:
            return
        logger.info(
            "Fixed gripper override enabled: "
            f"commanding physical width={width_m * 1000.0:.3f} mm and ignoring policy gripper output"
        )
        self._send_gripper_width_m(width_m)
        self.last_fixed_gripper_width_command = width_m

    def _send_gripper_raw(self, raw_width):
        raw_width = self._clip_gripper_raw(raw_width, log=True, context="continuous gripper command")
        width_m = self._raw_gripper_to_command_width(raw_width)
        self.send_command(
            "/move_gripper/left",
            {
                "width": width_m,
                "velocity": self.gripper_velocity,
                "force_limit": self.grasp_force,
            },
            timeout=self.gripper_http_timeout,
        )
        self.last_gripper_width_target[0] = float(raw_width)

    def _send_gripper_force(self, raw_width):
        raw_width = self._clip_gripper_raw(raw_width, log=True, context="force gripper command")
        self.send_command(
            "/move_gripper_force/left",
            {"velocity": self.gripper_velocity, "force_limit": self.grasp_force},
            timeout=self.gripper_http_timeout,
        )
        self.last_gripper_width_target[0] = float(raw_width)

    def _raw_gripper_to_binary_open(self, raw_width):
        raw_width = self._clip_gripper_raw(raw_width, log=True, context="binary gripper command")
        if self.last_gripper_binary_open is None:
            return raw_width >= self.gripper_binary_threshold
        if self.last_gripper_binary_open:
            return raw_width >= self.gripper_binary_threshold
        return raw_width >= self.gripper_binary_open_threshold

    def _send_gripper_binary(self, open_gripper):
        open_gripper = bool(open_gripper)
        if open_gripper:
            raw_width = self.gripper_raw_max
            logger.info(
                "Binary gripper command: open "
                f"raw={raw_width:.4f}, width={self.gripper_stroke:.4f}m"
            )
            self.send_command(
                "/move_gripper/left",
                {
                    "width": self.gripper_stroke,
                    "velocity": self.gripper_velocity,
                    "force_limit": self.grasp_force,
                },
                timeout=self.gripper_http_timeout,
            )
        else:
            raw_width = self.gripper_raw_min
            logger.info(
                "Binary gripper command: close "
                f"raw={raw_width:.4f}, force={self.grasp_force:.1f}N"
            )
            if self.use_force_control_for_gripper:
                self.send_command(
                    "/move_gripper_force/left",
                    {"velocity": self.gripper_velocity, "force_limit": self.grasp_force},
                    timeout=self.gripper_http_timeout,
                )
            else:
                self.send_command(
                    "/move_gripper/left",
                    {
                        "width": 0.0,
                        "velocity": self.gripper_velocity,
                        "force_limit": self.grasp_force,
                    },
                    timeout=self.gripper_http_timeout,
                )
        self.last_gripper_width_target[0] = float(raw_width)
        self.last_gripper_binary_open = open_gripper

    def send_gripper_command(self, left_gripper_width_target: float,
            right_gripper_width_target: float, is_bimanual: bool = False):
        if self.ignore_gripper_commands:
            logger.debug("Ignoring policy gripper command because policy commands are disabled")
            return
        if getattr(self, "fixed_gripper_width_m", None) is not None:
            self._send_fixed_gripper_width()
            return
        if self.ignore_policy_gripper_commands:
            logger.debug("Ignoring policy gripper command because policy commands are disabled")
            return
        if self.gripper_action_mode == "binary":
            open_gripper = self._raw_gripper_to_binary_open(left_gripper_width_target)
            if self.last_gripper_binary_open is None or open_gripper != self.last_gripper_binary_open:
                self._send_gripper_binary(open_gripper)
            return

        left_gripper_width_target = self._clip_gripper_raw(
            left_gripper_width_target,
            log=True,
            context="policy gripper command",
        )
        if self.enable_gripper_width_clipping:
            if left_gripper_width_target < self.gripper_width_threshold:
                left_gripper_width_target = self.min_gripper_width

        if abs(self.last_gripper_width_target[0] - left_gripper_width_target) >= self.gripper_control_width_precision:
            if (
                self.use_force_control_for_gripper
                and self.last_gripper_width_target[0] > left_gripper_width_target
            ):
                self._send_gripper_force(left_gripper_width_target)
            else:
                self._send_gripper_raw(left_gripper_width_target)

    def _post_tcp_pose(self, target_tcp, target_duration=None, timeout: Optional[float] = None):
        payload = {"target_tcp": _normalize_pose7d(target_tcp).tolist()}
        if target_duration is not None:
            payload["target_duration"] = float(target_duration)
        self.send_command("/move_tcp/left", payload, timeout=timeout)

    def execute_action(self, action: np.ndarray, use_relative_action: bool = False,
            is_bimanual: bool = False):
        if use_relative_action:
            raise NotImplementedError
        self.last_policy_action_command = np.asarray(action, dtype=np.float32).reshape(-1).copy()
        left_action = action[:8]
        self.send_gripper_command(float(left_action[-2]), float(left_action[-2]))
        if self.ignore_policy_tcp_commands:
            logger.debug("Ignoring policy TCP command because policy commands are disabled")
            return
        target_tcp = pose_6d_to_pose_7d(left_action[:6])
        self._post_tcp_pose(
            target_tcp,
            target_duration=self.tcp_target_duration,
            timeout=self.tcp_move_timeout,
        )

    def get_predicted_action(self, action: np.ndarray, type):
        return None

    def save_exp(self, episode_idx):
        return None

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=2.0)
        self.camera.close()
        if self.magnet2_reader is not None:
            self.magnet2_reader.stop()
            self.magnet2_reader = None
        if self.magnet_reader is not None:
            self.magnet_reader.stop()
            self.magnet_reader = None
        self.state_session.close()
        self.command_session.close()

    def destroy_node(self):
        self.close()
