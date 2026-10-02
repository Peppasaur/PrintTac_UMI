#!/usr/bin/env python3
"""Bridge GELLO joint commands to the RDP/Polymetis TCP controller."""

import argparse
import datetime
import json
import math
import pickle
import re
import shutil
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Dict, Optional

import cv2
import numpy as np
import requests
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32

try:
    from reactive_diffusion_policy.real_world.iphone_udp_camera import IPhoneUDPCamera
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from reactive_diffusion_policy.real_world.iphone_udp_camera import IPhoneUDPCamera


FR3_JOINT_NAMES = tuple(f"fr3_joint{i}" for i in range(1, 8))
MAGNET_SENSOR_COUNT = 5
MAGNET_USED_SENSOR_COUNT = 4
MAGNET_VALUES_PER_SENSOR = 4
MAGNET_SAMPLES_PER_FRAME = 8
MAGNET_FLOAT_PATTERN = re.compile(r"[+-]?\d+\.\d{2}")
MAX_MAGNET_INPUTS = 2
GRIPPER_DEFAULT_STROKES = {
    "franka_hand": 0.08,
    "robotiq_2f": 0.085,
}


def franka_dh_transform(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array(
        [
            [ct, -st, 0.0, a],
            [st * ca, ct * ca, -sa, -d * sa],
            [st * sa, ct * sa, ca, d * ca],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def franka_command_fk(joint_positions: np.ndarray) -> np.ndarray:
    q = np.asarray(joint_positions, dtype=np.float64).reshape(-1)
    if q.size < 7:
        raise ValueError(f"Expected at least 7 joint positions, got {q.size}.")

    # Matches gello_software/experiments/convert_zarr.py.
    a = [0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088]
    d = [0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.107]
    alpha = [0.0, -np.pi / 2.0, np.pi / 2.0, np.pi / 2.0, -np.pi / 2.0, np.pi / 2.0, np.pi / 2.0]

    transform = np.eye(4, dtype=np.float64)
    for index in range(7):
        transform = transform @ franka_dh_transform(a[index], d[index], alpha[index], q[index])
    return transform


def rotmat_from_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm <= 0:
        raise ValueError("Zero quaternion")
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def quat_wxyz_from_rotmat(rot: np.ndarray) -> np.ndarray:
    m = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s

    quat = np.array([qw, qx, qy, qz], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    if quat[0] < 0:
        quat *= -1.0
    return quat


def transform_from_pose7d(pose7d: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose7d, dtype=np.float64).reshape(7)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = pose[:3]
    transform[:3, :3] = rotmat_from_quat_wxyz(pose[3:7])
    return transform


def pose7d_from_transform(transform: np.ndarray) -> np.ndarray:
    pose = np.empty(7, dtype=np.float64)
    pose[:3] = transform[:3, 3]
    pose[3:7] = quat_wxyz_from_rotmat(transform[:3, :3])
    return pose


def rotvec_from_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    rot = rotmat_from_quat_wxyz(quat_wxyz)
    trace = float(np.trace(rot))
    cos_angle = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    angle = float(np.arccos(cos_angle))
    if angle < 1e-12:
        return np.zeros(3, dtype=np.float32)
    axis = np.array(
        [
            rot[2, 1] - rot[1, 2],
            rot[0, 2] - rot[2, 0],
            rot[1, 0] - rot[0, 1],
        ],
        dtype=np.float64,
    )
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:
        return np.zeros(3, dtype=np.float32)
    axis = axis / axis_norm
    return (axis * angle).astype(np.float32)


def action7_from_pose7d_gripper_raw(pose7d: np.ndarray, gripper_raw: float) -> np.ndarray:
    pose = np.asarray(pose7d, dtype=np.float64).reshape(7)
    return np.concatenate(
        [
            pose[:3].astype(np.float32),
            rotvec_from_quat_wxyz(pose[3:7]),
            np.array([float(gripper_raw)], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def normalize_record_magnet_ports(ports) -> tuple[str, ...]:
    if isinstance(ports, str):
        ports = [ports]
    normalized = tuple(str(port).strip() for port in ports if str(port).strip())
    if not normalized:
        raise ValueError("At least one --record-magnet-port is required")
    if len(normalized) > MAX_MAGNET_INPUTS:
        raise ValueError(
            f"At most {MAX_MAGNET_INPUTS} magnetometer ports are supported, "
            f"got {len(normalized)}"
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError("--record-magnet-port values must be unique")
    return normalized


def magnet_frame_keys(input_index: int) -> tuple[str, str, str]:
    if input_index < 0 or input_index >= MAX_MAGNET_INPUTS:
        raise ValueError(f"Unsupported magnetometer input index {input_index}")
    prefix = "magnet" if input_index == 0 else f"magnet{input_index + 1}"
    return (
        f"{prefix}_xyz",
        f"{prefix}_timestamp_ns",
        f"{prefix}_sample_count",
    )


def collect_magnet_frame_fields(readers) -> Dict[str, np.ndarray]:
    fields = {}
    for input_index, reader in enumerate(readers):
        xyz_key, timestamp_key, count_key = magnet_frame_keys(input_index)
        recent = reader.get_recent_samples()
        fields[xyz_key] = recent["magnet_xyz"]
        fields[timestamp_key] = recent["magnet_timestamp_ns"]
        fields[count_key] = recent["magnet_sample_count"]
    return fields


class MagnetometerReader:
    """Background serial reader matching the existing GELLO data format."""

    def __init__(
        self,
        port: str,
        baudrate: int,
        samples_per_frame: int = MAGNET_SAMPLES_PER_FRAME,
        buffer_size: int = 1024,
        idle_sleep: float = 0.0005,
        subtract_baseline: bool = True,
    ):
        self.port = port
        self.baudrate = int(baudrate)
        self.samples_per_frame = int(samples_per_frame)
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
        self._thread = threading.Thread(target=self._read_loop, name="magnet-reader", daemon=True)
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
    def sample_count(self) -> int:
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
                print(f"[WARN] Magnetometer read error: {exc}")
                self._running = False

    def _handle_line(self, line: str):
        values = self._parse_sensor_values(line)
        if len(values) != MAGNET_SENSOR_COUNT * MAGNET_VALUES_PER_SENSOR:
            return

        xyz_values = []
        for sensor_idx in range(MAGNET_USED_SENSOR_COUNT):
            start = sensor_idx * MAGNET_VALUES_PER_SENSOR
            _, x, y, z = values[start:start + MAGNET_VALUES_PER_SENSOR]
            xyz_values.append([x, y, z])
        xyz = np.asarray(xyz_values, dtype=np.float32)

        with self._lock:
            if self.subtract_baseline:
                if self._baseline_xyz is None:
                    self._baseline_xyz = xyz.copy()
                xyz = xyz - self._baseline_xyz
            self._samples.append(
                {
                    "timestamp_ns": time.time_ns(),
                    "xyz": xyz.astype(np.float32),
                }
            )
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

    def get_recent_samples(self) -> Dict[str, np.ndarray]:
        with self._lock:
            samples = list(self._samples)[-self.samples_per_frame:]

        xyz = np.full(
            (self.samples_per_frame, MAGNET_USED_SENSOR_COUNT, 3),
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


def camera_candidates(source: str, backend: str = "auto"):
    if source != "auto":
        try:
            return [int(source)]
        except ValueError:
            return [source]
    patterns = (
        (
            "/dev/v4l/by-id/*ZED*video-index0",
            "/dev/v4l/by-id/*ZED*video-index*",
            "/dev/v4l/by-path/*video-index0",
            "/dev/video*",
        )
        if str(backend).lower() in ("zed_v4l", "zed")
        else (
            "/dev/v4l/by-path/*:1.3-video-index0",
            "/dev/v4l/by-id/*RealSense*video-index2",
            "/dev/v4l/by-id/*RealSense*video-index0",
            "/dev/v4l/by-path/*video-index2",
            "/dev/v4l/by-path/*video-index0",
            "/dev/video*",
        )
    )
    import glob

    result = []
    seen = set()
    for pattern in patterns:
        for candidate in sorted(glob.glob(pattern)):
            if candidate not in seen:
                result.append(candidate)
                seen.add(candidate)
    return result or [0]


def bgr_color_score(frame):
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        return 0.0
    frame_f = frame.astype(np.float32)
    return float(
        max(
            np.mean(np.abs(frame_f[..., 0] - frame_f[..., 1])),
            np.mean(np.abs(frame_f[..., 1] - frame_f[..., 2])),
            np.mean(np.abs(frame_f[..., 0] - frame_f[..., 2])),
        )
    )


class RecordingCamera:
    def __init__(
        self,
        source: str = "auto",
        backend: str = "opencv",
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0,
        image_height: int = 224,
        image_width: int = 224,
        zed_view: str = "left",
        flip: bool = False,
        require_color: bool = True,
        color_threshold: float = 1.5,
        iphone_bind_host: str = "0.0.0.0",
        iphone_video_port: int = 5560,
        iphone_combined_port: int = 5558,
        iphone_phone_ip: str = "",
        iphone_registration_port: int = 5559,
        iphone_startup_timeout: float = 5.0,
        iphone_read_timeout: float = 1.0,
        iphone_hello_interval: float = 2.0,
    ):
        self.source_arg = str(source)
        self.backend = str(backend).lower()
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.zed_view = str(zed_view).lower()
        self.flip = bool(flip)
        self.require_color = bool(require_color)
        self.color_threshold = float(color_threshold)
        self.iphone_bind_host = str(iphone_bind_host)
        self.iphone_video_port = int(iphone_video_port)
        self.iphone_combined_port = int(iphone_combined_port)
        self.iphone_phone_ip = str(iphone_phone_ip or "")
        if (
            self.backend == "iphone"
            and not self.iphone_phone_ip
            and self.source_arg in ("", "auto", "none", "None")
        ):
            self.iphone_phone_ip = "172.20.10.1"
        self.iphone_registration_port = int(iphone_registration_port)
        self.iphone_startup_timeout = float(iphone_startup_timeout)
        self.iphone_read_timeout = float(iphone_read_timeout)
        self.iphone_hello_interval = float(iphone_hello_interval)
        self.cap = None
        self.iphone_camera = None
        self.source = None

    def open(self):
        if self.backend == "iphone":
            self.iphone_camera = IPhoneUDPCamera(
                source=self.source_arg,
                bind_host=self.iphone_bind_host,
                video_port=self.iphone_video_port,
                combined_port=self.iphone_combined_port,
                phone_ip=self.iphone_phone_ip,
                registration_port=self.iphone_registration_port,
                startup_timeout=self.iphone_startup_timeout,
                read_timeout=self.iphone_read_timeout,
                hello_interval=self.iphone_hello_interval,
                require_color=self.require_color,
                color_threshold=self.color_threshold,
            )
            first_frame = self.iphone_camera.open()
            self.source = self.iphone_phone_ip or self.source_arg
            print(
                "[INFO] iPhone recording camera opened: "
                f"phone={self.source}, "
                f"bind={self.iphone_bind_host}:{self.iphone_video_port}, "
                f"shape={first_frame.shape}, "
                f"color_score={bgr_color_score(first_frame):.3f}"
            )
            return

        errors = []
        for source in camera_candidates(self.source_arg, backend=self.backend):
            cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
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
                    last_score = bgr_color_score(frame)
                    if (
                        frame.ndim == 3
                        and frame.shape[2] == 3
                        and (not self.require_color or last_score >= self.color_threshold)
                    ):
                        self.cap = cap
                        self.source = source
                        print(
                            f"[INFO] Recording camera opened: {source}, "
                            f"shape={frame.shape}, color_score={last_score:.3f}"
                        )
                        return
                time.sleep(0.05)
            cap.release()
            errors.append(f"{source}: shape={last_shape}, color_score={last_score}")
        raise RuntimeError("Could not open recording camera. " + "; ".join(errors))

    def read_rgb(self):
        if self.backend == "iphone":
            if self.iphone_camera is None:
                raise RuntimeError("iPhone recording camera is not open")
            frame_bgr = self.iphone_camera.read()
        else:
            if self.cap is None:
                raise RuntimeError("Recording camera is not open")
            ok, frame_bgr = self.cap.read()
            if not ok or frame_bgr is None:
                raise RuntimeError(
                    f"Failed to read frame from recording camera source: {self.source}"
                )
            frame_bgr = self._select_view(frame_bgr)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.flip:
            rgb = cv2.rotate(rgb, cv2.ROTATE_180)
        return rgb.astype(np.uint8)

    def _select_view(self, frame_bgr):
        if self.backend != "zed_v4l" or self.zed_view == "full":
            return frame_bgr
        h, w = frame_bgr.shape[:2]
        if w < int(2.5 * h):
            return frame_bgr
        half_w = w // 2
        if self.zed_view == "left":
            return frame_bgr[:, :half_w]
        if self.zed_view == "right":
            return frame_bgr[:, half_w:2 * half_w]
        return frame_bgr

    def close(self):
        if self.iphone_camera is not None:
            self.iphone_camera.close()
            self.iphone_camera = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class RawEpisodeRecorder:
    def __init__(
        self,
        args: argparse.Namespace,
        get_command_state: Callable[[], Optional[Dict[str, np.ndarray]]],
        get_actual_tcp: Callable[[], np.ndarray],
    ):
        self.args = args
        self.get_command_state = get_command_state
        self.get_actual_tcp = get_actual_tcp
        self.output_dir = Path(args.record_output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.camera = None
        self.magnet_reader = None
        self.magnet_readers = []
        self.thread = None
        self.keyboard_thread = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.is_recording = False
        self.episode_dir: Optional[Path] = None
        self.frame_idx = 0
        self.episode_count = 0
        self.last_error_log_time = 0.0
        self.status_text = "IDLE  [S]tart"
        self.status_color = "idle"

    def start(self):
        if not self.args.record_no_camera:
            self.camera = RecordingCamera(
                source=self.args.record_camera_source,
                backend=self.args.record_camera_backend,
                width=self.args.record_camera_width,
                height=self.args.record_camera_height,
                fps=self.args.record_camera_fps,
                image_height=self.args.record_image_height,
                image_width=self.args.record_image_width,
                zed_view=self.args.record_zed_view,
                flip=self.args.record_camera_flip,
                require_color=not self.args.record_allow_gray_camera,
                color_threshold=self.args.record_camera_color_threshold,
                iphone_bind_host=self.args.record_iphone_bind_host,
                iphone_video_port=self.args.record_iphone_video_port,
                iphone_combined_port=self.args.record_iphone_combined_port,
                iphone_phone_ip=self.args.record_iphone_phone_ip,
                iphone_registration_port=self.args.record_iphone_registration_port,
                iphone_startup_timeout=self.args.record_iphone_startup_timeout,
                iphone_read_timeout=self.args.record_iphone_read_timeout,
                iphone_hello_interval=self.args.record_iphone_hello_interval,
            )
            self.camera.open()
        if not self.args.record_no_magnet:
            ports = normalize_record_magnet_ports(self.args.record_magnet_port)
            try:
                for input_index, port in enumerate(ports):
                    reader = MagnetometerReader(
                        port=port,
                        baudrate=self.args.record_magnet_baudrate,
                        samples_per_frame=self.args.record_magnet_samples_per_frame,
                        subtract_baseline=not self.args.record_magnet_no_baseline,
                    )
                    reader.start()
                    self.magnet_readers.append(reader)
                    print(
                        f"[INFO] Recording magnetometer {input_index + 1} from {port} "
                        f"at {self.args.record_magnet_baudrate} baud"
                    )
            except Exception:
                for reader in self.magnet_readers:
                    reader.stop()
                self.magnet_readers.clear()
                if self.camera is not None:
                    self.camera.close()
                    self.camera = None
                raise
            self.magnet_reader = self.magnet_readers[0]
        self.thread = threading.Thread(target=self._record_loop, name="raw-episode-recorder", daemon=True)
        self.thread.start()
        if self.args.record_keyboard:
            self.keyboard_thread = threading.Thread(
                target=self._control_ui_loop,
                name="record-control-ui",
                daemon=True,
            )
            self.keyboard_thread.start()
        if self.args.record_on_start:
            self.start_episode()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self.camera is not None:
            self.camera.close()
        for reader in self.magnet_readers:
            reader.stop()
        self.magnet_readers.clear()
        self.magnet_reader = None

    def start_episode(self):
        with self.lock:
            if self.is_recording:
                return
            name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.episode_dir = self.output_dir / name
            self.episode_dir.mkdir(parents=True, exist_ok=True)
            if self.args.record_magnet_reset_baseline_on_start:
                for reader in self.magnet_readers:
                    reader.reset_baseline()
            self.frame_idx = 0
            self.episode_count += 1
            self.is_recording = True
            self.status_text = f"REC ep{self.episode_count}  [Q]save [D]iscard"
            self.status_color = "recording"
        print(f"[INFO] Recording episode {self.episode_count}: {self.episode_dir}")

    def stop_episode(self):
        with self.lock:
            if not self.is_recording:
                return
            episode_dir = self.episode_dir
            frame_idx = self.frame_idx
            self.is_recording = False
            self.episode_dir = None
            self.frame_idx = 0
            self.status_text = f"SAVED ep{self.episode_count}: {frame_idx}f  [S]tart"
            self.status_color = "saved"
        print(f"[INFO] Saved episode: {episode_dir} ({frame_idx} frames)")

    def discard_episode(self):
        with self.lock:
            episode_dir = self.episode_dir
            self.is_recording = False
            self.episode_dir = None
            self.frame_idx = 0
            self.status_text = f"DISCARDED ep{self.episode_count}  [S]tart"
            self.status_color = "discarded"
        if episode_dir is not None and episode_dir.exists():
            shutil.rmtree(episode_dir)
            print(f"[INFO] Discarded episode: {episode_dir}")

    def _control_ui_loop(self):
        try:
            self._tk_control_window_loop()
        except Exception as exc:
            print(f"[WARN] Recording control window unavailable ({exc}); falling back to terminal input.")
            self._keyboard_loop()

    def _tk_control_window_loop(self):
        import tkinter as tk

        colors = {
            "idle": "#808080",
            "recording": "#00a000",
            "discarded": "#c00000",
            "saved": "#0064c8",
        }
        root = tk.Tk()
        root.title("GELLO Polymetis Recorder")
        root.geometry("420x220")
        root.resizable(False, False)
        label = tk.Label(
            root,
            text="IDLE  [S]tart",
            fg="white",
            bg=colors["idle"],
            font=("Arial", 24, "bold"),
            justify="center",
        )
        label.pack(fill="both", expand=True)

        def update_status():
            with self.lock:
                text = self.status_text
                color = self.status_color
                if self.is_recording:
                    text = f"REC ep{self.episode_count}: {self.frame_idx}f\n[Q]save  [D]iscard"
                    color = "recording"
            label.configure(text=text, bg=colors.get(color, colors["idle"]))

        def on_key(event):
            key = (event.char or "").lower()
            if key == "s":
                self.start_episode()
            elif key == "q":
                self.stop_episode()
            elif key == "d":
                self.discard_episode()
            update_status()

        def on_close():
            root.withdraw()
            print("[INFO] Recording control window hidden; recording thread is still running.")

        root.bind("<KeyPress>", on_key)
        root.protocol("WM_DELETE_WINDOW", on_close)
        while not self.stop_event.is_set():
            update_status()
            root.update()
            time.sleep(0.05)
        root.destroy()

    def _keyboard_loop(self):
        print("[INFO] Recording controls: type 's'+Enter to start, 'q'+Enter to save, 'd'+Enter to discard")
        while not self.stop_event.is_set():
            try:
                line = input().strip().lower()
            except EOFError:
                return
            if line == "s":
                self.start_episode()
            elif line == "q":
                self.stop_episode()
            elif line == "d":
                self.discard_episode()

    def _record_loop(self):
        period = 1.0 / max(float(self.args.record_rate), 1e-6)
        while not self.stop_event.is_set():
            start = time.monotonic()
            with self.lock:
                active = self.is_recording
            if active:
                try:
                    self._record_frame()
                except Exception as exc:
                    now = time.monotonic()
                    if now - self.last_error_log_time > 1.0:
                        print(f"[WARN] Recording frame failed: {exc}")
                        self.last_error_log_time = now
            elapsed = time.monotonic() - start
            time.sleep(max(0.0, period - elapsed))

    def _record_frame(self):
        command_state = self.get_command_state()
        if command_state is None:
            return
        actual_tcp = self.get_actual_tcp()
        actual_action = action7_from_pose7d_gripper_raw(
            actual_tcp,
            float(command_state["gripper_raw"][0]),
        )
        command_action = action7_from_pose7d_gripper_raw(
            command_state["target_tcp"],
            float(command_state["gripper_raw"][0]),
        )
        frame = {
            "timestamp": np.array([time.time()], dtype=np.float64),
            "collection_time_ns": np.array([time.time_ns()], dtype=np.int64),
            "robot0_eef_pos": actual_action[:3].astype(np.float32),
            "robot0_eef_rot_axis_angle": actual_action[3:6].astype(np.float32),
            "robot0_gripper_width": command_state["gripper_raw"].astype(np.float32),
            "action": command_action.astype(np.float32),
            "command_tcp_pose_wxyz": command_state["target_tcp"].astype(np.float32),
            "gello_command": command_state["gello_command"].astype(np.float32),
            "gello_gripper_input_raw": command_state["gripper_input_raw"].astype(np.float32),
        }
        if self.camera is not None:
            frame["camera0_rgb"] = self.camera.read_rgb()
        if self.magnet_readers:
            frame.update(collect_magnet_frame_fields(self.magnet_readers))
        else:
            frame["magnet_xyz"] = np.full(
                (
                    self.args.record_magnet_samples_per_frame,
                    MAGNET_USED_SENSOR_COUNT,
                    3,
                ),
                np.nan,
                dtype=np.float32,
            )
            frame["magnet_timestamp_ns"] = np.zeros(
                self.args.record_magnet_samples_per_frame,
                dtype=np.int64,
            )
            frame["magnet_sample_count"] = np.zeros(1, dtype=np.int32)

        with self.lock:
            if not self.is_recording or self.episode_dir is None:
                return
            path = self.episode_dir / f"frame_{self.frame_idx:06d}.pkl"
            self.frame_idx += 1
        with open(path, "wb") as f:
            pickle.dump(frame, f, protocol=pickle.HIGHEST_PROTOCOL)


class TeleopRecordingControlServer:
    """Expose the recorder's existing start/save/discard operations to the experiment UI."""

    def __init__(self, recorder: RawEpisodeRecorder, host: str, port: int):
        self.recorder = recorder

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args: object) -> None:
                return

            def _status_payload(self) -> dict:
                with recorder.lock:
                    episode_dir = recorder.episode_dir
                    return {
                        "ok": True,
                        "state": "recording" if recorder.is_recording else "idle",
                        "episode_count": recorder.episode_count,
                        "frame_count": recorder.frame_idx,
                        "episode_dir": str(episode_dir) if episode_dir is not None else "",
                    }

            def _write_json(self, status: int, payload: dict) -> None:
                body = json.dumps(payload, sort_keys=True).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                if self.path.split("?", 1)[0] == "/status":
                    self._write_json(200, self._status_payload())
                    return
                self._write_json(404, {"ok": False, "error": "unknown endpoint"})

            def do_POST(self) -> None:
                endpoint = self.path.split("?", 1)[0]
                if endpoint == "/start":
                    recorder.start_episode()
                elif endpoint == "/stop":
                    recorder.stop_episode()
                elif endpoint == "/discard":
                    recorder.discard_episode()
                else:
                    self._write_json(404, {"ok": False, "error": "unknown endpoint"})
                    return
                payload = self._status_payload()
                payload["action"] = endpoint[1:]
                self._write_json(200, payload)

        class Server(ThreadingHTTPServer):
            allow_reuse_address = True
            daemon_threads = True

        self.server = Server((host, port), Handler)
        self.host = str(self.server.server_address[0])
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            name="teleop-recording-control",
            daemon=True,
        )

    @property
    def port(self) -> int:
        return int(self.server.server_address[1])

    def start(self) -> None:
        self.thread.start()
        print(f"[INFO] Teleop recording control listening on http://{self.host}:{self.port}")

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)


class GelloPolymetisTcpTeleop(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("gello_polymetis_tcp_teleop")
        self.args = args
        self.session = requests.Session()
        self.gripper_session = requests.Session()
        self.record_session = requests.Session()
        self.latest_q: Optional[np.ndarray] = None
        self.latest_msg_time: Optional[float] = None
        self.correction: Optional[np.ndarray] = np.eye(4, dtype=np.float64) if args.alignment == "absolute" else None
        self.last_log_time = 0.0
        self.sent_count = 0
        self.startup_until: Optional[float] = None
        self.latest_gripper_width: Optional[float] = None
        self.latest_gripper_raw = np.array([args.default_gripper_raw], dtype=np.float32)
        self.latest_gripper_input_raw = np.array([args.default_gripper_raw], dtype=np.float32)
        self.latest_gripper_msg_time: Optional[float] = None
        self.last_sent_gripper_width: Optional[float] = None
        self.last_gripper_send_time = 0.0
        self.last_gripper_log_time = 0.0
        self.gripper_lock = threading.Lock()
        self.gripper_request_in_flight = False
        self.gripper_action_type = None
        self.gripper_action_client = None
        self.gripper_grasp_action_type = None
        self.gripper_grasp_action_client = None
        self.gripper_action_goal_count = 0
        self.gripper_action_goal_handle = None
        self.gripper_action_active = False
        self.gripper_action_canceling = False
        self.gripper_action_goal_pending_response = False
        self.gripper_action_active_width: Optional[float] = None
        self.gripper_action_pending_width: Optional[float] = None
        self.gripper_action_active_kind: Optional[str] = None
        self.gripper_action_active_is_closing = False
        self.current_gripper_width: Optional[float] = None
        self.recording_lock = threading.Lock()
        self.latest_target_tcp: Optional[np.ndarray] = None
        self.latest_recording_q: Optional[np.ndarray] = None
        self.recorder: Optional[RawEpisodeRecorder] = None
        self.recording_control_server: Optional[TeleopRecordingControlServer] = None

        if not args.disable_gripper and args.gripper_command_transport == "ros2_action":
            try:
                from franka_msgs.action import Move as FrankaGripperMove
                from franka_msgs.action import Grasp as FrankaGripperGrasp
            except ImportError as exc:
                raise RuntimeError(
                    "--gripper-command-transport ros2_action requires franka_msgs "
                    "in the Python environment."
                ) from exc
            self.gripper_action_type = FrankaGripperMove
            self.gripper_action_client = ActionClient(
                self,
                FrankaGripperMove,
                args.gripper_move_action,
            )
            self.gripper_grasp_action_type = FrankaGripperGrasp
            self.gripper_grasp_action_client = ActionClient(
                self,
                FrankaGripperGrasp,
                args.gripper_grasp_action,
            )

        self.create_subscription(JointState, args.gello_topic, self._joint_state_cb, 1)
        if not args.disable_gripper:
            self.create_subscription(Float32, args.gripper_topic, self._gripper_cb, 1)
            if args.gripper_command_transport == "ros2_action":
                self.create_subscription(
                    JointState,
                    args.gripper_joint_states_topic,
                    self._gripper_joint_state_cb,
                    1,
                )
        self.create_timer(1.0 / args.rate, self._timer_cb)
        self.get_logger().info(
            f"Listening to {args.gello_topic}; sending TCP targets to {args.robot_server}/move_tcp/left "
            f"at {args.rate:g}Hz. dry_run={args.dry_run}"
        )
        if not args.disable_gripper:
            if args.gripper_command_transport == "http":
                gripper_target = f"{args.robot_server}/move_gripper/left"
            elif args.gripper_command_transport == "ros2_action":
                gripper_target = f"ROS2 action {args.gripper_move_action}"
            else:
                gripper_target = "none (subscribing/recording only)"
            self.get_logger().info(
                f"Listening to {args.gripper_topic}; sending gripper targets to "
                f"{gripper_target}"
            )
            self.get_logger().info(
                "Gripper control: "
                f"type={args.gripper_type}, stroke={args.gripper_stroke:.4f}m, "
                f"transport={args.gripper_command_transport}, "
                f"action_mode={args.gripper_action_mode}, action_step={args.gripper_action_step:.4f}m, "
                f"close_action={args.gripper_close_action}, grasp_force={args.gripper_grasp_force:.1f}N, "
                f"mode={args.gripper_control_mode}, input_mode={args.gripper_input_mode}, "
                f"idle_raw={args.gripper_idle_raw:.4f}, pressed_raw={args.gripper_pressed_raw:.4f}, "
                f"press_threshold={args.gripper_press_threshold:.4f}, "
                f"press_active={args.gripper_press_active}"
            )
            if self.gripper_action_client is not None:
                if self.gripper_action_client.wait_for_server(
                    timeout_sec=max(0.0, float(args.gripper_action_server_timeout))
                ):
                    self.get_logger().info(f"Franka Hand action server ready: {args.gripper_move_action}")
                else:
                    self.get_logger().warning(
                        "Franka Hand action server is not ready yet: "
                        f"{args.gripper_move_action}"
                    )
            if self.gripper_grasp_action_client is not None and args.gripper_close_action == "grasp":
                if self.gripper_grasp_action_client.wait_for_server(
                    timeout_sec=max(0.0, float(args.gripper_action_server_timeout))
                ):
                    self.get_logger().info(f"Franka Hand grasp action server ready: {args.gripper_grasp_action}")
                else:
                    self.get_logger().warning(
                        "Franka Hand grasp action server is not ready yet: "
                        f"{args.gripper_grasp_action}"
                    )
        if args.alignment == "absolute":
            self.get_logger().warning(
                "Using absolute GELLO FK poses with no startup alignment. "
                "The first command may move the robot to the GELLO absolute pose."
            )
        if args.record_output_dir:
            self.recorder = RawEpisodeRecorder(
                args=args,
                get_command_state=self._get_recording_command_state,
                get_actual_tcp=self._get_recording_current_tcp,
            )
            self.recorder.start()
            if args.record_control_port:
                try:
                    self.recording_control_server = TeleopRecordingControlServer(
                        self.recorder,
                        args.record_control_host,
                        args.record_control_port,
                    )
                    self.recording_control_server.start()
                except OSError as exc:
                    self.get_logger().error(
                        "Teleop recording control is unavailable at "
                        f"{args.record_control_host}:{args.record_control_port}: {exc}"
                    )

    def _joint_state_cb(self, msg: JointState):
        if len(msg.position) < 7:
            self.get_logger().warning(f"Ignoring JointState with {len(msg.position)} positions")
            return

        if msg.name and all(name in msg.name for name in FR3_JOINT_NAMES):
            name_to_idx = {name: i for i, name in enumerate(msg.name)}
            q = np.array([msg.position[name_to_idx[name]] for name in FR3_JOINT_NAMES], dtype=np.float64)
        else:
            q = np.asarray(msg.position[:7], dtype=np.float64)

        if not np.all(np.isfinite(q)):
            self.get_logger().warning("Ignoring non-finite GELLO joint command")
            return

        self.latest_q = q
        self.latest_msg_time = time.monotonic()

    def _gripper_cb(self, msg: Float32):
        value = float(msg.data)
        if self.args.gripper_input_mode == "width":
            input_raw = float(np.clip(value / max(self.args.gripper_stroke, 1e-9), 0.0, 1.0))
        else:
            ratio = value / 100.0 if abs(value) > 1.0 else value
            ratio = float(np.clip(ratio, 0.0, 1.0))
            if self.args.gripper_input_mode == "close_ratio":
                ratio = 1.0 - ratio
            input_raw = ratio
        raw = float(np.clip(self._map_gripper_control_raw(input_raw), 0.0, 1.0))
        width = raw * self.args.gripper_stroke
        self.latest_gripper_width = float(np.clip(width, 0.0, self.args.gripper_stroke))
        self.latest_gripper_raw = np.array([float(raw)], dtype=np.float32)
        self.latest_gripper_input_raw = np.array([float(input_raw)], dtype=np.float32)
        self.latest_gripper_msg_time = time.monotonic()

    def _gripper_joint_state_cb(self, msg: JointState):
        if len(msg.position) == 0:
            return
        width = 2.0 * float(msg.position[0])
        if np.isfinite(width):
            self.current_gripper_width = float(np.clip(width, 0.0, self.args.gripper_stroke))

    def _map_gripper_control_raw(self, input_raw: float) -> float:
        input_raw = float(np.clip(input_raw, 0.0, 1.0))
        if self.args.gripper_control_mode == "direct":
            return input_raw
        if self.args.gripper_control_mode == "press_open":
            if self.args.gripper_press_active == "above":
                is_pressed = input_raw >= self.args.gripper_press_threshold
            else:
                is_pressed = input_raw <= self.args.gripper_press_threshold
            return self.args.gripper_pressed_raw if is_pressed else self.args.gripper_idle_raw
        raise ValueError(f"Unsupported gripper_control_mode: {self.args.gripper_control_mode}")

    def _get_current_tcp(self) -> np.ndarray:
        response = self.session.get(f"{self.args.robot_server}/get_current_tcp/left", timeout=self.args.timeout)
        response.raise_for_status()
        return np.asarray(response.json(), dtype=np.float64)

    def _get_recording_current_tcp(self) -> np.ndarray:
        response = self.record_session.get(
            f"{self.args.robot_server}/get_current_tcp/left",
            timeout=self.args.record_robot_timeout,
        )
        response.raise_for_status()
        return np.asarray(response.json(), dtype=np.float64)

    def _post_tcp(self, pose7d: np.ndarray, target_duration: float):
        if self.args.dry_run:
            return
        payload = {
            "target_tcp": [float(x) for x in pose7d],
            "target_duration": float(target_duration),
        }
        response = self.session.post(
            f"{self.args.robot_server}/move_tcp/left",
            json=payload,
            timeout=self.args.timeout,
        )
        response.raise_for_status()

    def _post_gripper(self, width: float):
        if self.args.dry_run:
            return
        width = float(np.clip(width, 0.0, self.args.gripper_stroke))
        payload = {
            "width": width,
            "velocity": float(self.args.gripper_velocity),
            "force_limit": float(self.args.gripper_force),
        }
        response = self.gripper_session.post(
            f"{self.args.robot_server}/move_gripper/left",
            json=payload,
            timeout=self.args.gripper_timeout,
        )
        response.raise_for_status()

    def _try_send_gripper_action(self, width: float) -> bool:
        if self.args.dry_run:
            return True
        if self.gripper_action_client is None or self.gripper_action_type is None:
            return False

        width = float(np.clip(width, 0.0, self.args.gripper_stroke))
        if self.args.gripper_action_mode == "preempt":
            return self._try_send_preemptive_gripper_action(width)
        if self.args.gripper_action_mode == "servo":
            return self._try_send_servo_gripper_action(width)
        return self._send_gripper_action_goal(width)

    def _try_send_servo_gripper_action(self, width: float) -> bool:
        self.gripper_action_pending_width = width
        if (
            self.gripper_action_active
            or self.gripper_action_canceling
            or self.gripper_action_goal_pending_response
        ):
            self.last_sent_gripper_width = width
            return True
        if (
            self.current_gripper_width is None
            and self.gripper_action_active_width is None
            and self.last_sent_gripper_width is None
        ):
            now = time.monotonic()
            if now - self.last_gripper_log_time > 1.0:
                self.get_logger().warning(
                    "Waiting for Franka Hand joint state before servo gripper control: "
                    f"{self.args.gripper_joint_states_topic}"
                )
                self.last_gripper_log_time = now
        sent = self._send_pending_gripper_action()
        if sent:
            self.last_sent_gripper_width = width
        return sent

    def _try_send_preemptive_gripper_action(self, width: float) -> bool:
        if self.gripper_action_goal_pending_response:
            self.gripper_action_pending_width = width
            self.last_sent_gripper_width = width
            return True

        if not self.gripper_action_active and not self.gripper_action_canceling:
            return self._send_gripper_action_goal(width)

        self.gripper_action_pending_width = width
        should_cancel = (
            self.gripper_action_active
            and not self.gripper_action_canceling
            and self.gripper_action_goal_handle is not None
            and (
                self.gripper_action_active_width is None
                or abs(width - self.gripper_action_active_width) >= self.args.gripper_min_delta
            )
        )
        if should_cancel:
            cancel_future = self.gripper_action_goal_handle.cancel_goal_async()
            cancel_future.add_done_callback(self._gripper_action_cancel_cb)
            self.gripper_action_canceling = True
        self.last_sent_gripper_width = width
        return True

    def _send_gripper_action_goal(self, width: float, force_action_kind: Optional[str] = None) -> bool:
        width = float(np.clip(width, 0.0, self.args.gripper_stroke))
        is_closing = self._is_gripper_closing(width)
        action_kind, action_display_name, action_client, action_type = self._select_gripper_action(
            width,
            force_action_kind=force_action_kind,
            is_closing=is_closing,
        )
        if action_client is None or action_type is None or not action_client.server_is_ready():
            now = time.monotonic()
            if now - self.last_gripper_log_time > 1.0:
                self.get_logger().warning(f"Franka Hand action server is not ready: {action_display_name}")
                self.last_gripper_log_time = now
            return False
        goal_msg = action_type.Goal()
        goal_msg.width = width
        goal_msg.speed = float(self.args.gripper_velocity)
        if action_kind == "grasp":
            goal_msg.force = float(self.args.gripper_grasp_force)
            goal_msg.epsilon.inner = float(self.args.gripper_grasp_epsilon_inner)
            goal_msg.epsilon.outer = float(self.args.gripper_grasp_epsilon_outer)
        future = action_client.send_goal_async(goal_msg)
        future.add_done_callback(
            lambda done, goal_width=width, goal_kind=action_kind, goal_is_closing=is_closing: (
                self._gripper_action_goal_response_cb(done, goal_width, goal_kind, goal_is_closing)
            )
        )
        self.gripper_action_goal_count += 1
        self.last_sent_gripper_width = width
        if self.args.gripper_action_mode in ("preempt", "servo"):
            self.gripper_action_goal_pending_response = True
        return True

    def _select_gripper_action(self, width: float, force_action_kind: Optional[str] = None,
            is_closing: Optional[bool] = None):
        if force_action_kind == "grasp":
            return "grasp", self.args.gripper_grasp_action, self.gripper_grasp_action_client, self.gripper_grasp_action_type
        if force_action_kind == "move":
            return "move", self.args.gripper_move_action, self.gripper_action_client, self.gripper_action_type
        if self.args.gripper_close_action == "grasp":
            closing = self._is_gripper_closing(width) if is_closing is None else is_closing
            if closing:
                return "grasp", self.args.gripper_grasp_action, self.gripper_grasp_action_client, self.gripper_grasp_action_type
        return "move", self.args.gripper_move_action, self.gripper_action_client, self.gripper_action_type

    def _is_gripper_closing(self, width: float) -> bool:
        if self.current_gripper_width is not None:
            reference_width = self.current_gripper_width
        elif self.gripper_action_active_width is not None:
            reference_width = self.gripper_action_active_width
        elif self.last_sent_gripper_width is not None:
            reference_width = self.last_sent_gripper_width
        else:
            return False
        return width < reference_width - self.args.gripper_min_delta

    def _gripper_action_goal_response_cb(self, future, width: float, action_kind: str, is_closing: bool):
        self.gripper_action_goal_pending_response = False
        try:
            goal_handle = future.result()
        except Exception as exc:
            now = time.monotonic()
            if now - self.last_gripper_log_time > 1.0:
                self.get_logger().error(f"Failed to send Franka Hand action goal: {exc}")
                self.last_gripper_log_time = now
            return

        if not goal_handle.accepted:
            now = time.monotonic()
            if now - self.last_gripper_log_time > 1.0:
                self.get_logger().warning("Franka Hand action goal rejected")
                self.last_gripper_log_time = now
            self._clear_gripper_action_if_current(None)
            self._send_pending_gripper_action()
            return

        if self.args.gripper_action_mode in ("preempt", "servo"):
            self.gripper_action_goal_handle = goal_handle
            self.gripper_action_active = True
            self.gripper_action_canceling = False
            self.gripper_action_active_width = width
            self.gripper_action_active_kind = action_kind
            self.gripper_action_active_is_closing = bool(is_closing)
        if self.args.gripper_action_mode == "preempt":
            if (
                self.gripper_action_pending_width is not None
                and abs(self.gripper_action_pending_width - width) >= self.args.gripper_min_delta
            ):
                cancel_future = goal_handle.cancel_goal_async()
                cancel_future.add_done_callback(self._gripper_action_cancel_cb)
                self.gripper_action_canceling = True
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda done, result_goal_handle=goal_handle, goal_width=width, goal_kind=action_kind, goal_is_closing=is_closing: (
                self._gripper_action_result_cb(
                    done,
                    result_goal_handle,
                    goal_width,
                    goal_kind,
                    goal_is_closing,
                )
            )
        )

    def _gripper_action_cancel_cb(self, future):
        try:
            future.result()
        except Exception as exc:
            now = time.monotonic()
            if now - self.last_gripper_log_time > 1.0:
                self.get_logger().warning(f"Failed to cancel Franka Hand action goal: {exc}")
                self.last_gripper_log_time = now

    def _gripper_action_result_cb(self, future, goal_handle=None, width: Optional[float] = None,
            action_kind: Optional[str] = None, is_closing: bool = False):
        try:
            result = future.result().result
        except Exception as exc:
            now = time.monotonic()
            if now - self.last_gripper_log_time > 1.0:
                self.get_logger().warning(f"Failed to get Franka Hand action result: {exc}")
                self.last_gripper_log_time = now
            self._clear_gripper_action_if_current(goal_handle)
            self._send_pending_gripper_action()
            return
        success = bool(getattr(result, "success", True))
        completed_width = self.gripper_action_active_width
        if not success:
            now = time.monotonic()
            if now - self.last_gripper_log_time > 1.0:
                self.get_logger().warning(f"Franka Hand action result was unsuccessful: {result}")
                self.last_gripper_log_time = now
            if (
                self.args.gripper_close_action == "move_then_grasp"
                and action_kind == "move"
                and is_closing
                and width is not None
            ):
                self._clear_gripper_action_if_current(goal_handle)
                self._send_gripper_action_goal(width, force_action_kind="grasp")
                return
        elif self.args.gripper_action_mode == "servo" and completed_width is not None:
            self.current_gripper_width = completed_width
        self._clear_gripper_action_if_current(goal_handle)
        self._send_pending_gripper_action()

    def _clear_gripper_action_if_current(self, goal_handle):
        if self.args.gripper_action_mode not in ("preempt", "servo"):
            return
        if goal_handle is not None and goal_handle is not self.gripper_action_goal_handle:
            return
        self.gripper_action_goal_handle = None
        self.gripper_action_active = False
        self.gripper_action_canceling = False
        self.gripper_action_goal_pending_response = False
        self.gripper_action_active_width = None
        self.gripper_action_active_kind = None
        self.gripper_action_active_is_closing = False

    def _send_pending_gripper_action(self):
        if self.args.gripper_action_mode not in ("preempt", "servo"):
            return False
        if self.gripper_action_pending_width is None:
            return False
        if self.gripper_action_active or self.gripper_action_canceling:
            return False
        pending_width = self.gripper_action_pending_width
        self.gripper_action_pending_width = None
        if self.args.gripper_action_mode == "servo":
            next_width = self._get_next_servo_gripper_width(pending_width)
            if next_width is None:
                self.gripper_action_pending_width = pending_width
                return False
            if abs(next_width - pending_width) >= self.args.gripper_min_delta:
                self.gripper_action_pending_width = pending_width
            return self._send_gripper_action_goal(next_width)
        if (
            self.gripper_action_active_width is not None
            and abs(pending_width - self.gripper_action_active_width) < self.args.gripper_min_delta
        ):
            return False
        return self._send_gripper_action_goal(pending_width)

    def _get_next_servo_gripper_width(self, target_width: float) -> Optional[float]:
        target_width = float(np.clip(target_width, 0.0, self.args.gripper_stroke))
        if self.current_gripper_width is not None:
            base_width = self.current_gripper_width
        elif self.gripper_action_active_width is not None:
            base_width = self.gripper_action_active_width
        elif self.last_sent_gripper_width is not None:
            base_width = self.last_sent_gripper_width
        else:
            return None
        max_step = max(float(self.args.gripper_action_step), 1e-4)
        delta = target_width - base_width
        if abs(delta) < self.args.gripper_min_delta:
            return None
        if abs(delta) <= max_step:
            return target_width
        return float(np.clip(base_width + math.copysign(max_step, delta), 0.0, self.args.gripper_stroke))

    def _try_start_gripper_request(self, width: float) -> bool:
        with self.gripper_lock:
            if self.gripper_request_in_flight:
                return False
            self.gripper_request_in_flight = True

        thread = threading.Thread(
            target=self._gripper_request_worker,
            args=(float(width),),
            name="gello-gripper-http",
            daemon=True,
        )
        thread.start()
        return True

    def _gripper_request_worker(self, width: float):
        try:
            self._post_gripper(width)
        except Exception as exc:
            now = time.monotonic()
            if now - self.last_gripper_log_time > 1.0:
                self.get_logger().error(f"Failed to send gripper target: {exc}")
                self.last_gripper_log_time = now
        else:
            self.last_sent_gripper_width = width
        finally:
            with self.gripper_lock:
                self.gripper_request_in_flight = False

    def _update_recording_command(self, target_tcp: np.ndarray, q: np.ndarray):
        with self.recording_lock:
            self.latest_target_tcp = np.asarray(target_tcp, dtype=np.float64).copy()
            self.latest_recording_q = np.asarray(q, dtype=np.float32).copy()

    def _get_recording_command_state(self) -> Optional[Dict[str, np.ndarray]]:
        with self.recording_lock:
            if self.latest_target_tcp is None or self.latest_recording_q is None:
                return None
            return {
                "target_tcp": self.latest_target_tcp.copy(),
                "gello_command": self.latest_recording_q.copy(),
                "gripper_raw": self.latest_gripper_raw.astype(np.float32, copy=True),
                "gripper_input_raw": self.latest_gripper_input_raw.astype(np.float32, copy=True),
            }

    def _ensure_alignment(self, q: np.ndarray) -> bool:
        if self.correction is not None:
            return True
        try:
            current_tcp = self._get_current_tcp()
        except Exception as exc:
            self.get_logger().error(f"Could not read current TCP for alignment: {exc}")
            return False

        raw_gello_tf = franka_command_fk(q)
        current_tf = transform_from_pose7d(current_tcp)
        self.correction = current_tf @ np.linalg.inv(raw_gello_tf)
        self.get_logger().info(
            "Aligned first GELLO FK pose to current robot TCP. "
            f"current_xyz={np.round(current_tcp[:3], 4).tolist()}"
        )
        return True

    def _timer_cb(self):
        now = time.monotonic()
        self._maybe_send_gripper(now)
        if self.latest_q is None or self.latest_msg_time is None:
            return
        if self.startup_until is not None and now < self.startup_until:
            if now - self.last_log_time > 1.0:
                remaining = self.startup_until - now
                self.get_logger().info(f"Moving to first absolute GELLO pose; continuous follow starts in {remaining:.1f}s")
                self.last_log_time = now
            return
        self.startup_until = None

        if now - self.latest_msg_time > self.args.max_command_age:
            if now - self.last_log_time > 1.0:
                self.get_logger().warning("No recent GELLO command; holding by not sending new targets")
                self.last_log_time = now
            return
        if not self._ensure_alignment(self.latest_q):
            return

        try:
            target_tf = self.correction @ franka_command_fk(self.latest_q)
            target_tcp = pose7d_from_transform(target_tf)
            target_duration = self.args.target_duration
            first_absolute_target = (
                self.sent_count == 0
                and self.args.alignment == "absolute"
                and self.args.first_target_duration > 0
            )
            if first_absolute_target:
                target_duration = self.args.first_target_duration
            self._post_tcp(target_tcp, target_duration=target_duration)
            self._update_recording_command(target_tcp, self.latest_q)
            self.sent_count += 1
            if first_absolute_target and not self.args.dry_run:
                self.startup_until = now + self.args.first_target_duration
                self.get_logger().info(
                    "Sent first absolute GELLO target; pausing stream while robot moves there. "
                    f"duration={self.args.first_target_duration:.2f}s"
                )
        except Exception as exc:
            if now - self.last_log_time > 1.0:
                self.get_logger().error(f"Failed to send TCP target: {exc}")
                self.last_log_time = now
            return

        if now - self.last_log_time > self.args.print_every:
            self.get_logger().info(
                f"sent={self.sent_count} target_xyz={np.round(target_tcp[:3], 4).tolist()}"
            )
            self.last_log_time = now

    def _maybe_send_gripper(self, now: float):
        if self.args.disable_gripper or self.latest_gripper_width is None:
            return
        if self.args.gripper_command_transport == "none":
            return
        if self.latest_gripper_msg_time is None or now - self.latest_gripper_msg_time > self.args.max_command_age:
            return
        if now - self.last_gripper_send_time < self.args.gripper_command_interval:
            return
        if (
            self.last_sent_gripper_width is not None
            and abs(self.latest_gripper_width - self.last_sent_gripper_width) < self.args.gripper_min_delta
        ):
            return
        if self.args.gripper_command_transport == "http":
            sent = self._try_start_gripper_request(self.latest_gripper_width)
        elif self.args.gripper_command_transport == "ros2_action":
            sent = self._try_send_gripper_action(self.latest_gripper_width)
        else:
            raise ValueError(f"Unsupported gripper_command_transport: {self.args.gripper_command_transport}")
        if sent:
            self.last_gripper_send_time = now


def parse_args():
    parser = argparse.ArgumentParser(description="Teleoperate FR3 by bridging GELLO joints to RDP/Polymetis TCP targets.")
    parser.add_argument("--gello-topic", default="/gello/joint_states")
    parser.add_argument("--robot-server", default="http://127.0.0.1:8092")
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--target-duration", type=float, default=None)
    parser.add_argument(
        "--first-target-duration",
        type=float,
        default=3.0,
        help="Duration for the first absolute target command; set 0 to disable.",
    )
    parser.add_argument(
        "--alignment",
        choices=("absolute", "first_frame"),
        default="absolute",
        help="'absolute' sends raw GELLO FK poses; 'first_frame' aligns first GELLO pose to current robot TCP.",
    )
    parser.add_argument("--timeout", type=float, default=0.2)
    parser.add_argument("--max-command-age", type=float, default=0.5)
    parser.add_argument("--print-every", type=float, default=1.0)
    parser.add_argument("--gripper-topic", default="/gripper/gripper_client/target_gripper_width_percent")
    parser.add_argument(
        "--gripper-type",
        choices=tuple(GRIPPER_DEFAULT_STROKES.keys()),
        default="franka_hand",
        help="Select gripper geometry preset. Franka Hand uses 0.08m stroke; Robotiq 2F uses 0.085m.",
    )
    parser.add_argument("--gripper-input-mode", choices=("open_ratio", "close_ratio", "width"), default="open_ratio")
    parser.add_argument("--gripper-control-mode", choices=("direct", "press_open"), default="direct")
    parser.add_argument("--gripper-idle-raw", type=float, default=0.5)
    parser.add_argument("--gripper-pressed-raw", type=float, default=1.0)
    parser.add_argument("--gripper-press-threshold", type=float, default=0.5)
    parser.add_argument("--gripper-press-active", choices=("above", "below"), default="above")
    parser.add_argument(
        "--gripper-stroke",
        type=float,
        default=None,
        help="Override gripper max opening in meters. Defaults to the selected --gripper-type preset.",
    )
    parser.add_argument("--gripper-velocity", type=float, default=0.08)
    parser.add_argument("--gripper-force", type=float, default=20.0)
    parser.add_argument("--gripper-timeout", type=float, default=1.0)
    parser.add_argument(
        "--gripper-command-transport",
        choices=("http", "ros2_action", "none"),
        default="http",
        help=(
            "How to command the gripper. Use ros2_action for lower-latency "
            "Franka Hand control via franka_msgs/action/Move."
        ),
    )
    parser.add_argument(
        "--gripper-move-action",
        default="franka_gripper/move",
        help="ROS2 Move action name used when --gripper-command-transport=ros2_action.",
    )
    parser.add_argument(
        "--gripper-grasp-action",
        default="franka_gripper/grasp",
        help="ROS2 Grasp action name used for closing when --gripper-close-action=grasp.",
    )
    parser.add_argument(
        "--gripper-close-action",
        choices=("move_then_grasp", "grasp", "move"),
        default="move_then_grasp",
        help=(
            "How to handle closing commands. move_then_grasp follows position with Move first, "
            "then sends Grasp only if the closing Move fails/stalls on contact."
        ),
    )
    parser.add_argument("--gripper-grasp-force", type=float, default=35.0)
    parser.add_argument("--gripper-grasp-epsilon-inner", type=float, default=0.005)
    parser.add_argument("--gripper-grasp-epsilon-outer", type=float, default=0.08)
    parser.add_argument(
        "--gripper-joint-states-topic",
        default="franka_gripper/joint_states",
        help="ROS2 Franka Hand joint states topic used by --gripper-action-mode=servo.",
    )
    parser.add_argument(
        "--gripper-action-server-timeout",
        type=float,
        default=1.0,
        help="Seconds to wait for the ROS2 gripper action server at startup; set 0 to skip waiting.",
    )
    parser.add_argument(
        "--gripper-action-mode",
        choices=("servo", "preempt", "stream"),
        default="servo",
        help=(
            "ROS2 gripper action behavior. servo sends short incremental Move goals toward "
            "the latest target. preempt keeps one active goal, cancels it on large target "
            "changes, then sends the latest target. stream sends every target without waiting "
            "and may overload/reject goals."
        ),
    )
    parser.add_argument(
        "--gripper-action-step",
        type=float,
        default=0.004,
        help="Maximum Franka Hand width change per Move goal in servo mode, in meters.",
    )
    parser.add_argument("--gripper-command-interval", type=float, default=0.05)
    parser.add_argument("--gripper-min-delta", type=float, default=0.0005)
    parser.add_argument("--default-gripper-raw", type=float, default=1.0)
    parser.add_argument("--disable-gripper", action="store_true")
    parser.add_argument("--record-output-dir", default=None, help="Optional raw episode output directory.")
    parser.add_argument("--record-rate", type=float, default=25.0)
    parser.add_argument("--record-on-start", action="store_true")
    parser.add_argument("--record-keyboard", action="store_true")
    parser.add_argument("--record-robot-timeout", type=float, default=1.0)
    parser.add_argument(
        "--record-control-host",
        default="127.0.0.1",
        help="Host for experiment UI recording control. Keep the default to accept local requests only.",
    )
    parser.add_argument(
        "--record-control-port",
        type=int,
        default=8765,
        help="HTTP port for experiment UI recording control; set 0 to disable.",
    )
    parser.add_argument("--record-no-camera", action="store_true")
    parser.add_argument(
        "--record-camera-backend",
        choices=("opencv", "realsense", "zed_v4l", "iphone"),
        default="opencv",
    )
    parser.add_argument("--record-camera-source", default="auto")
    parser.add_argument("--record-camera-width", type=int, default=1280)
    parser.add_argument("--record-camera-height", type=int, default=720)
    parser.add_argument("--record-camera-fps", type=float, default=30.0)
    parser.add_argument("--record-zed-view", choices=("left", "right", "full"), default="left")
    parser.add_argument("--record-camera-flip", action="store_true")
    parser.add_argument("--record-allow-gray-camera", action="store_true")
    parser.add_argument("--record-camera-color-threshold", type=float, default=1.5)
    parser.add_argument("--record-iphone-bind-host", default="0.0.0.0")
    parser.add_argument("--record-iphone-video-port", type=int, default=5560)
    parser.add_argument("--record-iphone-combined-port", type=int, default=5558)
    parser.add_argument("--record-iphone-phone-ip", default="")
    parser.add_argument("--record-iphone-registration-port", type=int, default=5559)
    parser.add_argument("--record-iphone-startup-timeout", type=float, default=5.0)
    parser.add_argument("--record-iphone-read-timeout", type=float, default=1.0)
    parser.add_argument("--record-iphone-hello-interval", type=float, default=2.0)
    parser.add_argument(
        "--record-image-height",
        type=int,
        default=224,
        help="Compatibility option; recording now stores raw RGB and pack script handles crop/resize.",
    )
    parser.add_argument(
        "--record-image-width",
        type=int,
        default=224,
        help="Compatibility option; recording now stores raw RGB and pack script handles crop/resize.",
    )
    parser.add_argument("--record-no-magnet", action="store_true")
    parser.add_argument(
        "--record-magnet-port",
        nargs="+",
        default=["/dev/ttyACM0"],
        help=(
            "One or two serial ports. With two ports, the first is stored as "
            "magnet_* and the second as magnet2_*."
        ),
    )
    parser.add_argument("--record-magnet-baudrate", type=int, default=115200)
    parser.add_argument("--record-magnet-samples-per-frame", type=int, default=MAGNET_SAMPLES_PER_FRAME)
    parser.add_argument("--record-magnet-no-baseline", action="store_true")
    parser.add_argument(
        "--record-no-magnet-reset-baseline-on-start",
        dest="record_magnet_reset_baseline_on_start",
        action="store_false",
    )
    parser.set_defaults(record_magnet_reset_baseline_on_start=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.record_magnet_port = list(normalize_record_magnet_ports(args.record_magnet_port))
    if args.rate <= 0:
        raise ValueError("--rate must be positive")
    if args.target_duration is None:
        args.target_duration = 1.0 / args.rate
    if args.gripper_stroke is None:
        args.gripper_stroke = GRIPPER_DEFAULT_STROKES[args.gripper_type]
    if args.gripper_stroke <= 0:
        raise ValueError("--gripper-stroke must be positive")
    args.gripper_idle_raw = float(np.clip(args.gripper_idle_raw, 0.0, 1.0))
    args.gripper_pressed_raw = float(np.clip(args.gripper_pressed_raw, 0.0, 1.0))
    args.gripper_press_threshold = float(np.clip(args.gripper_press_threshold, 0.0, 1.0))
    args.default_gripper_raw = float(np.clip(args.default_gripper_raw, 0.0, 1.0))
    if args.record_output_dir and not args.record_on_start and not args.record_keyboard:
        args.record_keyboard = True
    if not 0 <= args.record_control_port <= 65535:
        raise ValueError("--record-control-port must be in [0, 65535]")
    for option in (
        "record_iphone_video_port",
        "record_iphone_combined_port",
        "record_iphone_registration_port",
    ):
        value = getattr(args, option)
        if not 0 < value <= 65535:
            raise ValueError(f"--{option.replace('_', '-')} must be in [1, 65535]")
    for option in (
        "record_iphone_startup_timeout",
        "record_iphone_read_timeout",
        "record_iphone_hello_interval",
    ):
        if getattr(args, option) <= 0:
            raise ValueError(f"--{option.replace('_', '-')} must be positive")
    return args


def main():
    args = parse_args()
    rclpy.init()
    node = GelloPolymetisTcpTeleop(args)
    try:
        rclpy.spin(node)
    finally:
        if node.recording_control_server is not None:
            node.recording_control_server.stop()
        if node.recorder is not None:
            node.recorder.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
