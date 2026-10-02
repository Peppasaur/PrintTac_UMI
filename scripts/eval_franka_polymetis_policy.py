#!/usr/bin/env python3
"""
Evaluate an RDP/DP image policy on a Franka through the local Polymetis HTTP
server used by scripts/replay_franka_polymetis_dataset.py.

This script is intentionally narrow: it supports the single-arm 10D Franka DP
checkpoint produced by train_dp.sh:

    left_wrist_img + left_robot_tcp_pose + left_robot_gripper_width -> action

The policy action is interpreted as xyz + rot6d + gripper width.  In execution
mode it is converted to x,y,z,qw,qx,qy,qz and sent to /move_tcp/left.
"""
import argparse
import glob
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import dill
import hydra
import numpy as np
import requests
import scipy.spatial.transform as st
import torch
import zarr
from omegaconf import OmegaConf


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from reactive_diffusion_policy.common.ensemble import EnsembleBuffer
from reactive_diffusion_policy.common.precise_sleep import precise_wait
from reactive_diffusion_policy.common.pytorch_util import dict_apply
from reactive_diffusion_policy.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


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


def normalize_pose7d(pose):
    pose = np.asarray(pose, dtype=np.float64).copy()
    norm = np.linalg.norm(pose[..., 3:7], axis=-1, keepdims=True)
    pose[..., 3:7] /= np.maximum(norm, 1e-12)
    return pose


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


def rotvec_pose_to_pose7d(pose6):
    pose6 = np.asarray(pose6, dtype=np.float64)
    if pose6.ndim == 1:
        pose6 = pose6[None]
    quat_wxyz = xyzw_to_wxyz(st.Rotation.from_rotvec(pose6[:, 3:6]).as_quat())
    pose7 = np.concatenate([pose6[:, :3], quat_wxyz], axis=-1)
    return normalize_pose7d(pose7)


def pose7d_to_pose9(pose7):
    pose7 = normalize_pose7d(np.asarray(pose7, dtype=np.float64))
    rot_mat = st.Rotation.from_quat(wxyz_to_xyzw(pose7[3:7])).as_matrix()
    rot6d = rot_mat[:, :2].T.reshape(6)
    return np.concatenate([pose7[:3], rot6d], axis=0).astype(np.float32)


def action10_to_pose7d(action):
    action = np.asarray(action, dtype=np.float64)
    if action.shape[-1] != 10:
        raise ValueError(f"Expected 10D action, got {action.shape}")
    pose9 = action[:9][None]
    rot_mat = ortho6d_to_rotation_matrix(pose9[:, 3:9])
    quat_wxyz = xyzw_to_wxyz(st.Rotation.from_matrix(rot_mat).as_quat())
    return normalize_pose7d(np.concatenate([pose9[:, :3], quat_wxyz], axis=-1)[0])


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def get_attr(cfg, dotted_key, default=None):
    node = cfg
    for key in dotted_key.split("."):
        if node is None or key not in node:
            return default
        node = node[key]
    return node


def shape_meta_keys(shape_meta):
    rgb_keys = []
    lowdim_keys = []
    for key, attr in shape_meta["obs"].items():
        if attr.get("type", "low_dim") == "rgb":
            rgb_keys.append(key)
        else:
            lowdim_keys.append(key)
    return rgb_keys, lowdim_keys


def open_zarr(path):
    path = os.path.expanduser(path)
    if path.endswith(".zip"):
        store = zarr.ZipStore(path, mode="r")
        return zarr.group(store), store
    return zarr.open(path, mode="r"), None


def get_episode_slice(episode_ends, episode_idx):
    if episode_idx < 0 or episode_idx >= len(episode_ends):
        raise ValueError(
            f"episode_idx must be in [0, {len(episode_ends) - 1}], got {episode_idx}"
        )
    start = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
    end = int(episode_ends[episode_idx])
    return slice(start, end), start, end


def load_match_episode(dataset_path, episode_idx):
    root, store = open_zarr(dataset_path)
    try:
        if "data" not in root or "meta" not in root:
            raise KeyError("match dataset must contain data and meta groups")
        data = root["data"]
        episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
        episode_slice, start, end = get_episode_slice(episode_ends, episode_idx)

        if "robot0_eef_pos" in data and "robot0_eef_rot_axis_angle" in data:
            pos = np.asarray(data["robot0_eef_pos"][episode_slice][0], dtype=np.float64)
            rot = np.asarray(data["robot0_eef_rot_axis_angle"][episode_slice][0], dtype=np.float64)
            start_pose = rotvec_pose_to_pose7d(np.concatenate([pos, rot]))[0]
            pose_source = "robot0_eef_pos + robot0_eef_rot_axis_angle"
        elif "left_robot_tcp_pose" in data:
            pose = np.asarray(data["left_robot_tcp_pose"][episode_slice][0], dtype=np.float64)
            if pose.shape[0] == 9:
                start_pose = action10_to_pose7d(np.concatenate([pose, [0.0]]))
                pose_source = "left_robot_tcp_pose xyz+rot6d"
            elif pose.shape[0] == 7:
                start_pose = normalize_pose7d(pose)
                pose_source = "left_robot_tcp_pose xyz+quat"
            else:
                raise ValueError(f"Unsupported left_robot_tcp_pose shape {pose.shape}")
        else:
            raise KeyError("No supported robot pose keys in match dataset")

        if "robot0_gripper_width" in data:
            gripper = np.asarray(data["robot0_gripper_width"][episode_slice][0]).reshape(-1)[0]
            gripper_source = "robot0_gripper_width"
        elif "left_robot_gripper_width" in data:
            gripper = np.asarray(data["left_robot_gripper_width"][episode_slice][0]).reshape(-1)[0]
            gripper_source = "left_robot_gripper_width"
        elif "action" in data and data["action"].shape[-1] in (7, 10):
            gripper = np.asarray(data["action"][episode_slice][0]).reshape(-1)[-1]
            gripper_source = "action[-1]"
        else:
            gripper = None
            gripper_source = None

        return {
            "start_pose": start_pose,
            "gripper": None if gripper is None else float(gripper),
            "pose_source": pose_source,
            "gripper_source": gripper_source,
            "episode_start": start,
            "episode_end": end,
            "episode_count": len(episode_ends),
        }
    finally:
        if store is not None:
            store.close()


def camera_source_candidates(source):
    if source != "auto":
        try:
            return [int(source)]
        except ValueError:
            return [source]

    candidates = []
    # D435i commonly exposes multiple V4L nodes.  On many machines the color
    # sensor is USB interface 1.3, while interface 1.0 nodes are depth/IR.
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

    deduped = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped or [0]


def bgr_color_score(frame):
    """Return a small colorfulness score; grayscale-expanded BGR is near zero."""
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        return 0.0
    frame_f = frame.astype(np.float32)
    bg = np.mean(np.abs(frame_f[..., 0] - frame_f[..., 1]))
    gr = np.mean(np.abs(frame_f[..., 1] - frame_f[..., 2]))
    br = np.mean(np.abs(frame_f[..., 0] - frame_f[..., 2]))
    return float(max(bg, gr, br))


class OpenCVCamera:
    def __init__(self, source, width=None, height=None, fps=None, require_color=True, color_threshold=1.5):
        self.requested_source = source
        self.source = None
        self.candidates = camera_source_candidates(source)
        self.width = width
        self.height = height
        self.fps = fps
        self.require_color = require_color
        self.color_threshold = color_threshold
        self.cap = None
        self.color_score = None

    def __enter__(self):
        errors = []
        for source in self.candidates:
            cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
            if self.width is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.width))
            if self.height is not None:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.height))
            if self.fps is not None:
                cap.set(cv2.CAP_PROP_FPS, float(self.fps))

            if not cap.isOpened():
                errors.append(f"{source}: open failed")
                cap.release()
                continue

            ok_frame = False
            last_shape = None
            last_color_score = None
            for _ in range(10):
                ok, frame = cap.read()
                if ok and frame is not None:
                    last_shape = frame.shape
                    last_color_score = bgr_color_score(frame)
                    ok_frame = (
                        frame.ndim == 3
                        and frame.shape[2] == 3
                        and (
                            not self.require_color
                            or last_color_score >= self.color_threshold
                        )
                    )
                    if ok_frame:
                        break
                time.sleep(0.05)

            if ok_frame:
                self.source = source
                self.cap = cap
                self.color_score = last_color_score
                return self

            errors.append(
                f"{source}: no usable color BGR frame, "
                f"last_shape={last_shape}, color_score={last_color_score}"
            )
            cap.release()

        candidate_text = ", ".join(str(x) for x in self.candidates)
        detail = "; ".join(errors)
        raise RuntimeError(
            "Could not open a usable OpenCV camera source. "
            f"requested={self.requested_source!r}; candidates=[{candidate_text}]. "
            f"Failures: {detail}"
        )

    def __exit__(self, exc_type, exc, tb):
        if self.cap is not None:
            self.cap.release()

    def read(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame from camera source: {self.source}")
        return frame

    def actual_resolution(self):
        return (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )


def preprocess_bgr_like_convert_zarr(frame_bgr, output_shape):
    """BGR OpenCV frame -> RGB uint8 matching GELLO convert_zarr then RDP resize."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"Expected BGR image HxWx3, got {frame_bgr.shape}")

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    sq = min(h, w)
    y0 = (h - sq) // 2
    x0 = (w - sq) // 2
    rgb = rgb[y0 : y0 + sq, x0 : x0 + sq]
    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)

    c, out_h, out_w = output_shape
    if c != 3:
        raise ValueError(f"Expected RGB image shape [3,H,W], got {output_shape}")
    if (out_h, out_w) != (224, 224):
        rgb = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
    return rgb.astype(np.uint8)


def save_rgb(path, rgb):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def get_current_tcp(session, base_url, timeout):
    response = session.get(f"{base_url}/get_current_tcp/left", timeout=timeout)
    response.raise_for_status()
    pose = np.asarray(response.json(), dtype=np.float64)
    if pose.shape != (7,):
        raise ValueError(f"Expected current TCP shape (7,), got {pose.shape}")
    return normalize_pose7d(pose)


def get_current_gripper(session, base_url, timeout, fallback):
    try:
        response = session.get(f"{base_url}/get_current_robot_states", timeout=timeout)
        response.raise_for_status()
        state = response.json()
        return float(state.get("leftGripperState", [fallback])[0])
    except Exception:
        return float(fallback)


def gripper_width_to_obs(width_m, mode, stroke, raw_min=0.0, raw_max=1.0):
    width_m = float(width_m)
    if mode == "width":
        return width_m
    if mode == "open_ratio":
        open_ratio = np.clip(width_m / max(stroke, 1e-6), 0.0, 1.0)
        return float(raw_min + open_ratio * (raw_max - raw_min))
    raise ValueError(f"Unsupported gripper obs mode: {mode}")


def gripper_action_to_width(value, mode, stroke, raw_min=0.0, raw_max=1.0):
    value = float(value)
    if mode == "width":
        return float(np.clip(value, 0.0, stroke))
    if mode == "open_ratio":
        denom = max(float(raw_max) - float(raw_min), 1e-6)
        open_ratio = np.clip((value - raw_min) / denom, 0.0, 1.0)
        return float(open_ratio * stroke)
    raise ValueError(f"Unsupported gripper action mode: {mode}")


def get_current_gripper_obs(session, base_url, timeout, fallback_obs, obs_mode,
        stroke, raw_min=0.0, raw_max=1.0):
    try:
        response = session.get(f"{base_url}/get_current_robot_states", timeout=timeout)
        response.raise_for_status()
        state = response.json()
        width_m = float(state.get("leftGripperState", [fallback_obs])[0])
        return gripper_width_to_obs(width_m, obs_mode, stroke, raw_min, raw_max)
    except Exception:
        return float(fallback_obs)


def post_tcp(session, base_url, pose7d, timeout, target_duration=None):
    payload = {"target_tcp": [float(x) for x in normalize_pose7d(pose7d)]}
    if target_duration is not None:
        payload["target_duration"] = float(target_duration)
    response = session.post(f"{base_url}/move_tcp/left", json=payload, timeout=timeout)
    response.raise_for_status()


def post_gripper(session, base_url, width, velocity, force_limit, timeout):
    payload = {
        "width": float(width),
        "velocity": float(velocity),
        "force_limit": float(force_limit),
    }
    response = session.post(f"{base_url}/move_gripper/left", json=payload, timeout=timeout)
    response.raise_for_status()


def infer_gripper_raw_range(dataset_path):
    if dataset_path is None:
        return None
    dataset_path = os.path.expanduser(str(dataset_path))
    candidates = [dataset_path]
    if os.path.isdir(dataset_path):
        candidates.insert(0, os.path.join(dataset_path, "replay_buffer.zarr"))

    for candidate in candidates:
        if not os.path.exists(candidate):
            continue
        root, store = open_zarr(candidate)
        try:
            data = root["data"] if "data" in root else root
            if "action" not in data:
                continue
            action = np.asarray(data["action"][:, -1], dtype=np.float64)
            action = action[np.isfinite(action)]
            if action.size == 0:
                continue
            raw_min = float(np.min(action))
            raw_max = float(np.max(action))
            if raw_max > raw_min:
                return raw_min, raw_max, candidate
        finally:
            if store is not None:
                store.close()
    return None


def interpolate_pose7d(start_pose, end_pose, alpha):
    alpha = np.asarray(alpha, dtype=np.float64)
    start_pose = normalize_pose7d(start_pose)
    end_pose = normalize_pose7d(end_pose)
    pos = start_pose[:3][None, :] * (1.0 - alpha[:, None]) + end_pose[:3][None, :] * alpha[:, None]
    slerp = st.Slerp(
        [0.0, 1.0],
        st.Rotation.from_quat([wxyz_to_xyzw(start_pose[3:7]), wxyz_to_xyzw(end_pose[3:7])]),
    )
    quat_wxyz = xyzw_to_wxyz(slerp(alpha).as_quat())
    return normalize_pose7d(np.concatenate([pos, quat_wxyz], axis=-1))


def move_to_start(session, base_url, target_pose, duration, frequency, timeout):
    current_pose = get_current_tcp(session, base_url, timeout)
    n_steps = max(2, int(np.ceil(duration * frequency)))
    waypoints = interpolate_pose7d(current_pose, target_pose, np.linspace(0.0, 1.0, n_steps))
    t0 = time.monotonic()
    for i, waypoint in enumerate(waypoints[1:], start=1):
        step_duration = duration / (n_steps - 1)
        precise_wait(t0 + i * step_duration, time_func=time.monotonic)
        post_tcp(session, base_url, waypoint, timeout, target_duration=step_duration)


def load_policy(ckpt_path, device, num_inference_steps):
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if bool(get_attr(cfg, "training.use_ema", False)) and getattr(workspace, "ema_model", None) is not None:
        policy = workspace.ema_model
        load_method = "ema_model"
    else:
        load_method = "model"

    policy.num_inference_steps = int(num_inference_steps)
    policy.eval().to(torch.device(device))
    return cfg, policy, load_method


def build_obs_tensor(obs_history, shape_meta, rgb_key, device):
    if not obs_history:
        raise RuntimeError("Observation history is empty")
    obs_dict = {}
    for key, attr in shape_meta["obs"].items():
        shape = tuple(attr["shape"])
        if attr.get("type", "low_dim") == "rgb":
            if key != rgb_key:
                raise NotImplementedError(f"Only one RGB camera is supported, got {key}")
            imgs = np.stack([obs[key] for obs in obs_history], axis=0)
            obs_dict[key] = np.moveaxis(imgs, -1, 1).astype(np.float32) / 255.0
        elif key == "left_robot_tcp_pose":
            obs_dict[key] = np.stack([obs[key][: shape[0]] for obs in obs_history], axis=0).astype(np.float32)
        elif key == "left_robot_gripper_width":
            obs_dict[key] = np.stack([obs[key][: shape[0]] for obs in obs_history], axis=0).astype(np.float32)
        else:
            raise NotImplementedError(f"Unsupported obs key for this FR3 eval script: {key}")

    return dict_apply(obs_dict, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))


def collect_observation(camera, session, base_url, timeout, image_shape,
        gripper_fallback, gripper_obs_mode, gripper_stroke,
        gripper_raw_min, gripper_raw_max):
    frame_bgr = camera.read()
    image_rgb = preprocess_bgr_like_convert_zarr(frame_bgr, image_shape)
    tcp_pose7d = get_current_tcp(session, base_url, timeout)
    gripper_width = get_current_gripper_obs(
        session=session,
        base_url=base_url,
        timeout=timeout,
        fallback_obs=gripper_fallback,
        obs_mode=gripper_obs_mode,
        stroke=gripper_stroke,
        raw_min=gripper_raw_min,
        raw_max=gripper_raw_max,
    )
    return {
        "left_wrist_img": image_rgb,
        "left_robot_tcp_pose": pose7d_to_pose9(tcp_pose7d),
        "left_robot_gripper_width": np.array([gripper_width], dtype=np.float32),
        "timestamp": time.monotonic(),
    }


def select_obs_history(obs_list, n_obs_steps, downsample_steps):
    if n_obs_steps <= 0:
        raise ValueError(f"n_obs_steps must be positive, got {n_obs_steps}")
    downsample_steps = max(1, int(downsample_steps))
    history = list(obs_list)[::-downsample_steps][:n_obs_steps][::-1]
    if len(history) < n_obs_steps:
        return None
    return history


class ObservationSampler(threading.Thread):
    def __init__(self, camera, base_url, timeout, image_shape, gripper_fallback,
            gripper_obs_mode, gripper_stroke, gripper_raw_min, gripper_raw_max,
            sample_fps, history_len, stop_event):
        super().__init__(name="FrankaPolicyObservationSampler", daemon=True)
        self.camera = camera
        self.base_url = base_url
        self.timeout = timeout
        self.image_shape = image_shape
        self.gripper_fallback = gripper_fallback
        self.gripper_obs_mode = gripper_obs_mode
        self.gripper_stroke = gripper_stroke
        self.gripper_raw_min = gripper_raw_min
        self.gripper_raw_max = gripper_raw_max
        self.sample_interval = 1.0 / sample_fps
        self.buffer = deque(maxlen=history_len)
        self.lock = threading.Lock()
        self.stop_event = stop_event
        self.error = None

    def run(self):
        session = requests.Session()
        sample_idx = 0
        start_time = time.monotonic()
        try:
            while not self.stop_event.is_set():
                obs = collect_observation(
                    camera=self.camera,
                    session=session,
                    base_url=self.base_url,
                    timeout=self.timeout,
                    image_shape=self.image_shape,
                    gripper_fallback=self.gripper_fallback,
                    gripper_obs_mode=self.gripper_obs_mode,
                    gripper_stroke=self.gripper_stroke,
                    gripper_raw_min=self.gripper_raw_min,
                    gripper_raw_max=self.gripper_raw_max,
                )
                with self.lock:
                    self.buffer.append(obs)
                sample_idx += 1
                precise_wait(
                    start_time + sample_idx * self.sample_interval,
                    time_func=time.monotonic,
                )
        except Exception as exc:
            self.error = exc
            self.stop_event.set()
        finally:
            session.close()

    def __len__(self):
        with self.lock:
            return len(self.buffer)

    def get_history(self, n_obs_steps, downsample_steps):
        with self.lock:
            obs_list = list(self.buffer)
        return select_obs_history(obs_list, n_obs_steps, downsample_steps)

    def latest(self):
        with self.lock:
            if not self.buffer:
                return None
            return self.buffer[-1]


class GripperCommandThread(threading.Thread):
    def __init__(self, base_url, velocity, force, threshold, timeout,
            command_interval, stop_event, verbose):
        super().__init__(name="FrankaPolicyGripperThread", daemon=True)
        self.base_url = base_url
        self.velocity = velocity
        self.force = force
        self.threshold = threshold
        self.timeout = timeout
        self.command_interval = max(0.0, float(command_interval))
        self.stop_event = stop_event
        self.verbose = verbose
        self.lock = threading.Lock()
        self.target_width = None
        self.last_sent_width = None
        self.last_command_time = -float("inf")
        self.error = None

    def set_target(self, width):
        with self.lock:
            self.target_width = float(width)

    def run(self):
        session = requests.Session()
        try:
            while not self.stop_event.is_set():
                with self.lock:
                    width = self.target_width
                now = time.monotonic()
                should_send = width is not None
                if should_send and self.last_sent_width is not None:
                    should_send = abs(width - self.last_sent_width) >= self.threshold
                if should_send:
                    should_send = now - self.last_command_time >= self.command_interval

                if should_send:
                    try:
                        post_gripper(
                            session=session,
                            base_url=self.base_url,
                            width=width,
                            velocity=self.velocity,
                            force_limit=self.force,
                            timeout=self.timeout,
                        )
                        self.last_sent_width = width
                        self.last_command_time = now
                    except Exception as exc:
                        self.error = exc
                        self.last_command_time = now
                        if self.verbose:
                            print(f"gripper command failed: {exc}")

                time.sleep(0.01)
        finally:
            session.close()


class ActionCommandThread(threading.Thread):
    def __init__(self, tcp_buffer, gripper_buffer, base_url, control_fps,
            execute, enable_gripper, gripper_stroke, gripper_velocity,
            gripper_force, gripper_threshold, timeout, stop_event, verbose,
            tcp_pos_clip_range=None, gripper_action_mode="open_ratio",
            gripper_raw_min=0.0, gripper_raw_max=1.0, gripper_timeout=None,
            gripper_command_interval=0.5, tcp_target_duration=None):
        super().__init__(name="FrankaPolicyActionThread", daemon=True)
        self.tcp_buffer = tcp_buffer
        self.gripper_buffer = gripper_buffer
        self.base_url = base_url
        self.control_interval = 1.0 / control_fps
        self.execute = execute
        self.enable_gripper = enable_gripper
        self.gripper_stroke = gripper_stroke
        self.gripper_velocity = gripper_velocity
        self.gripper_force = gripper_force
        self.gripper_threshold = gripper_threshold
        self.gripper_action_mode = gripper_action_mode
        self.gripper_raw_min = gripper_raw_min
        self.gripper_raw_max = gripper_raw_max
        self.timeout = timeout
        self.gripper_timeout = timeout if gripper_timeout is None else gripper_timeout
        self.gripper_command_interval = max(0.0, float(gripper_command_interval))
        self.tcp_target_duration = tcp_target_duration
        self.stop_event = stop_event
        self.verbose = verbose
        self.tcp_pos_clip_range = tcp_pos_clip_range
        self.error = None

    def clip_tcp(self, tcp):
        tcp = np.asarray(tcp, dtype=np.float64).copy()
        if self.tcp_pos_clip_range is not None:
            lo, hi = self.tcp_pos_clip_range
            tcp[:3] = np.clip(tcp[:3], np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64))
        return tcp

    def run(self):
        session = requests.Session()
        gripper_thread = None
        try:
            if self.enable_gripper:
                gripper_thread = GripperCommandThread(
                    base_url=self.base_url,
                    velocity=self.gripper_velocity,
                    force=self.gripper_force,
                    threshold=self.gripper_threshold,
                    timeout=self.gripper_timeout,
                    command_interval=self.gripper_command_interval,
                    stop_event=self.stop_event,
                    verbose=self.verbose,
                )
                gripper_thread.start()
            while not self.stop_event.is_set():
                start = time.monotonic()
                tcp = self.tcp_buffer.get_action()
                gripper = self.gripper_buffer.get_action()
                if tcp is not None and gripper is not None:
                    tcp = self.clip_tcp(tcp)
                    action = np.concatenate([tcp, gripper], axis=-1)
                    pose7d = action10_to_pose7d(action)
                    raw_gripper = float(action[-1])
                    width = gripper_action_to_width(
                        raw_gripper,
                        self.gripper_action_mode,
                        self.gripper_stroke,
                        self.gripper_raw_min,
                        self.gripper_raw_max,
                    )
                    if self.verbose:
                        print(
                            "action",
                            f"xyz={np.array2string(action[:3], precision=4)}",
                            f"gripper_raw={raw_gripper:.4f}",
                            f"gripper_width_m={width:.4f}",
                        )
                    if self.execute:
                        post_tcp(
                            session,
                            self.base_url,
                            pose7d,
                            self.timeout,
                            target_duration=self.tcp_target_duration,
                        )
                        if gripper_thread is not None:
                            gripper_thread.set_target(width)
                elapsed = time.monotonic() - start
                time.sleep(max(0.0, self.control_interval - elapsed))
        except Exception as exc:
            self.error = exc
            self.stop_event.set()
        finally:
            if gripper_thread is not None:
                gripper_thread.join()
            session.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained single-arm RDP/DP checkpoint on FR3 + Polymetis."
    )
    parser.add_argument("--input", "-i", required=True, help="Path to .ckpt checkpoint.")
    parser.add_argument("--execute", action="store_true", help="Actually send actions to the robot.")
    parser.add_argument("--dry-run", action="store_true", help="Run policy/camera loop without robot commands.")
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8092)
    parser.add_argument("--camera-source", default="auto", help="'auto', an integer index, or a /dev/video* path.")
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument(
        "--allow-monochrome-camera",
        action="store_true",
        help="Allow grayscale-expanded BGR camera frames. This is for debugging only; policy eval expects RGB.",
    )
    parser.add_argument(
        "--camera-color-threshold",
        type=float,
        default=1.5,
        help="Minimum mean channel-difference score for accepting an OpenCV source as color.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument("--max-duration", type=float, default=20.0)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument(
        "--obs-sample-fps",
        type=float,
        default=None,
        help="Observation sampling FPS. Defaults to control_fps from the checkpoint env_runner.",
    )
    parser.add_argument(
        "--obs-downsample-steps",
        type=int,
        default=None,
        help=(
            "Temporal downsample step for observation history. Defaults to the training dataset "
            "obs_temporal_downsample_ratio if present, otherwise 1."
        ),
    )
    parser.add_argument("--save-camera-debug", default=None)
    parser.add_argument(
        "--save-camera-debug-dir",
        default=None,
        help="Optional directory to save processed policy camera frames during rollout.",
    )
    parser.add_argument(
        "--save-camera-debug-every",
        type=int,
        default=1,
        help="Save one processed frame every N policy inferences when --save-camera-debug-dir is set.",
    )
    parser.add_argument("--match-dataset", default=None, help="Optional zarr dataset for moving to an episode start pose.")
    parser.add_argument("--match-episode", type=int, default=0)
    parser.add_argument("--no-move-to-start", action="store_true")
    parser.add_argument("--move-to-start-duration", type=float, default=4.0)
    parser.add_argument("--enable-gripper", action="store_true")
    parser.add_argument("--gripper-stroke", type=float, default=0.085)
    parser.add_argument(
        "--gripper-obs-mode",
        choices=["open_ratio", "width"],
        default="open_ratio",
        help="Unit used by the checkpoint for gripper observation.",
    )
    parser.add_argument(
        "--gripper-action-mode",
        choices=["open_ratio", "width"],
        default="open_ratio",
        help="Unit used by the checkpoint for gripper action.",
    )
    parser.add_argument(
        "--gripper-raw-min",
        type=float,
        default=None,
        help="Minimum raw open-ratio value in the training data. Auto-inferred from action[-1] by default.",
    )
    parser.add_argument(
        "--gripper-raw-max",
        type=float,
        default=None,
        help="Maximum raw open-ratio value in the training data. Auto-inferred from action[-1] by default.",
    )
    parser.add_argument(
        "--no-infer-gripper-range",
        action="store_true",
        help="Use [0, 1] for open-ratio gripper scaling instead of inferring from the training zarr.",
    )
    parser.add_argument("--gripper-velocity", type=float, default=0.08)
    parser.add_argument("--gripper-force", type=float, default=20.0)
    parser.add_argument("--gripper-threshold", type=float, default=0.002)
    parser.add_argument("--http-timeout", type=float, default=1.0)
    parser.add_argument(
        "--tcp-target-duration",
        type=float,
        default=None,
        help="Seconds given to the robot server to reach each policy TCP waypoint. Defaults to 1/control_fps.",
    )
    parser.add_argument("--gripper-http-timeout", type=float, default=5.0)
    parser.add_argument(
        "--gripper-command-interval",
        type=float,
        default=0.5,
        help="Minimum seconds between gripper POST commands during rollout.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dry_run:
        args.execute = False

    base_url = f"http://{args.server_host}:{args.server_port}"
    cfg, policy, load_method = load_policy(args.input, args.device, args.num_inference_steps)
    shape_meta = cfg.task.shape_meta
    rgb_keys, lowdim_keys = shape_meta_keys(shape_meta)
    if len(rgb_keys) != 1:
        raise NotImplementedError(f"Expected exactly one RGB key, got {rgb_keys}")
    rgb_key = rgb_keys[0]
    if rgb_key != "left_wrist_img":
        raise NotImplementedError(f"This eval script expects left_wrist_img, got {rgb_key}")
    action_dim = int(shape_meta["action"]["shape"][0])
    if action_dim != 10:
        raise NotImplementedError(f"This eval script supports 10D action only, got {action_dim}")
    if bool(get_attr(cfg, "task.dataset.relative_action", False)):
        raise NotImplementedError("Relative-action checkpoints are not supported by this FR3 eval script yet.")

    env_runner_cfg = cfg.task.get("env_runner", {})
    control_fps = float(env_runner_cfg.get("control_fps", 12.0))
    inference_fps = float(env_runner_cfg.get("inference_fps", 6.0))
    steps_per_inference = max(1, int(round(control_fps / inference_fps)))
    tcp_update_interval = int(env_runner_cfg.get("tcp_action_update_interval", steps_per_inference))
    gripper_update_interval = int(env_runner_cfg.get("gripper_action_update_interval", steps_per_inference))
    latency_step = int(env_runner_cfg.get("latency_step", 0))
    gripper_latency_step = int(env_runner_cfg.get("gripper_latency_step", latency_step))
    n_obs_steps = int(policy.n_obs_steps)
    dataset_cfg = cfg.task.get("dataset", {})
    dataset_obs_downsample_steps = int(dataset_cfg.get("obs_temporal_downsample_ratio", 1))
    env_obs_downsample_steps = int(env_runner_cfg.get("obs_temporal_downsample_ratio", dataset_obs_downsample_steps))
    if args.obs_downsample_steps is None:
        obs_downsample_steps = dataset_obs_downsample_steps
    else:
        obs_downsample_steps = int(args.obs_downsample_steps)
    obs_downsample_steps = max(1, obs_downsample_steps)
    obs_sample_fps = float(args.obs_sample_fps) if args.obs_sample_fps is not None else control_fps
    history_len = max(1, 1 + (n_obs_steps - 1) * obs_downsample_steps)
    image_shape = tuple(shape_meta["obs"][rgb_key]["shape"])
    tcp_target_duration = (
        float(args.tcp_target_duration)
        if args.tcp_target_duration is not None
        else 1.0 / max(control_fps, 1e-6)
    )
    tcp_pos_clip_range = env_runner_cfg.get("tcp_pos_clip_range", None)
    if tcp_pos_clip_range is not None:
        tcp_pos_clip_range = OmegaConf.to_container(tcp_pos_clip_range, resolve=True)

    gripper_raw_min = 0.0 if args.gripper_raw_min is None else float(args.gripper_raw_min)
    gripper_raw_max = 1.0 if args.gripper_raw_max is None else float(args.gripper_raw_max)
    gripper_range_source = "cli/default"
    if (
        not args.no_infer_gripper_range
        and args.gripper_action_mode == "open_ratio"
        and (args.gripper_raw_min is None or args.gripper_raw_max is None)
    ):
        inferred = infer_gripper_raw_range(get_attr(cfg, "task.dataset_path", None))
        if inferred is None:
            inferred = infer_gripper_raw_range(get_attr(cfg, "task.dataset.dataset_path", None))
        if inferred is not None:
            inferred_min, inferred_max, inferred_source = inferred
            if args.gripper_raw_min is None:
                gripper_raw_min = inferred_min
            if args.gripper_raw_max is None:
                gripper_raw_max = inferred_max
            gripper_range_source = inferred_source
    if gripper_raw_max <= gripper_raw_min:
        raise ValueError(
            f"gripper raw max must be greater than min, got {gripper_raw_min}..{gripper_raw_max}"
        )

    tcp_buffer = EnsembleBuffer(**OmegaConf.to_container(
        env_runner_cfg.get("tcp_ensemble_buffer_params", {"ensemble_mode": "new"}),
        resolve=True,
    ))
    gripper_buffer = EnsembleBuffer(**OmegaConf.to_container(
        env_runner_cfg.get("gripper_ensemble_buffer_params", {"ensemble_mode": "new"}),
        resolve=True,
    ))

    print("Checkpoint:", args.input, f"({load_method})")
    print("Task:", cfg.task.name)
    print("Device:", args.device)
    print("Obs keys:", list(shape_meta["obs"].keys()))
    print("Action:", "10D xyz + rot6d + gripper_width")
    print(
        "Timing:",
        f"control_fps={control_fps:g}",
        f"inference_fps={inference_fps:g}",
        f"steps_per_inference={steps_per_inference}",
        f"obs_sample_fps={obs_sample_fps:g}",
        f"tcp_target_duration={tcp_target_duration:g}s",
        f"history_len={history_len}",
    )
    print(
        "Obs history:",
        f"n_obs_steps={n_obs_steps}",
        f"effective_downsample={obs_downsample_steps}",
        f"dataset_downsample={dataset_obs_downsample_steps}",
        f"env_runner_downsample={env_obs_downsample_steps}",
    )
    print(
        "Action schedule:",
        f"tcp_update_interval={tcp_update_interval}",
        f"gripper_update_interval={gripper_update_interval}",
        f"latency_step={latency_step}",
        f"gripper_latency_step={gripper_latency_step}",
    )
    print("Robot server:", base_url, "execute=", args.execute)
    print(
        "Gripper:",
        f"enable={args.enable_gripper}",
        f"obs_mode={args.gripper_obs_mode}",
        f"action_mode={args.gripper_action_mode}",
        f"stroke={args.gripper_stroke:g}m",
        f"raw_range=[{gripper_raw_min:.4f}, {gripper_raw_max:.4f}]",
        f"raw_range_source={gripper_range_source}",
        f"command_interval={args.gripper_command_interval:g}s",
    )
    print(
        "Camera preprocess:",
        "BGR->RGB center-square 224x224 like gello convert_zarr,"
        f" then resize to {image_shape[2]}x{image_shape[1]}",
    )

    session = requests.Session()
    gripper_fallback = gripper_raw_max if args.gripper_obs_mode == "open_ratio" else args.gripper_stroke
    try:
        match = None
        if args.match_dataset is not None:
            match = load_match_episode(args.match_dataset, args.match_episode)
            print(
                "Match dataset:",
                args.match_dataset,
                f"episode={args.match_episode}/{match['episode_count'] - 1}",
                f"rows=[{match['episode_start']}, {match['episode_end']})",
            )
            print("Match pose source:", match["pose_source"])
            if match["gripper"] is not None:
                gripper_fallback = match["gripper"]
                print("Match gripper source:", match["gripper_source"], f"value={gripper_fallback:.4f}")
            if args.execute and not args.no_move_to_start:
                print("Moving to matched start pose...")
                move_to_start(
                    session=session,
                    base_url=base_url,
                    target_pose=match["start_pose"],
                    duration=args.move_to_start_duration,
                    frequency=control_fps,
                    timeout=args.http_timeout,
                )
                if args.enable_gripper and match["gripper"] is not None:
                    post_gripper(
                        session=session,
                        base_url=base_url,
                        width=gripper_action_to_width(
                            match["gripper"],
                            args.gripper_action_mode,
                            args.gripper_stroke,
                            gripper_raw_min,
                            gripper_raw_max,
                        ),
                        velocity=args.gripper_velocity,
                        force_limit=args.gripper_force,
                        timeout=args.gripper_http_timeout,
                    )

        with OpenCVCamera(
            args.camera_source,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            require_color=not args.allow_monochrome_camera,
            color_threshold=args.camera_color_threshold,
        ) as camera:
            print(
                "Camera opened:",
                camera.source,
                "resolution=",
                camera.actual_resolution(),
                f"color_score={camera.color_score:.3f}",
            )
            for _ in range(max(0, args.warmup_frames)):
                camera.read()

            stop_event = threading.Event()
            obs_sampler = ObservationSampler(
                camera=camera,
                base_url=base_url,
                timeout=args.http_timeout,
                image_shape=image_shape,
                gripper_fallback=gripper_fallback,
                gripper_obs_mode=args.gripper_obs_mode,
                gripper_stroke=args.gripper_stroke,
                gripper_raw_min=gripper_raw_min,
                gripper_raw_max=gripper_raw_max,
                sample_fps=obs_sample_fps,
                history_len=history_len,
                stop_event=stop_event,
            )
            obs_sampler.start()
            while obs_sampler.get_history(n_obs_steps, obs_downsample_steps) is None:
                if obs_sampler.error is not None:
                    stop_event.set()
                    obs_sampler.join()
                    raise RuntimeError("observation sampler failed") from obs_sampler.error
                time.sleep(0.01)

            if args.save_camera_debug:
                latest_obs = obs_sampler.latest()
                if latest_obs is None:
                    raise RuntimeError("No observation available for camera debug save")
                save_rgb(args.save_camera_debug, latest_obs[rgb_key])
                print("Saved camera debug frame:", args.save_camera_debug)

            action_thread = ActionCommandThread(
                tcp_buffer=tcp_buffer,
                gripper_buffer=gripper_buffer,
                base_url=base_url,
                control_fps=control_fps,
                execute=args.execute,
                enable_gripper=args.enable_gripper,
                gripper_stroke=args.gripper_stroke,
                gripper_velocity=args.gripper_velocity,
                gripper_force=args.gripper_force,
                gripper_threshold=args.gripper_threshold,
                timeout=args.http_timeout,
                stop_event=stop_event,
                verbose=args.verbose,
                tcp_pos_clip_range=tcp_pos_clip_range,
                gripper_action_mode=args.gripper_action_mode,
                gripper_raw_min=gripper_raw_min,
                gripper_raw_max=gripper_raw_max,
                gripper_timeout=args.gripper_http_timeout,
                gripper_command_interval=args.gripper_command_interval,
                tcp_target_duration=tcp_target_duration,
            )
            action_thread.start()

            start_time = time.monotonic()
            step_count = 0
            debug_frame_idx = 0
            try:
                while time.monotonic() - start_time < args.max_duration:
                    loop_start = time.monotonic()
                    if obs_sampler.error is not None:
                        raise RuntimeError("observation sampler failed") from obs_sampler.error
                    latest_obs = obs_sampler.latest()
                    if latest_obs is None:
                        precise_wait(loop_start + 1.0 / inference_fps, time_func=time.monotonic)
                        continue
                    if args.save_camera_debug_dir is not None:
                        save_every = max(1, int(args.save_camera_debug_every))
                        if debug_frame_idx % save_every == 0:
                            debug_path = Path(args.save_camera_debug_dir) / (
                                f"frame_{debug_frame_idx:04d}_step_{step_count:06d}.png"
                            )
                            save_rgb(debug_path, latest_obs[rgb_key])
                            print("Saved camera debug frame:", debug_path)
                        debug_frame_idx += 1
                    history = obs_sampler.get_history(n_obs_steps, obs_downsample_steps)
                    if history is None:
                        precise_wait(loop_start + 1.0 / inference_fps, time_func=time.monotonic)
                        continue

                    obs_dict = build_obs_tensor(history, shape_meta, rgb_key, policy.device)
                    with torch.no_grad():
                        action_dict = policy.predict_action(obs_dict)
                    action_all = action_dict["action"][0].detach().cpu().numpy()
                    if action_all.ndim != 2 or action_all.shape[-1] != 10:
                        raise RuntimeError(f"Expected policy action Tx10, got {action_all.shape}")

                    if step_count % tcp_update_interval == 0:
                        tcp_action = action_all[latency_step:, :9]
                        if len(tcp_action) > 0:
                            tcp_buffer.add_action(tcp_action, step_count)
                    if step_count % gripper_update_interval == 0:
                        gripper_action = action_all[gripper_latency_step:, 9:]
                        if len(gripper_action) > 0:
                            gripper_buffer.add_action(gripper_action, step_count)

                    if args.verbose:
                        print(
                            f"inference step={step_count}",
                            f"action_shape={action_all.shape}",
                            f"first_xyz={np.array2string(action_all[0, :3], precision=4)}",
                            f"first_gripper={action_all[0, 9]:.4f}",
                        )

                    if action_thread.error is not None:
                        raise RuntimeError("action thread failed") from action_thread.error
                    step_count += steps_per_inference
                    precise_wait(loop_start + 1.0 / inference_fps, time_func=time.monotonic)
            except KeyboardInterrupt:
                print("Interrupted; stopping eval.")
            finally:
                stop_event.set()
                action_thread.join()
                obs_sampler.join()
                if action_thread.error is not None:
                    raise RuntimeError("action thread failed") from action_thread.error
                if obs_sampler.error is not None:
                    raise RuntimeError("observation sampler failed") from obs_sampler.error
    finally:
        session.close()


if __name__ == "__main__":
    main()
