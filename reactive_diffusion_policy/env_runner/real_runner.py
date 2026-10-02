import threading
import time
import os.path as osp
import hydra
import numpy as np
import torch
import tqdm
import zarr
from loguru import logger
from typing import Dict, Tuple, Union, Optional
from scipy.spatial.transform import Rotation as ScipyRotation
import transforms3d as t3d
import py_cli_interaction
from omegaconf import DictConfig, ListConfig
from reactive_diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from reactive_diffusion_policy.common.pytorch_util import dict_apply
from reactive_diffusion_policy.common.precise_sleep import precise_sleep
from reactive_diffusion_policy.real_world.real_inference_util import (
    get_real_obs_dict)
from reactive_diffusion_policy.real_world.real_world_transforms import RealWorldTransforms
from reactive_diffusion_policy.common.space_utils import ortho6d_to_rotation_matrix
from reactive_diffusion_policy.common.space_utils import (
    pose_3d_9d_to_homo_matrix_batch,
    homo_matrix_to_pose_9d_batch,
)
from reactive_diffusion_policy.common.ensemble import EnsembleBuffer
from reactive_diffusion_policy.common.action_utils import (
    get_gripper_action_indices,
    interpolate_actions_with_ratio,
    relative_gripper_actions_to_absolute_actions,
    relative_actions_to_absolute_actions,
    absolute_actions_to_relative_actions,
    get_inter_gripper_actions
)
import requests

try:
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    _RCLPY_IMPORT_ERROR = None
except Exception as exc:
    rclpy = None
    MultiThreadedExecutor = None
    _RCLPY_IMPORT_ERROR = exc


import os
import psutil
from copy import deepcopy

# add this to prevent assigning too may threads when using numpy
os.environ["OPENBLAS_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["NUMEXPR_NUM_THREADS"] = "12"
os.environ["OMP_NUM_THREADS"] = "12"

import cv2
# add this to prevent assigning too may threads when using open-cv
cv2.setNumThreads(12)

def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(value)

# Get the total number of CPU cores
total_cores = psutil.cpu_count()
# Define the number of cores you want to bind to
num_cores_to_bind = 10
# Calculate the indices of the first ten cores
# Ensure the number of cores to bind does not exceed the total number of cores
cores_to_bind = set(range(min(num_cores_to_bind, total_cores)))
# Set CPU affinity for the current process to the first ten cores
os.sched_setaffinity(0, cores_to_bind)

class RealRunner:
    def __init__(self,
                 output_dir: str,
                 transform_params: DictConfig,
                 env_params: DictConfig,
                 shape_meta: DictConfig,
                 tcp_ensemble_buffer_params: DictConfig,
                 gripper_ensemble_buffer_params: DictConfig,
                 latent_tcp_ensemble_buffer_params: DictConfig = None,
                 latent_gripper_ensemble_buffer_params: DictConfig = None,
                 use_latent_action_with_rnn_decoder: bool = False,
                 use_relative_action: bool = False,
                 use_relative_tcp_obs_for_relative_action: bool = True,
                 use_relative_gripper_action: bool = False,
                 contact_action_scale: float = 1.0,
                 contact_action_scale_mode: str = "fixed",
                 contact_action_scale_threshold: Optional[float] = None,
                 contact_action_scale_obs_key: str = "left_gripper1_marker_offset_emb",
                 contact_action_scale_dims: int = 12,
                 contact_action_scale_max_translation: Optional[float] = None,
                 contact_action_scale_magnet_divisor: float = 300.0,
                 contact_action_scale_log_every: int = 6,
                 action_interpolation_ratio: int = 1,
                 eval_episodes=10,
                 max_duration_time: float = 30,
                 tcp_action_update_interval: int = 6,
                 gripper_action_update_interval: int = 10,
                 tcp_pos_clip_range: ListConfig = ListConfig([[0.6, -0.4, 0.03], [1.0, 0.45, 0.4]]),
                 tcp_rot_clip_range: ListConfig = ListConfig([[-np.pi, 0., np.pi], [-np.pi, 0., np.pi]]),
                 tqdm_interval_sec = 5.0,
                 control_fps: float = 12,
                 inference_fps: float = 6,
                 latency_step: int = 0,
                 gripper_latency_step: Optional[int] = None,
                 n_obs_steps: int = 2,
                 obs_temporal_downsample_ratio: int = 2,
                 dataset_obs_temporal_downsample_ratio: int = 1,
                 downsample_extended_obs: bool = True,
                 enable_video_recording: bool = False,
                 vcamera_server_ip: Optional[Union[str, ListConfig]] = None,
                 vcamera_server_port: Optional[Union[int, ListConfig]] = None,
                 env_class: str = "reactive_diffusion_policy.env.real_bimanual.real_env.RealRobotEnvironment",
                 use_ros_executor: bool = True,
                 ask_reset_confirmation: bool = True,
                 open_gripper_on_start: bool = True,
                 start_gripper_width_mm: Optional[float] = None,
                 debug_policy_actions: bool = False,
                 debug_policy_action_every: int = 1,
                 policy_tcp_pose_obs_mode: str = "none",
                 policy_tcp_pose_obs_dataset_episode: Optional[int] = None,
                 policy_tcp_pose_obs_dataset_path: Optional[str] = None,
                 policy_relative_action_frame_mode: str = "none",
                 task_name=None,
                 ):
        self.task_name = task_name
        self.transforms = RealWorldTransforms(option=transform_params)
        self.shape_meta = dict(shape_meta)
        self.eval_episodes = eval_episodes

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys

        extended_rgb_keys = list()
        extended_lowdim_keys = list()
        extended_obs_shape_meta = shape_meta.get('extended_obs', dict())
        for key, attr in extended_obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                extended_rgb_keys.append(key)
            elif type == 'low_dim':
                extended_lowdim_keys.append(key)
        self.extended_rgb_keys = extended_rgb_keys
        self.extended_lowdim_keys = extended_lowdim_keys

        self.use_ros_executor = use_ros_executor
        self.ask_reset_confirmation = ask_reset_confirmation
        self.open_gripper_on_start = _as_bool(open_gripper_on_start)
        self.start_gripper_width_mm = (
            None if start_gripper_width_mm is None else float(start_gripper_width_mm)
        )
        if self.start_gripper_width_mm is not None and (
                not np.isfinite(self.start_gripper_width_mm)
                or self.start_gripper_width_mm < 0.0):
            raise ValueError(
                "start_gripper_width_mm must be finite and non-negative, "
                f"got {start_gripper_width_mm}"
            )
        self.policy_tcp_pose_obs_mode = str(policy_tcp_pose_obs_mode or "none").lower()
        valid_tcp_obs_modes = (
            "none",
            "dataset_tool_offset",
            "dataset_tool_rotation_offset",
            "dataset_base_offset",
            "dataset_xyz_offset",
        )
        if self.policy_tcp_pose_obs_mode not in valid_tcp_obs_modes:
            raise ValueError(
                "policy_tcp_pose_obs_mode must be one of "
                f"{valid_tcp_obs_modes}, got {policy_tcp_pose_obs_mode!r}"
            )
        self.policy_tcp_pose_obs_dataset_episode = (
            None
            if policy_tcp_pose_obs_dataset_episode is None
            else int(policy_tcp_pose_obs_dataset_episode)
        )
        self.policy_tcp_pose_obs_dataset_path = (
            str(policy_tcp_pose_obs_dataset_path)
            if policy_tcp_pose_obs_dataset_path
            else str(env_params.get("dataset_path", ""))
        )
        self.policy_tcp_pose_obs_default_episode = int(env_params.get("reset_episode", 0))
        self._policy_tcp_pose_obs_offset_mat = None
        self._policy_tcp_pose_obs_xyz_offset = None
        self._policy_tcp_pose_obs_dataset_start_pose9 = None
        self._policy_tcp_pose_obs_warned_missing = False
        self.policy_relative_action_frame_mode = str(policy_relative_action_frame_mode or "none").lower()
        valid_action_frame_modes = (
            "none",
            "dataset_tool_offset",
            "dataset_tool_rotation_offset",
        )
        if self.policy_relative_action_frame_mode not in valid_action_frame_modes:
            raise ValueError(
                "policy_relative_action_frame_mode must be one of "
                f"{valid_action_frame_modes}, got {policy_relative_action_frame_mode!r}"
            )
        self._policy_relative_action_frame_offset_mat = None
        self._policy_relative_action_frame_offset_inv = None
        self._policy_relative_action_frame_logged = False
        if self.use_ros_executor:
            if rclpy is None or MultiThreadedExecutor is None:
                raise RuntimeError(
                    "rclpy is required when use_ros_executor=True. "
                    "Set task.env_runner.use_ros_executor=False for non-ROS environments."
                ) from _RCLPY_IMPORT_ERROR
            rclpy.init(args=None)
        env_cls = hydra.utils.get_class(env_class)
        self.env = env_cls(transforms=self.transforms, **env_params)
        self._set_startup_gripper_state(settle_seconds=2.0)

        self.max_duration_time = max_duration_time
        self.tcp_action_update_interval = tcp_action_update_interval
        self.gripper_action_update_interval = gripper_action_update_interval
        self.tcp_pos_clip_range = tcp_pos_clip_range
        self.tcp_rot_clip_range = tcp_rot_clip_range
        self.tqdm_interval_sec = tqdm_interval_sec
        self.control_fps = control_fps
        self.control_interval_time = 1.0 / control_fps
        self.inference_fps = inference_fps
        self.inference_interval_time = 1.0 / inference_fps
        assert self.control_fps % self.inference_fps == 0
        self.latency_step = latency_step
        self.gripper_latency_step = gripper_latency_step if gripper_latency_step is not None else latency_step
        self.n_obs_steps = n_obs_steps
        self.obs_temporal_downsample_ratio = obs_temporal_downsample_ratio
        self.dataset_obs_temporal_downsample_ratio = dataset_obs_temporal_downsample_ratio
        self.downsample_extended_obs = downsample_extended_obs
        self.use_latent_action_with_rnn_decoder = use_latent_action_with_rnn_decoder
        if self.use_latent_action_with_rnn_decoder:
            assert latent_tcp_ensemble_buffer_params.ensemble_mode == 'new', "Only support new ensemble mode for latent action."
            assert latent_gripper_ensemble_buffer_params.ensemble_mode == 'new', "Only support new ensemble mode for latent action."
            self.tcp_ensemble_buffer = EnsembleBuffer(**latent_tcp_ensemble_buffer_params)
            self.gripper_ensemble_buffer = EnsembleBuffer(**latent_gripper_ensemble_buffer_params)
        else:
            self.tcp_ensemble_buffer = EnsembleBuffer(**tcp_ensemble_buffer_params)
            self.gripper_ensemble_buffer = EnsembleBuffer(**gripper_ensemble_buffer_params)
        self.use_relative_action = use_relative_action
        self.use_relative_tcp_obs_for_relative_action = use_relative_tcp_obs_for_relative_action
        self.use_relative_gripper_action = bool(use_relative_gripper_action)
        logger.info(
            "Policy action representation: "
            f"relative_tcp={self.use_relative_action}, "
            f"relative_gripper={self.use_relative_gripper_action}"
        )
        self.contact_action_scale = float(contact_action_scale)
        self.contact_action_scale_mode = str(contact_action_scale_mode).lower()
        if self.contact_action_scale_mode == "dynamic":
            self.contact_action_scale_mode = "magnet"
        if self.contact_action_scale_mode not in ("fixed", "magnet"):
            raise ValueError(
                "contact_action_scale_mode must be 'fixed' or 'magnet', "
                f"got {contact_action_scale_mode!r}"
            )
        self.contact_action_scale_threshold = (
            None
            if contact_action_scale_threshold is None
            else float(contact_action_scale_threshold)
        )
        self.contact_action_scale_obs_key = str(contact_action_scale_obs_key)
        self.contact_action_scale_dims = int(contact_action_scale_dims)
        self.contact_action_scale_max_translation = (
            None
            if contact_action_scale_max_translation is None
            else float(contact_action_scale_max_translation)
        )
        self.contact_action_scale_magnet_divisor = float(contact_action_scale_magnet_divisor)
        if self.contact_action_scale_magnet_divisor <= 0:
            raise ValueError("contact_action_scale_magnet_divisor must be positive")
        self.contact_action_scale_log_every = max(1, int(contact_action_scale_log_every))
        self._contact_action_scale_active = None
        self._contact_action_scale_missing_warned = False
        self._contact_action_scale_apply_count = 0
        self.action_interpolation_ratio = action_interpolation_ratio

        self.enable_video_recording = enable_video_recording
        if enable_video_recording:
            assert isinstance(vcamera_server_ip, str) and isinstance(vcamera_server_port, int) or \
                     isinstance(vcamera_server_ip, ListConfig) and isinstance(vcamera_server_port, ListConfig), \
                "vcamera_server_ip and vcamera_server_port should be a string or ListConfig."
        if isinstance(vcamera_server_ip, str):
            vcamera_server_ip_list = [vcamera_server_ip]
            vcamera_server_port_list = [vcamera_server_port]
        elif isinstance(vcamera_server_ip, ListConfig):
            vcamera_server_ip_list = list(vcamera_server_ip)
            vcamera_server_port_list = list(vcamera_server_port)
        else:
            vcamera_server_ip_list = []
            vcamera_server_port_list = []
        self.vcamera_server_ip_list = vcamera_server_ip_list
        self.vcamera_server_port_list = vcamera_server_port_list
        self.video_dir = osp.join(output_dir, 'videos')

        self.stop_event = threading.Event()
        self.action_thread_error = None
        self.session = requests.Session()
        self.debug_policy_actions = bool(debug_policy_actions)
        self.debug_policy_action_every = max(1, int(debug_policy_action_every))
        self._empty_action_warn_count = 0

    def _set_startup_gripper_state(self, settle_seconds: float):
        if self.start_gripper_width_mm is not None:
            if not hasattr(self.env, "send_gripper_width_m_direct"):
                raise RuntimeError(
                    "START_GRIPPER_WIDTH_MM requires an environment with "
                    "send_gripper_width_m_direct()"
                )
            logger.info(
                "Setting startup gripper width to "
                f"{self.start_gripper_width_mm:.3f} mm before policy execution."
            )
            self.env.send_gripper_width_m_direct(self.start_gripper_width_mm / 1000.0)
            time.sleep(max(0.0, float(settle_seconds)))
        elif self.open_gripper_on_start:
            self.env.send_gripper_command_direct(
                self.env.max_gripper_width,
                self.env.max_gripper_width,
            )
            time.sleep(max(0.0, float(settle_seconds)))
        elif hasattr(self.env, "sync_commanded_gripper_state_to_measured"):
            logger.info("Skipping startup gripper command.")
            self.env.sync_commanded_gripper_state_to_measured()

    @staticmethod
    def spin_executor(executor):
        executor.spin()

    @staticmethod
    def _open_zarr_readonly(path):
        path = os.path.expanduser(str(path))
        if path.endswith(".zip"):
            store = zarr.ZipStore(path, mode="r")
            return zarr.group(store), store
        return zarr.open(path, mode="r"), None

    @staticmethod
    def _dataset_candidates(dataset_path):
        if not dataset_path:
            return []
        dataset_path = os.path.expanduser(str(dataset_path))
        candidates = []
        if os.path.isdir(dataset_path):
            replay_buffer_path = os.path.join(dataset_path, "replay_buffer.zarr")
            if os.path.exists(replay_buffer_path):
                candidates.append(replay_buffer_path)
        candidates.append(dataset_path)
        result = []
        seen = set()
        for candidate in candidates:
            if candidate not in seen:
                result.append(candidate)
                seen.add(candidate)
        return result

    @staticmethod
    def _episode_slice_from_ends(episode_ends, episode_idx):
        if len(episode_ends) == 0:
            raise ValueError("Dataset has no episodes.")
        if episode_idx < 0 or episode_idx >= len(episode_ends):
            raise ValueError(
                f"episode_idx must be in [0, {len(episode_ends) - 1}], got {episode_idx}"
            )
        start = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
        end = int(episode_ends[episode_idx])
        if end <= start:
            raise ValueError(f"Episode {episode_idx} is empty: rows=[{start}, {end})")
        return slice(start, end), start, end

    @staticmethod
    def _pose_array_to_pose9(pose):
        pose = np.asarray(pose, dtype=np.float64).reshape(-1)
        if pose.shape[0] == 9:
            return pose
        if pose.shape[0] == 7:
            quat_wxyz = pose[3:7]
            quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
            rot_mat = ScipyRotation.from_quat(quat_xyzw).as_matrix()
            return np.concatenate([pose[:3], rot_mat[:, :2].T.reshape(-1)])
        if pose.shape[0] == 6:
            rot_mat = ScipyRotation.from_rotvec(pose[3:6]).as_matrix()
            return np.concatenate([pose[:3], rot_mat[:, :2].T.reshape(-1)])
        if pose.shape[0] == 3:
            return pose
        raise ValueError(f"Unsupported TCP pose shape {pose.shape}")

    @staticmethod
    def _pose9_summary(pose):
        pose = np.asarray(pose, dtype=np.float64).reshape(-1)
        return np.round(pose[:3], 4).tolist()

    def _load_policy_tcp_pose_dataset_start(self):
        if self._policy_tcp_pose_obs_dataset_start_pose9 is not None:
            return self._policy_tcp_pose_obs_dataset_start_pose9

        dataset_path = self.policy_tcp_pose_obs_dataset_path
        episode_idx = (
            self.policy_tcp_pose_obs_dataset_episode
            if self.policy_tcp_pose_obs_dataset_episode is not None
            else self.policy_tcp_pose_obs_default_episode
        )
        for candidate in self._dataset_candidates(dataset_path):
            if not os.path.exists(candidate):
                continue
            root, store = self._open_zarr_readonly(candidate)
            try:
                if "data" not in root or "meta" not in root or "episode_ends" not in root["meta"]:
                    continue
                data = root["data"]
                episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
                episode_slice, row_start, row_end = self._episode_slice_from_ends(
                    episode_ends, episode_idx
                )

                if "left_robot_tcp_pose" in data:
                    pose = np.asarray(data["left_robot_tcp_pose"][episode_slice][0], dtype=np.float64)
                    pose9 = self._pose_array_to_pose9(pose)
                    source = f"left_robot_tcp_pose[{pose.shape[-1]}]"
                elif "robot0_eef_pos" in data and "robot0_eef_rot_axis_angle" in data:
                    pos = np.asarray(data["robot0_eef_pos"][episode_slice][0], dtype=np.float64)
                    rotvec = np.asarray(
                        data["robot0_eef_rot_axis_angle"][episode_slice][0],
                        dtype=np.float64,
                    )
                    pose9 = self._pose_array_to_pose9(np.concatenate([pos, rotvec]))
                    source = "robot0_eef_pos + robot0_eef_rot_axis_angle"
                else:
                    continue

                self._policy_tcp_pose_obs_dataset_start_pose9 = {
                    "pose9": pose9.astype(np.float64),
                    "path": candidate,
                    "episode_idx": int(episode_idx),
                    "episode_count": len(episode_ends),
                    "row_start": row_start,
                    "row_end": row_end,
                    "source": source,
                }
                return self._policy_tcp_pose_obs_dataset_start_pose9
            finally:
                if store is not None:
                    store.close()

        raise FileNotFoundError(
            "Could not load dataset TCP start pose for policy obs alignment from "
            f"dataset_path={dataset_path!r}"
        )

    def _reset_policy_tcp_pose_obs_alignment(self):
        self._policy_tcp_pose_obs_offset_mat = None
        self._policy_tcp_pose_obs_xyz_offset = None
        self._policy_tcp_pose_obs_warned_missing = False
        self._policy_relative_action_frame_offset_mat = None
        self._policy_relative_action_frame_offset_inv = None
        self._policy_relative_action_frame_logged = False

    def _ensure_policy_relative_action_frame_alignment(self, base_absolute_action):
        if self.policy_relative_action_frame_mode == "none":
            return
        if self.policy_relative_action_frame_mode not in (
                "dataset_tool_offset",
                "dataset_tool_rotation_offset"):
            raise ValueError(
                f"Unsupported policy_relative_action_frame_mode={self.policy_relative_action_frame_mode!r}"
            )
        if self._policy_relative_action_frame_offset_mat is not None:
            return

        base_absolute_action = np.asarray(base_absolute_action, dtype=np.float64).reshape(-1)
        if base_absolute_action.shape[0] < 9:
            raise ValueError(
                "policy relative action frame alignment requires a 9D left TCP base pose, "
                f"got shape {base_absolute_action.shape}"
            )
        dataset_info = self._load_policy_tcp_pose_dataset_start()
        dataset_pose = dataset_info["pose9"]
        real_start_pose = self._pose_array_to_pose9(base_absolute_action[:9])
        dataset_start_mat = pose_3d_9d_to_homo_matrix_batch(dataset_pose[None, :])[0]
        real_start_mat = pose_3d_9d_to_homo_matrix_batch(real_start_pose[None, :])[0]
        offset_mat = np.linalg.inv(real_start_mat) @ dataset_start_mat
        if self.policy_relative_action_frame_mode == "dataset_tool_rotation_offset":
            rotation_only_offset = np.eye(4, dtype=np.float64)
            rotation_only_offset[:3, :3] = offset_mat[:3, :3]
            offset_mat = rotation_only_offset
        self._policy_relative_action_frame_offset_mat = offset_mat
        self._policy_relative_action_frame_offset_inv = np.linalg.inv(
            self._policy_relative_action_frame_offset_mat
        )
        logger.info(
            "Policy relative action frame alignment enabled: "
            f"mode={self.policy_relative_action_frame_mode}, "
            f"dataset={dataset_info['path']}, "
            f"episode={dataset_info['episode_idx']}/{dataset_info['episode_count']}, "
            f"rows=[{dataset_info['row_start']}, {dataset_info['row_end']}), "
            f"source={dataset_info['source']}, "
            f"real_start_xyz={self._pose9_summary(real_start_pose)}, "
            f"dataset_start_xyz={self._pose9_summary(dataset_pose)}, "
            f"offset_xyz={np.round(self._policy_relative_action_frame_offset_mat[:3, 3], 4).tolist()}, "
            f"offset_angle_deg={np.rad2deg(ScipyRotation.from_matrix(self._policy_relative_action_frame_offset_mat[:3, :3]).magnitude()):.2f}"
        )

    def _maybe_transform_relative_action_frame(self, action: np.ndarray, base_absolute_action: np.ndarray):
        if self.policy_relative_action_frame_mode == "none":
            return action
        if not self.use_relative_action:
            return action

        self._ensure_policy_relative_action_frame_alignment(base_absolute_action)
        action = action.copy()
        action_dim = action.shape[-1]
        if action_dim in (3, 4):
            tcp_slices = [slice(0, 3)]
        elif action_dim in (9, 10):
            tcp_slices = [slice(0, 9)]
        else:
            raise NotImplementedError(
                "policy relative action frame alignment currently supports single-arm "
                f"3D/9D TCP actions only, got action_dim={action_dim}"
            )

        offset = self._policy_relative_action_frame_offset_mat
        offset_inv = self._policy_relative_action_frame_offset_inv
        for tcp_slice in tcp_slices:
            tcp_dim = tcp_slice.stop - tcp_slice.start
            rel_mats = pose_3d_9d_to_homo_matrix_batch(action[:, tcp_slice])
            aligned_mats = offset[None, :, :] @ rel_mats @ offset_inv[None, :, :]
            action[:, tcp_slice] = homo_matrix_to_pose_9d_batch(aligned_mats)[:, :tcp_dim]

        if not self._policy_relative_action_frame_logged:
            first_before = np.asarray(base_absolute_action, dtype=np.float64).reshape(-1)[:3]
            logger.info(
                "Policy relative action frame transform is active. "
                "Relative TCP actions are conjugated by the dataset tool offset before "
                f"conversion to robot absolute targets; real_base_xyz={np.round(first_before, 4).tolist()}."
            )
            self._policy_relative_action_frame_logged = True
        return action

    def _ensure_policy_tcp_pose_obs_alignment(self, real_tcp_pose_seq):
        if self.policy_tcp_pose_obs_mode == "none":
            return

        real_tcp_pose_seq = np.asarray(real_tcp_pose_seq, dtype=np.float64)
        if real_tcp_pose_seq.ndim != 2 or real_tcp_pose_seq.shape[0] == 0:
            raise ValueError(
                "left_robot_tcp_pose observation must have shape (T, D) for policy obs alignment, "
                f"got {real_tcp_pose_seq.shape}"
            )
        obs_dim = real_tcp_pose_seq.shape[1]
        if obs_dim not in (3, 9):
            raise ValueError(
                "policy TCP pose obs alignment only supports 3D or 9D TCP pose, "
                f"got D={obs_dim}"
            )

        if (
            self._policy_tcp_pose_obs_offset_mat is not None
            or self._policy_tcp_pose_obs_xyz_offset is not None
        ):
            return

        dataset_info = self._load_policy_tcp_pose_dataset_start()
        dataset_pose = dataset_info["pose9"]
        real_start_pose = self._pose_array_to_pose9(real_tcp_pose_seq[-1])

        if self.policy_tcp_pose_obs_mode == "dataset_xyz_offset" or obs_dim == 3:
            self._policy_tcp_pose_obs_xyz_offset = dataset_pose[:3] - real_start_pose[:3]
            fake_start_xyz = real_start_pose[:3] + self._policy_tcp_pose_obs_xyz_offset
            offset_text = f"xyz_offset={np.round(self._policy_tcp_pose_obs_xyz_offset, 4).tolist()}"
        else:
            if dataset_pose.shape[0] != 9 or real_start_pose.shape[0] != 9:
                raise ValueError(
                    f"{self.policy_tcp_pose_obs_mode} requires 9D TCP poses, "
                    f"got dataset={dataset_pose.shape[0]}, real={real_start_pose.shape[0]}"
                )
            dataset_start_mat = pose_3d_9d_to_homo_matrix_batch(dataset_pose[None, :])[0]
            real_start_mat = pose_3d_9d_to_homo_matrix_batch(real_start_pose[None, :])[0]
            if self.policy_tcp_pose_obs_mode in (
                    "dataset_tool_offset",
                    "dataset_tool_rotation_offset"):
                self._policy_tcp_pose_obs_offset_mat = np.linalg.inv(real_start_mat) @ dataset_start_mat
                if self.policy_tcp_pose_obs_mode == "dataset_tool_rotation_offset":
                    rotation_only_offset = np.eye(4, dtype=np.float64)
                    rotation_only_offset[:3, :3] = self._policy_tcp_pose_obs_offset_mat[:3, :3]
                    self._policy_tcp_pose_obs_offset_mat = rotation_only_offset
            elif self.policy_tcp_pose_obs_mode == "dataset_base_offset":
                self._policy_tcp_pose_obs_offset_mat = dataset_start_mat @ np.linalg.inv(real_start_mat)
            else:
                raise ValueError(f"Unsupported policy_tcp_pose_obs_mode={self.policy_tcp_pose_obs_mode!r}")
            fake_start_mat = (
                real_start_mat @ self._policy_tcp_pose_obs_offset_mat
                if self.policy_tcp_pose_obs_mode in (
                    "dataset_tool_offset",
                    "dataset_tool_rotation_offset",
                )
                else self._policy_tcp_pose_obs_offset_mat @ real_start_mat
            )
            fake_start_xyz = fake_start_mat[:3, 3]
            offset_text = f"offset_xyz={np.round(self._policy_tcp_pose_obs_offset_mat[:3, 3], 4).tolist()}"

        if self.use_relative_action and self.use_relative_tcp_obs_for_relative_action:
            logger.info(
                "Policy TCP pose obs alignment runs before relative TCP obs conversion; "
                "robot action base remains the real measured TCP pose."
            )
        logger.info(
            "Policy TCP pose obs alignment enabled: "
            f"mode={self.policy_tcp_pose_obs_mode}, "
            f"dataset={dataset_info['path']}, "
            f"episode={dataset_info['episode_idx']}/{dataset_info['episode_count']}, "
            f"rows=[{dataset_info['row_start']}, {dataset_info['row_end']}), "
            f"source={dataset_info['source']}, "
            f"real_start_xyz={self._pose9_summary(real_start_pose)}, "
            f"dataset_start_xyz={self._pose9_summary(dataset_pose)}, "
            f"fake_start_xyz={np.round(fake_start_xyz, 4).tolist()}, "
            f"{offset_text}"
        )

    def _maybe_adjust_policy_tcp_pose_obs(
            self,
            policy_obs_dict: Dict,
            real_obs_dict: Dict,
            warn_if_missing: bool = True):
        if self.policy_tcp_pose_obs_mode == "none":
            return policy_obs_dict
        key = "left_robot_tcp_pose"
        if key not in policy_obs_dict or key not in real_obs_dict:
            if warn_if_missing and not self._policy_tcp_pose_obs_warned_missing:
                logger.warning(
                    "policy_tcp_pose_obs_mode is enabled but left_robot_tcp_pose is missing; "
                    "policy observation is unchanged."
                )
                self._policy_tcp_pose_obs_warned_missing = True
            return policy_obs_dict

        real_seq = np.asarray(real_obs_dict[key], dtype=np.float64)
        self._ensure_policy_tcp_pose_obs_alignment(real_seq)
        adjusted = np.asarray(policy_obs_dict[key]).copy()
        obs_dim = min(adjusted.shape[1], real_seq.shape[1])

        if self._policy_tcp_pose_obs_xyz_offset is not None or obs_dim == 3:
            xyz_offset = self._policy_tcp_pose_obs_xyz_offset
            if xyz_offset is None:
                xyz_offset = np.zeros(3, dtype=np.float64)
            adjusted[:, :3] = real_seq[:, :3] + xyz_offset
        else:
            real_pose9_seq = real_seq[:, :9]
            real_mats = pose_3d_9d_to_homo_matrix_batch(real_pose9_seq)
            if self.policy_tcp_pose_obs_mode in (
                    "dataset_tool_offset",
                    "dataset_tool_rotation_offset"):
                fake_mats = real_mats @ self._policy_tcp_pose_obs_offset_mat
            elif self.policy_tcp_pose_obs_mode == "dataset_base_offset":
                fake_mats = self._policy_tcp_pose_obs_offset_mat[None, :, :] @ real_mats
            else:
                raise ValueError(f"Unsupported policy_tcp_pose_obs_mode={self.policy_tcp_pose_obs_mode!r}")
            adjusted[:, :9] = homo_matrix_to_pose_9d_batch(fake_mats)

        policy_obs_dict[key] = adjusted.astype(np.asarray(policy_obs_dict[key]).dtype, copy=False)
        return policy_obs_dict

    def pre_process_obs(self, obs_dict: Dict) -> Tuple[Dict, Dict]:
        obs_dict = deepcopy(obs_dict)

        for key in self.lowdim_keys:
            if "wrt" not in key:
                obs_dict[key] = obs_dict[key][:, :self.shape_meta['obs'][key]['shape'][0]]

        # inter-gripper relative action
        obs_dict.update(get_inter_gripper_actions(obs_dict, self.lowdim_keys, self.transforms))
        for key in self.lowdim_keys:
            obs_dict[key] = obs_dict[key][:, :self.shape_meta['obs'][key]['shape'][0]]

        absolute_obs_dict = dict()
        for key in self.lowdim_keys:
            absolute_obs_dict[key] = obs_dict[key].copy()

        # convert absolute action to relative action
        if self.use_relative_action and self.use_relative_tcp_obs_for_relative_action:
            for key in self.lowdim_keys:
                if 'robot_tcp_pose' in key and 'wrt' not in key:
                    base_absolute_action = obs_dict[key][-1].copy()
                    obs_dict[key] = absolute_actions_to_relative_actions(obs_dict[key], base_absolute_action=base_absolute_action)

        return obs_dict, absolute_obs_dict

    def pre_process_extended_obs(self, extended_obs_dict: Dict) -> Tuple[Dict, Dict]:
        extended_obs_dict = deepcopy(extended_obs_dict)

        absolute_extended_obs_dict = dict()
        for key in self.extended_lowdim_keys:
            extended_obs_dict[key] = extended_obs_dict[key][:, :self.shape_meta['extended_obs'][key]['shape'][0]]
            absolute_extended_obs_dict[key] = extended_obs_dict[key].copy()

        # convert absolute action to relative action
        if self.use_relative_action and self.use_relative_tcp_obs_for_relative_action:
            for key in self.extended_lowdim_keys:
                if 'robot_tcp_pose' in key and 'wrt' not in key:
                    base_absolute_action = extended_obs_dict[key][-1].copy()
                    extended_obs_dict[key] = absolute_actions_to_relative_actions(extended_obs_dict[key], base_absolute_action=base_absolute_action)

        return extended_obs_dict, absolute_extended_obs_dict

    def _get_gripper_action_base(self, obs_dict: Dict) -> np.ndarray:
        action_dim = int(self.shape_meta['action']['shape'][0])
        gripper_count = get_gripper_action_indices(action_dim).size
        keys = ("left_robot_gripper_width", "right_robot_gripper_width")[:gripper_count]
        missing = [key for key in keys if key not in obs_dict]
        if missing:
            raise KeyError(
                "use_relative_gripper_action requires absolute gripper observations; "
                f"missing {missing}"
            )
        return np.concatenate([
            np.asarray(obs_dict[key][-1], dtype=np.float32).reshape(-1)
            for key in keys
        ])

    def _maybe_make_gripper_action_absolute(
            self,
            action: np.ndarray,
            absolute_obs_dict: Dict) -> np.ndarray:
        if not self.use_relative_gripper_action:
            return action
        base = self._get_gripper_action_base(absolute_obs_dict)
        return relative_gripper_actions_to_absolute_actions(action, base)

    def post_process_action(self, action: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Post-process the action before sending to the robot
        """
        assert len(action.shape) == 2  # (action_steps, d_a)
        if self.env.data_processing_manager.use_6d_rotation:
            if action.shape[-1] == 4 or action.shape[-1] == 8:
                # convert to 6D pose
                left_trans_batch = action[:, :3]  # (action_steps, 3)
                # we use default euler angles as 0
                left_euler_batch = np.zeros_like(left_trans_batch)
                left_action_6d = np.concatenate([left_trans_batch, left_euler_batch], axis=1)  # (action_steps, 6)
                if action.shape[-1] == 8:
                    right_trans_batch = action[:, 3:6]  # (action_steps, 3)
                    right_euler_batch = np.zeros_like(right_trans_batch)
                    right_action_6d = np.concatenate([right_trans_batch, right_euler_batch], axis=1)
                else:
                    right_action_6d = None
            elif action.shape[-1] == 10 or action.shape[-1] == 20:
                # convert to 6D pose
                left_rot_mat_batch = ortho6d_to_rotation_matrix(action[:, 3:9])  # (action_steps, 3, 3)
                left_euler_batch = np.array([t3d.euler.mat2euler(rot_mat) for rot_mat in left_rot_mat_batch])  # (action_steps, 3)
                left_trans_batch = action[:, :3]  # (action_steps, 3)
                left_action_6d = np.concatenate([left_trans_batch, left_euler_batch], axis=1)  # (action_steps, 6)
                if action.shape[-1] == 20:
                    right_rot_mat_batch = ortho6d_to_rotation_matrix(action[:, 12:18])
                    right_euler_batch = np.array([t3d.euler.mat2euler(rot_mat) for rot_mat in right_rot_mat_batch])
                    right_trans_batch = action[:, 9:12]
                    right_action_6d = np.concatenate([right_trans_batch, right_euler_batch], axis=1)
                else:
                    right_action_6d = None
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        # clip action (x, y, z)
        left_action_6d[:, :3] = np.clip(left_action_6d[:, :3], np.array(self.tcp_pos_clip_range[0]), np.array(self.tcp_pos_clip_range[1]))
        if right_action_6d is not None:
            right_action_6d[:, :3] = np.clip(right_action_6d[:, :3], np.array(self.tcp_pos_clip_range[2]), np.array(self.tcp_pos_clip_range[3]))
        # clip action (r, p, y)
        left_action_6d[:, 3:] = np.clip(left_action_6d[:, 3:], np.array(self.tcp_rot_clip_range[0]), np.array(self.tcp_rot_clip_range[1]))
        if right_action_6d is not None:
            right_action_6d[:, 3:] = np.clip(right_action_6d[:, 3:], np.array(self.tcp_rot_clip_range[2]), np.array(self.tcp_rot_clip_range[3]))
        # add gripper action
        if action.shape[-1] == 4:
            left_action = np.concatenate([left_action_6d, action[:, 3][:, np.newaxis],
                                          np.zeros((action.shape[0], 1))], axis=1)
            right_action = None
        elif action.shape[-1] == 8:
            left_action = np.concatenate([left_action_6d, action[:, 6][:, np.newaxis],
                                          np.zeros((action.shape[0], 1))], axis=1)
            right_action = np.concatenate([right_action_6d, action[:, 7][:, np.newaxis],
                                           np.zeros((action.shape[0], 1))], axis=1)
        elif action.shape[-1] == 10:
            left_action = np.concatenate([left_action_6d, action[:, 9][:, np.newaxis],
                                          np.zeros((action.shape[0], 1))], axis=1)
            right_action = None
        elif action.shape[-1] == 20:
            left_action = np.concatenate([left_action_6d, action[:, 18][:, np.newaxis],
                                          np.zeros((action.shape[0], 1))], axis=1)
            right_action = np.concatenate([right_action_6d, action[:, 19][:, np.newaxis],
                                          np.zeros((action.shape[0], 1))], axis=1)

        else:
            raise NotImplementedError

        if right_action is None:
            right_action = left_action.copy()
            is_bimanual = False
        else:
            is_bimanual = True
        action_all = np.concatenate([left_action, right_action], axis=-1)
        return (action_all, is_bimanual)

    def _relative_tcp_translation_slices(self, action_dim: int):
        if action_dim in (3, 4, 9, 10):
            return [slice(0, 3)]
        if action_dim in (6, 8):
            return [slice(0, 3), slice(3, 6)]
        if action_dim in (18, 20):
            return [slice(0, 3), slice(9, 12)]
        raise NotImplementedError(f"Unsupported action dim for contact scaling: {action_dim}")

    def _clip_relative_translation_norms(self, action: np.ndarray):
        if self.contact_action_scale_max_translation is None:
            return action
        max_norm = float(self.contact_action_scale_max_translation)
        if max_norm <= 0:
            return action
        for dim_slice in self._relative_tcp_translation_slices(action.shape[-1]):
            xyz = action[:, dim_slice]
            norms = np.linalg.norm(xyz, axis=-1, keepdims=True)
            scale = np.minimum(1.0, max_norm / np.maximum(norms, 1e-9))
            action[:, dim_slice] = xyz * scale
        return action

    def _max_relative_translation_norm(self, action: np.ndarray):
        max_norm = 0.0
        for dim_slice in self._relative_tcp_translation_slices(action.shape[-1]):
            xyz = action[:, dim_slice]
            if xyz.size == 0:
                continue
            max_norm = max(max_norm, float(np.max(np.linalg.norm(xyz, axis=-1))))
        return max_norm

    def _contact_metric_from_obs(self, env_obs: Dict[str, np.ndarray]):
        values = env_obs.get(self.contact_action_scale_obs_key)
        if values is None:
            if not self._contact_action_scale_missing_warned:
                logger.warning(
                    "Contact action scaling requested but obs key is missing: "
                    f"{self.contact_action_scale_obs_key!r}"
                )
                self._contact_action_scale_missing_warned = True
            return None
        values = np.asarray(values, dtype=np.float32)
        if values.size == 0:
            return None
        latest = values.reshape(values.shape[0], -1)[-1]
        dim = min(max(self.contact_action_scale_dims, 1), latest.shape[0])
        return float(np.linalg.norm(latest[:dim]))

    def _maybe_scale_relative_action_on_contact(self, action: np.ndarray, env_obs: Dict[str, np.ndarray]):
        if self.contact_action_scale_mode == "fixed" and self.contact_action_scale == 1.0:
            return action
        if self.contact_action_scale_threshold is None:
            return action
        if not self.use_relative_action:
            if self._contact_action_scale_active is None:
                logger.warning("Contact action scaling is ignored because use_relative_action=False")
                self._contact_action_scale_active = False
            return action

        metric = self._contact_metric_from_obs(env_obs)
        if metric is None:
            return action

        active = metric >= self.contact_action_scale_threshold
        if self.contact_action_scale_mode == "magnet":
            effective_scale = 1.0 + metric / self.contact_action_scale_magnet_divisor
        else:
            effective_scale = self.contact_action_scale
        if active != self._contact_action_scale_active:
            logger.info(
                "Contact action scaling "
                f"{'enabled' if active else 'disabled'}: "
                f"metric={metric:.3f}, "
                f"threshold={self.contact_action_scale_threshold:.3f}, "
                f"scale={effective_scale:.3f}, "
                f"mode={self.contact_action_scale_mode}"
            )
            self._contact_action_scale_active = active
        if not active:
            return action

        scaled = action.copy()
        before_max_norm = self._max_relative_translation_norm(scaled)
        for dim_slice in self._relative_tcp_translation_slices(scaled.shape[-1]):
            scaled[:, dim_slice] *= effective_scale
        scaled = self._clip_relative_translation_norms(scaled)
        after_max_norm = self._max_relative_translation_norm(scaled)
        self._contact_action_scale_apply_count += 1
        if (
            self._contact_action_scale_apply_count == 1
            or self._contact_action_scale_apply_count % self.contact_action_scale_log_every == 0
        ):
            logger.info(
                "Contact action scaling active: "
                f"metric={metric:.3f}, "
                f"threshold={self.contact_action_scale_threshold:.3f}, "
                f"scale={effective_scale:.3f}, "
                f"mode={self.contact_action_scale_mode}, "
                f"max_rel_translation={before_max_norm:.4f}->{after_max_norm:.4f}m"
            )
        return scaled

    def action_command_thread(self, policy: Union[DiffusionUnetImagePolicy], stop_event):
        try:
            while not stop_event.is_set():
                start_time = time.time()
                # get step action from ensemble buffer
                tcp_step_action = self.tcp_ensemble_buffer.get_action()
                gripper_step_action = self.gripper_ensemble_buffer.get_action()
                if tcp_step_action is None or gripper_step_action is None:  # no action in the buffer => no movement.
                    if self.debug_policy_actions:
                        self._empty_action_warn_count += 1
                        if self._empty_action_warn_count % self.debug_policy_action_every == 0:
                            action_step_count = getattr(self, "action_step_count", 0)
                            logger.info(
                                "Action thread has no command: "
                                f"step={action_step_count}, "
                                f"tcp_empty={tcp_step_action is None}, "
                                f"gripper_empty={gripper_step_action is None}"
                            )
                    cur_time = time.time()
                    precise_sleep(max(0., self.control_interval_time - (cur_time - start_time)))
                    logger.debug(f"Step: {self.action_step_count}, control_interval_time: {self.control_interval_time}, "
                                 f"cur_time-start_time: {cur_time - start_time}")
                    self.action_step_count += 1
                    continue
                self._empty_action_warn_count = 0

                if self.use_latent_action_with_rnn_decoder:
                    tcp_extended_obs_step = int(tcp_step_action[-1])
                    gripper_extended_obs_step = int(gripper_step_action[-1])
                    tcp_step_action = tcp_step_action[:-1]
                    gripper_step_action = gripper_step_action[:-1]

                    longer_extended_obs_step = max(tcp_extended_obs_step, gripper_extended_obs_step)
                    obs_temporal_downsample_ratio = self.obs_temporal_downsample_ratio if self.downsample_extended_obs else 1
                    extended_obs = self.env.get_obs(longer_extended_obs_step,
                                                        temporal_downsample_ratio= obs_temporal_downsample_ratio)

                    if self.use_relative_action:
                        action_dim = self.shape_meta['obs']['left_robot_tcp_pose']['shape'][0]
                        if 'right_robot_tcp_pose' in self.shape_meta['obs']:
                            action_dim += self.shape_meta['obs']['right_robot_tcp_pose']['shape'][0]
                        tcp_base_absolute_action = tcp_step_action[-action_dim:]
                        gripper_base_absolute_action = gripper_step_action[-action_dim:]
                        tcp_step_action = tcp_step_action[:-action_dim]
                        gripper_step_action = gripper_step_action[:-action_dim]

                    if self.use_relative_gripper_action:
                        gripper_action_dim = get_gripper_action_indices(
                            int(self.shape_meta['action']['shape'][0])
                        ).size
                        tcp_gripper_base_absolute_action = tcp_step_action[-gripper_action_dim:]
                        gripper_gripper_base_absolute_action = gripper_step_action[-gripper_action_dim:]
                        tcp_step_action = tcp_step_action[:-gripper_action_dim]
                        gripper_step_action = gripper_step_action[:-gripper_action_dim]

                    np_extended_real_obs_dict = dict(extended_obs)
                    np_extended_real_obs_dict = get_real_obs_dict(
                        env_obs=np_extended_real_obs_dict, shape_meta=self.shape_meta, is_extended_obs=True)
                    np_extended_policy_obs_dict = deepcopy(np_extended_real_obs_dict)
                    self._maybe_adjust_policy_tcp_pose_obs(
                        np_extended_policy_obs_dict,
                        np_extended_real_obs_dict,
                        warn_if_missing=False,
                    )
                    np_extended_obs_dict, _ = self.pre_process_extended_obs(np_extended_policy_obs_dict)
                    extended_obs_dict = dict_apply(np_extended_obs_dict, lambda x: torch.from_numpy(x).unsqueeze(0))

                    tcp_step_latent_action = torch.from_numpy(tcp_step_action.astype(np.float32)).unsqueeze(0)
                    gripper_step_latent_action = torch.from_numpy(gripper_step_action.astype(np.float32)).unsqueeze(0)

                    dataset_obs_temporal_downsample_ratio = self.dataset_obs_temporal_downsample_ratio
                    tcp_step_action = policy.predict_from_latent_action(tcp_step_latent_action, extended_obs_dict, tcp_extended_obs_step, dataset_obs_temporal_downsample_ratio)['action'][0].detach().cpu().numpy()
                    gripper_step_action = policy.predict_from_latent_action(gripper_step_latent_action, extended_obs_dict, gripper_extended_obs_step, dataset_obs_temporal_downsample_ratio)['action'][0].detach().cpu().numpy()
                    if (
                        hasattr(self.env, "set_predicted_magnet_future")
                        and getattr(self.env, "enable_policy_recording", False)
                    ):
                        try:
                            prediction_result = policy.predict_from_latent_action(
                                tcp_step_latent_action,
                                extended_obs_dict,
                                tcp_extended_obs_step,
                                dataset_obs_temporal_downsample_ratio,
                                extend_obs_pad_after=True,
                                return_predicted_extended_obs=True,
                            )
                            magnet_key = getattr(
                                self.env,
                                "magnet_tactile_key",
                                "left_gripper1_marker_offset_emb",
                            )
                            predicted = prediction_result.get("predicted_extended_obs", {}).get(magnet_key)
                            predicted_norm = prediction_result.get("predicted_normalized_extended_obs", {}).get(magnet_key)
                            self.env.set_predicted_magnet_future(
                                None if predicted is None else predicted[0].detach().cpu().numpy(),
                                None if predicted_norm is None else predicted_norm[0].detach().cpu().numpy(),
                            )
                        except Exception as exc:
                            logger.warning(f"Predicted magnet future update failed: {exc}")
                    if self.use_relative_action:
                        tcp_step_action = self._maybe_transform_relative_action_frame(
                            tcp_step_action,
                            tcp_base_absolute_action,
                        )
                        gripper_step_action = self._maybe_transform_relative_action_frame(
                            gripper_step_action,
                            gripper_base_absolute_action,
                        )
                        tcp_step_action = relative_actions_to_absolute_actions(tcp_step_action, tcp_base_absolute_action)
                        gripper_step_action = relative_actions_to_absolute_actions(gripper_step_action, gripper_base_absolute_action)
                    if self.use_relative_gripper_action:
                        tcp_step_action = relative_gripper_actions_to_absolute_actions(
                            tcp_step_action,
                            tcp_gripper_base_absolute_action,
                        )
                        gripper_step_action = relative_gripper_actions_to_absolute_actions(
                            gripper_step_action,
                            gripper_gripper_base_absolute_action,
                        )

                    if tcp_step_action.shape[-1] == 4: # (x, y, z, gripper_width)
                        tcp_len = 3
                    elif tcp_step_action.shape[-1] == 8: # (x_l, y_l, z_l, x_r, y_r, z_r, gripper_width_l, gripper_width_r)
                        tcp_len = 6
                    elif tcp_step_action.shape[-1] == 10: # (x, y, z, rx1, rx2, rx3, ry1, ry2, ry3)
                        tcp_len = 9
                    elif tcp_step_action.shape[-1] == 20: # (x_l, y_l, z_l, rotation_l, x_r, y_r, z_r, rotation_r, gripper_width_l, gripper_width_r)
                        tcp_len = 18
                    else:
                        raise NotImplementedError

                    if self.env.enable_exp_recording:
                        self.env.get_predicted_action(tcp_step_action[:, :tcp_len], type='partial_tcp')
                        self.env.get_predicted_action(gripper_step_action[:, tcp_len:], type='partial_gripper')

                        full_tcp_step_action = policy.predict_from_latent_action(tcp_step_latent_action, extended_obs_dict, tcp_extended_obs_step, dataset_obs_temporal_downsample_ratio, extend_obs_pad_after=True)['action'][0].detach().cpu().numpy()
                        full_gripper_step_action = policy.predict_from_latent_action(gripper_step_latent_action, extended_obs_dict, gripper_extended_obs_step, dataset_obs_temporal_downsample_ratio, extend_obs_pad_after=True)['action'][0].detach().cpu().numpy()
                        if self.use_relative_action:
                            full_tcp_step_action = self._maybe_transform_relative_action_frame(
                                full_tcp_step_action,
                                tcp_base_absolute_action,
                            )
                            full_gripper_step_action = self._maybe_transform_relative_action_frame(
                                full_gripper_step_action,
                                gripper_base_absolute_action,
                            )
                            full_tcp_step_action = relative_actions_to_absolute_actions(full_tcp_step_action, tcp_base_absolute_action)
                            full_gripper_step_action = relative_actions_to_absolute_actions(full_gripper_step_action, gripper_base_absolute_action)
                        if self.use_relative_gripper_action:
                            full_tcp_step_action = relative_gripper_actions_to_absolute_actions(
                                full_tcp_step_action,
                                tcp_gripper_base_absolute_action,
                            )
                            full_gripper_step_action = relative_gripper_actions_to_absolute_actions(
                                full_gripper_step_action,
                                gripper_gripper_base_absolute_action,
                            )
                        self.env.get_predicted_action(full_tcp_step_action[:, :tcp_len], type='full_tcp')
                        self.env.get_predicted_action(full_gripper_step_action[:, tcp_len:], type='full_gripper')

                    tcp_step_action = tcp_step_action[-1]
                    gripper_step_action = gripper_step_action[-1]

                    tcp_step_action = tcp_step_action[:tcp_len]
                    gripper_step_action = gripper_step_action[tcp_len:]

                combined_action = np.concatenate([tcp_step_action, gripper_step_action], axis=-1)
                # convert to 16-D robot action (TCP + gripper of both arms)
                # TODO: handle rotation in temporal ensemble buffer!
                step_action, is_bimanual = self.post_process_action(combined_action[np.newaxis, :])
                step_action = step_action.squeeze(0)

                # send action to the robot
                if self.debug_policy_actions and self.action_step_count % self.debug_policy_action_every == 0:
                    logger.info(
                        "Action command: "
                        f"step={self.action_step_count}, "
                        f"tcp_xyz={np.round(step_action[:3], 4).tolist()}, "
                        f"tcp_rpy={np.round(step_action[3:6], 4).tolist()}, "
                        f"gripper={float(step_action[6]):.4f}"
                    )
                self.env.execute_action(step_action, use_relative_action=False, is_bimanual=is_bimanual)

                cur_time = time.time()
                precise_sleep(max(0., self.control_interval_time - (cur_time - start_time)))
                self.action_step_count += 1
        except BaseException as e:
            self.action_thread_error = e
            stop_event.set()
            logger.exception(f"Action command thread failed: {e}")

    def start_record_video(self, video_path):
        for vcamera_server_ip, vcamera_server_port in zip(self.vcamera_server_ip_list, self.vcamera_server_port_list):
            response = self.session.post(f'http://{vcamera_server_ip}:{vcamera_server_port}/start_recording/{video_path}')
            if response.status_code == 200:
                logger.info(f"Start recording video to {video_path}")
            else:
                logger.error(f"Failed to start recording video to {video_path}")

    def stop_record_video(self):
        for vcamera_server_ip, vcamera_server_port in zip(self.vcamera_server_ip_list, self.vcamera_server_port_list):
            response = self.session.post(f'http://{vcamera_server_ip}:{vcamera_server_port}/stop_recording')
            if response.status_code == 200:
                logger.info(f"Stop recording video")
            else:
                logger.error(f"Failed to stop recording video")

    def run(self, policy: Union[DiffusionUnetImagePolicy]):
        if self.use_latent_action_with_rnn_decoder:
            assert policy.at.use_rnn_decoder, "Policy should use rnn decoder for latent action."
        else:
            assert not hasattr(policy, 'at') or not policy.at.use_rnn_decoder, "Policy should not use rnn decoder for action."

        device = policy.device

        try:
            spin_thread = None
            if self.use_ros_executor:
                executor = MultiThreadedExecutor()
                executor.add_node(self.env)
                spin_thread = threading.Thread(target=self.spin_executor, args=(executor,), daemon=True)
                spin_thread.start()

            time.sleep(2)
            for episode_idx in tqdm.tqdm(range(0, self.eval_episodes),
                                         desc=f"Eval for {self.task_name}",
                                         leave=False, mininterval=self.tqdm_interval_sec):
                logger.info(f"Start evaluation episode {episode_idx}")
                # ask user whether the environment resetting is done
                if self.ask_reset_confirmation:
                    reset_flag = py_cli_interaction.parse_cli_bool(
                        'Has the environment reset finished?', default_value=True)
                else:
                    reset_flag = True
                if not reset_flag:
                    logger.warning("Skip this episode.")
                    continue

                logger.info("Start episode rollout.")
                # start rollout
                self.env.reset()
                self._set_startup_gripper_state(settle_seconds=1.0)

                policy.reset()
                self.tcp_ensemble_buffer.clear()
                self.gripper_ensemble_buffer.clear()
                logger.debug("Reset environment and policy.")
                if hasattr(self.env, "prepare_policy_start"):
                    self.env.prepare_policy_start()
                self._reset_policy_tcp_pose_obs_alignment()

                if self.enable_video_recording:
                    video_path = os.path.join(self.video_dir, f'episode_{episode_idx}.mp4')
                    self.start_record_video(video_path)
                    logger.info(f"Start recording video to {video_path}")
                if hasattr(self.env, "start_policy_recording"):
                    try:
                        self.env.start_policy_recording(
                            episode_idx,
                            policy_normalizer=getattr(policy, "normalizer", None),
                        )
                    except TypeError:
                        self.env.start_policy_recording(episode_idx)

                self.stop_event.clear()
                self.action_thread_error = None
                time.sleep(0.5)
                self.action_step_count = 0
                # start a new thread for action command
                action_thread = threading.Thread(target=self.action_command_thread, args=(policy, self.stop_event,),
                                                 daemon=True)
                if hasattr(self.env, "notify_policy_started"):
                    self.env.notify_policy_started()
                action_thread.start()

                step_count = 0
                steps_per_inference = int(self.control_fps / self.inference_fps)
                start_timestamp = time.time()
                last_timestamp = start_timestamp
                try:
                    while True:
                        if self.action_thread_error is not None:
                            raise RuntimeError("Action command thread failed") from self.action_thread_error

                        # profiler = Profiler()
                        # profiler.start()
                        start_time = time.time()
                        # get obs
                        obs = self.env.get_obs(
                            obs_steps=self.n_obs_steps,
                            temporal_downsample_ratio=self.obs_temporal_downsample_ratio)
                        # obs = dict()

                        if len(obs) == 0:
                            logger.warning("No observation received! Skip this step.")
                            cur_time = time.time()
                            precise_sleep(max(0., self.inference_interval_time - (cur_time - start_time)))
                            step_count += steps_per_inference
                            continue

                        # create obs dict
                        np_real_obs_dict = dict(obs)
                        # get transformed real obs dict
                        np_real_obs_dict = get_real_obs_dict(
                            env_obs=np_real_obs_dict, shape_meta=self.shape_meta)
                        np_policy_obs_dict = deepcopy(np_real_obs_dict)
                        self._maybe_adjust_policy_tcp_pose_obs(
                            np_policy_obs_dict,
                            np_real_obs_dict,
                        )
                        np_obs_dict, _ = self.pre_process_obs(np_policy_obs_dict)
                        _, np_absolute_obs_dict = self.pre_process_obs(np_real_obs_dict)

                        # device transfer
                        obs_dict = dict_apply(np_obs_dict,
                                              lambda x: torch.from_numpy(x).unsqueeze(0).to(
                                                  device=device))

                        policy_time = time.time()
                        # run policy
                        with torch.no_grad():
                            if self.use_latent_action_with_rnn_decoder:
                                action_dict = policy.predict_action(obs_dict,
                                                                    dataset_obs_temporal_downsample_ratio=self.dataset_obs_temporal_downsample_ratio,
                                                                    return_latent_action=True)
                            else:
                                action_dict = policy.predict_action(obs_dict)
                        logger.debug(f"Policy inference time: {time.time() - policy_time:.3f}s")

                        # device_transfer
                        np_action_dict = dict_apply(action_dict,
                                                    lambda x: x.detach().to('cpu').numpy())

                        action_all = np_action_dict['action'].squeeze(0)
                        if self.use_latent_action_with_rnn_decoder:
                            # add first absolute action to get absolute action
                            if self.use_relative_gripper_action:
                                base_absolute_gripper_action = self._get_gripper_action_base(
                                    np_absolute_obs_dict
                                )
                                action_all = np.concatenate([
                                    action_all,
                                    base_absolute_gripper_action[np.newaxis, :].repeat(
                                        action_all.shape[0], axis=0
                                    ),
                                ], axis=-1)
                            if self.use_relative_action:
                                base_absolute_action = np.concatenate([
                                    np_absolute_obs_dict['left_robot_tcp_pose'][-1] if 'left_robot_tcp_pose' in np_absolute_obs_dict else np.array([]),
                                    np_absolute_obs_dict['right_robot_tcp_pose'][-1] if 'right_robot_tcp_pose' in np_absolute_obs_dict else np.array([])
                                ], axis=-1)
                                action_all = np.concatenate([
                                    action_all,
                                    base_absolute_action[np.newaxis, :].repeat(action_all.shape[0], axis=0)
                                ], axis=-1)
                            # add action step to get corresponding observation
                            action_all = np.concatenate([
                                action_all,
                                np.arange(self.n_obs_steps * self.dataset_obs_temporal_downsample_ratio, action_all.shape[0] + self.n_obs_steps * self.dataset_obs_temporal_downsample_ratio)[:, np.newaxis]
                            ], axis=-1)
                        else:
                            if self.use_relative_action:
                                base_absolute_action = np.concatenate([
                                    np_absolute_obs_dict['left_robot_tcp_pose'][-1] if 'left_robot_tcp_pose' in np_absolute_obs_dict else np.array([]),
                                    np_absolute_obs_dict['right_robot_tcp_pose'][-1] if 'right_robot_tcp_pose' in np_absolute_obs_dict else np.array([])
                                ], axis=-1)
                                action_all = self._maybe_scale_relative_action_on_contact(action_all, obs)
                                action_all = self._maybe_transform_relative_action_frame(
                                    action_all,
                                    base_absolute_action,
                                )
                                action_all = relative_actions_to_absolute_actions(action_all, base_absolute_action)
                            action_all = self._maybe_make_gripper_action_absolute(
                                action_all,
                                np_absolute_obs_dict,
                            )

                        if self.action_interpolation_ratio > 1:
                            if self.use_latent_action_with_rnn_decoder:
                                action_all = action_all.repeat(self.action_interpolation_ratio, axis=0)
                            else:
                                action_all = interpolate_actions_with_ratio(action_all, self.action_interpolation_ratio)

                        if (
                            self.debug_policy_actions
                            and (step_count // steps_per_inference) % self.debug_policy_action_every == 0
                        ):
                            obs_xyz = None
                            obs_gripper = None
                            if 'left_robot_tcp_pose' in np_absolute_obs_dict:
                                obs_xyz = np_absolute_obs_dict['left_robot_tcp_pose'][-1, :3]
                            if 'left_robot_gripper_width' in np_absolute_obs_dict:
                                obs_gripper = float(np_absolute_obs_dict['left_robot_gripper_width'][-1, 0])
                            if action_all.shape[-1] == 4:
                                action_xyz = action_all[:, :3]
                                action_gripper = action_all[:, 3]
                            elif action_all.shape[-1] == 8:
                                action_xyz = action_all[:, :3]
                                action_gripper = action_all[:, 6]
                            elif action_all.shape[-1] == 10:
                                action_xyz = action_all[:, :3]
                                action_gripper = action_all[:, 9]
                            elif action_all.shape[-1] == 20:
                                action_xyz = action_all[:, :3]
                                action_gripper = action_all[:, 18]
                            else:
                                action_xyz = action_all[:, :3]
                                action_gripper = None
                            first_xyz = action_xyz[0]
                            last_xyz = action_xyz[-1]
                            pos_span = float(np.linalg.norm(last_xyz - first_xyz))
                            if obs_xyz is not None:
                                first_delta = float(np.linalg.norm(first_xyz - obs_xyz))
                                last_delta = float(np.linalg.norm(last_xyz - obs_xyz))
                            else:
                                first_delta = float("nan")
                                last_delta = float("nan")
                            if action_gripper is not None:
                                gripper_summary = (
                                    f"gripper_first={float(action_gripper[0]):.4f}, "
                                    f"gripper_last={float(action_gripper[-1]):.4f}, "
                                    f"gripper_minmax=[{float(np.min(action_gripper)):.4f}, "
                                    f"{float(np.max(action_gripper)):.4f}]"
                                )
                            else:
                                gripper_summary = "gripper=NA"
                            obs_xyz_text = "NA" if obs_xyz is None else np.round(obs_xyz, 4).tolist()
                            obs_gripper_text = "NA" if obs_gripper is None else f"{obs_gripper:.4f}"
                            logger.info(
                                "Policy action prediction: "
                                f"step={step_count}, "
                                f"obs_xyz={obs_xyz_text}, "
                                f"obs_gripper={obs_gripper_text}, "
                                f"first_xyz={np.round(first_xyz, 4).tolist()}, "
                                f"last_xyz={np.round(last_xyz, 4).tolist()}, "
                                f"z_first_last=[{float(first_xyz[2]):.4f}, {float(last_xyz[2]):.4f}], "
                                f"delta_from_obs=[{first_delta:.4f}, {last_delta:.4f}], "
                                f"trajectory_span={pos_span:.4f}, "
                                f"{gripper_summary}"
                            )

                        # TODO: only takes the first n_action_steps and add to the ensemble buffer
                        if step_count % self.tcp_action_update_interval == 0:
                            if self.use_latent_action_with_rnn_decoder:
                                tcp_action = action_all[self.latency_step:, ...]
                            else:
                                if action_all.shape[-1] == 4:
                                    tcp_action = action_all[self.latency_step:, :3]
                                elif action_all.shape[-1] == 8:
                                    tcp_action = action_all[self.latency_step:, :6]
                                elif action_all.shape[-1] == 10:
                                    tcp_action = action_all[self.latency_step:, :9]
                                elif action_all.shape[-1] == 20:
                                    tcp_action = action_all[self.latency_step:, :18]
                                else:
                                    raise NotImplementedError
                            # add to ensemble buffer
                            logger.debug(f"Step: {step_count}, Add TCP action to ensemble buffer: {tcp_action}")
                            self.tcp_ensemble_buffer.add_action(tcp_action, step_count)

                            if self.env.enable_exp_recording and not self.use_latent_action_with_rnn_decoder:
                                self.env.get_predicted_action(tcp_action, type='full_tcp')

                        if step_count % self.gripper_action_update_interval == 0:
                            if self.use_latent_action_with_rnn_decoder:
                                gripper_action = action_all[self.gripper_latency_step:, ...]
                            else:
                                if action_all.shape[-1] == 4:
                                    gripper_action = action_all[self.gripper_latency_step:, 3:]
                                elif action_all.shape[-1] == 8:
                                    gripper_action = action_all[self.gripper_latency_step:, 6:]
                                elif action_all.shape[-1] == 10:
                                    gripper_action = action_all[self.gripper_latency_step:, 9:]
                                elif action_all.shape[-1] == 20:
                                    gripper_action = action_all[self.gripper_latency_step:, 18:]
                                else:
                                    raise NotImplementedError
                            # add to ensemble buffer
                            logger.debug(f"Step: {step_count}, Add gripper action to ensemble buffer: {gripper_action}")
                            self.gripper_ensemble_buffer.add_action(gripper_action, step_count)

                            if self.env.enable_exp_recording and not self.use_latent_action_with_rnn_decoder:
                                self.env.get_predicted_action(gripper_action, type='full_gripper')

                        cur_time = time.time()
                        precise_sleep(max(0., self.inference_interval_time - (cur_time - start_time)))
                        if cur_time - start_timestamp >= self.max_duration_time:
                            logger.info(f"Episode {episode_idx} reaches max duration time {self.max_duration_time} seconds.")
                            break
                        step_count += steps_per_inference
                        # profiler.stop()
                        # profiler.print()

                except KeyboardInterrupt:
                    logger.warning("KeyboardInterrupt! Terminate the episode now!")
                finally:
                    self.stop_event.set()
                    action_thread.join()
                    if self.enable_video_recording:
                        self.stop_record_video()
                    if hasattr(self.env, "stop_policy_recording"):
                        self.env.stop_policy_recording()
                    self.env.save_exp(episode_idx)

            # TODO: support success count
            if spin_thread is not None:
                spin_thread.join()
        finally:
            self.env.destroy_node()
