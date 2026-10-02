import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import zarr
from numcodecs import Blosc
from scipy.spatial.transform import Rotation, Slerp


IMAGE_SIZE = 224
POSE_DIM = 6
ACTION_DIM = 7
FORCE_DIM = 6
MAGNETIC_CHIP_COUNT = 5
MAGNETIC_VALUE_DIM = 4
EVAL_MAGNET_USED_CHIP_COUNT = 4
EVAL_MAGNET_SUBTRACT_BASELINE = True
MAGNET_ABNORMAL_ABS_THRESHOLD = 5000.0
MAGNET_WRAP_RECOVERY_ENABLED = True
MAGNET_WRAP_PERIOD = 9830.4
MAGNET_WRAP_SENSOR_INDEX = 2
MAGNET_WRAP_AXIS_INDEX = 2
MAGNET_WRAP_JUMP_THRESHOLD = MAGNET_WRAP_PERIOD * 0.5
RDP_SOURCE_ZARR_SCHEMA_VERSION = 5
STATIC_POS_THRESHOLD = 1e-4
STATIC_ROT_THRESHOLD = 1e-3
VIDEO_PANEL_WIDTH = 420
ZARR_CHUNK_ROWS = 1024
DEFAULT_EEF_CALIBRATION_RESULT = Path(
    "/home/shuwang/CodeFile/umi_data/poser_transform/offline_calibration_result.json"
)


@dataclass
class CaptureInputs:
    directory: Path
    manifest_path: Optional[Path]
    manifest: dict
    pose_csv: Optional[Path]
    magnetic_csv: Optional[Path]
    magnetic_left_csv: Optional[Path]
    force_csv: Optional[Path]
    gripper_csv: Optional[Path]
    video_path: Optional[Path]


@dataclass
class PoseSeries:
    time: np.ndarray
    position: np.ndarray
    quaternion: np.ndarray


@dataclass
class ForceSeries:
    time: np.ndarray
    values: np.ndarray


@dataclass
class GripperSeries:
    time: np.ndarray
    width_m: np.ndarray


@dataclass
class MagneticSeries:
    time: np.ndarray
    values: np.ndarray


@dataclass
class EpisodeArrays:
    camera_rgb: np.ndarray
    timestamp: np.ndarray
    eef_pos: np.ndarray
    eef_rot_axis_angle: np.ndarray
    gripper_width: np.ndarray
    demo_start_pose: np.ndarray
    demo_end_pose: np.ndarray
    action: np.ndarray
    force_torque: np.ndarray
    force_valid: np.ndarray
    magnet_xyz: np.ndarray
    magnet_timestamp_ns: np.ndarray
    magnet_sample_count: np.ndarray
    magnetic_txyz: np.ndarray
    magnetic_valid: np.ndarray
    magnetic_left_txyz: np.ndarray
    magnetic_left_valid: np.ndarray


@dataclass(frozen=True)
class EEFCalibration:
    path: Path
    scale_factor: float
    T_cam2gripper: np.ndarray
    T_base_world: np.ndarray


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert ARPoseStreamer capture folders into a dataset.zarr directory."
    )
    parser.add_argument(
        "--capture",
        action="append",
        required=True,
        type=Path,
        help="Capture folder. Can be passed multiple times for multiple episodes.",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output .zarr directory.")
    parser.add_argument(
        "--image-size",
        type=int,
        default=IMAGE_SIZE,
        help="Square RGB image size written to camera0_rgb. Default: 224.",
    )
    parser.add_argument(
        "--action-source",
        choices=("next_obs", "zero", "force"),
        default="next_obs",
        help=(
            "How to write data/action. next_obs writes the next sampled "
            "xyz+rotvec+gripper pose, matching the UMI/RDP source-zarr "
            "contract. zero writes all zeros. force copies force/torque into "
            "action[:,0:6]. Default: next_obs."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory if it already exists.",
    )
    parser.add_argument(
        "--trim-static",
        action="store_true",
        help="Trim static frames at the beginning and near-static pauses in the middle.",
    )
    parser.add_argument(
        "--static-pos-threshold",
        type=float,
        default=STATIC_POS_THRESHOLD,
        help="Position step threshold in meters for static-frame trimming.",
    )
    parser.add_argument(
        "--static-rot-threshold",
        type=float,
        default=STATIC_ROT_THRESHOLD,
        help="Rotation step threshold in radians for static-frame trimming.",
    )
    parser.add_argument(
        "--video-output",
        type=Path,
        default=None,
        help="Optional directory for synchronized camera/magnet visualization videos.",
    )
    parser.add_argument(
        "--video-panel-width",
        type=int,
        default=VIDEO_PANEL_WIDTH,
        help="Width of the magnetic plot panel in visualization videos.",
    )
    parser.add_argument(
        "--eef-calibration-result",
        type=Path,
        default=default_eef_calibration_result(),
        help=(
            "Offline calibration JSON used to convert iPhone/ARKit camera pose "
            "into gripper end-effector pose. Defaults to "
            f"{DEFAULT_EEF_CALIBRATION_RESULT} when it exists."
        ),
    )
    parser.add_argument(
        "--raw-iphone-pose",
        action="store_true",
        help="Write raw iPhone/ARKit pose as robot0_eef_* instead of calibrated gripper pose.",
    )
    parser.add_argument(
        "--no-magnet-wrap-recovery",
        action="store_true",
        help="Disable recovery of the known wrapped raw magnetic channel before interpolation.",
    )
    parser.add_argument(
        "--magnet-wrap-period",
        type=float,
        default=MAGNET_WRAP_PERIOD,
        help=f"Wrap period for the selected magnetic channel. Default: {MAGNET_WRAP_PERIOD:g}.",
    )
    parser.add_argument(
        "--magnet-wrap-sensor",
        type=int,
        default=MAGNET_WRAP_SENSOR_INDEX,
        help=f"Zero-based chip index to unwrap. Default: {MAGNET_WRAP_SENSOR_INDEX}.",
    )
    parser.add_argument(
        "--magnet-wrap-axis",
        type=int,
        default=MAGNET_WRAP_AXIS_INDEX,
        help=f"Zero-based value index (t=0,x=1,y=2,z=3) to unwrap. Default: {MAGNET_WRAP_AXIS_INDEX}.",
    )
    parser.add_argument(
        "--magnet-wrap-jump-threshold",
        type=float,
        default=None,
        help="Minimum raw step treated as a wrap. Defaults to half the wrap period.",
    )
    args = parser.parse_args()
    eef_calibration_result = None if args.raw_iphone_pose else args.eef_calibration_result
    magnet_wrap_recovery = not args.no_magnet_wrap_recovery
    magnet_wrap_jump_threshold = (
        args.magnet_wrap_jump_threshold
        if args.magnet_wrap_jump_threshold is not None
        else args.magnet_wrap_period * 0.5
    )

    if args.out.exists():
        if not args.overwrite:
            raise SystemExit(f"Output already exists: {args.out}. Use --overwrite to replace it.")
        if args.out.is_dir():
            shutil.rmtree(args.out)
        else:
            args.out.unlink()

    captures = [discover_capture(path) for path in args.capture]
    episodes = [
        build_episode(
            capture,
            image_size=args.image_size,
            action_source=args.action_source,
            trim_static=args.trim_static,
            static_pos_threshold=args.static_pos_threshold,
            static_rot_threshold=args.static_rot_threshold,
            eef_calibration_result=eef_calibration_result,
            magnet_wrap_recovery=magnet_wrap_recovery,
            magnet_wrap_period=args.magnet_wrap_period,
            magnet_wrap_sensor_index=args.magnet_wrap_sensor,
            magnet_wrap_axis_index=args.magnet_wrap_axis,
            magnet_wrap_jump_threshold=magnet_wrap_jump_threshold,
        )
        for capture in captures
    ]
    write_zarr(
        args.out,
        episodes,
        attrs=make_zarr_attrs(
            eef_calibration_result,
            action_source=args.action_source,
            magnet_wrap_recovery=magnet_wrap_recovery,
            magnet_wrap_period=args.magnet_wrap_period,
            magnet_wrap_sensor_index=args.magnet_wrap_sensor,
            magnet_wrap_axis_index=args.magnet_wrap_axis,
            magnet_wrap_jump_threshold=magnet_wrap_jump_threshold,
        ),
    )
    if args.video_output is not None:
        write_episode_visualizations(
            args.video_output,
            captures,
            episodes,
            overwrite=args.overwrite,
            panel_width=args.video_panel_width,
        )

    total_frames = sum(len(ep.timestamp) for ep in episodes)
    print(f"Wrote {args.out} with {len(episodes)} episode(s), {total_frames} frame(s).")
    return 0


def discover_capture(directory: Path) -> CaptureInputs:
    directory = directory.resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Capture directory not found: {directory}")

    manifest_path = first_existing(
        [
            directory / "capture_manifest.json",
            directory / "manifest__capture_manifest.json",
            *sorted(directory.glob("*manifest*.json")),
        ]
    )
    manifest = load_json(manifest_path) if manifest_path else {}

    pose_name = manifest.get("poseCSVFileName")
    pose_csv = find_capture_file(
        directory,
        logical_name=pose_name,
        component_prefix="pose_csv",
        fallback_patterns=("pose.csv", "pose_csv__*.csv", "*pose*.csv"),
        require_nonempty=True,
    )

    magnetic_name = (
        manifest.get("rightMagneticCSVFileName")
        or manifest.get("magneticCSVFileName")
    )
    magnetic_csv = find_capture_file(
        directory,
        logical_name=magnetic_name,
        component_prefix="magnetic_right_csv",
        fallback_patterns=(
            "magnetic_right.csv",
            "magnetic_right_csv__*.csv",
            "magnetic.csv",
            "magnetic_csv__*.csv",
        ),
        require_nonempty=True,
        required=False,
    )
    magnetic_left_name = manifest.get("leftMagneticCSVFileName")
    magnetic_left_csv = find_capture_file(
        directory,
        logical_name=magnetic_left_name,
        component_prefix="magnetic_left_csv",
        fallback_patterns=("magnetic_left.csv", "magnetic_left_csv__*.csv"),
        require_nonempty=True,
        required=False,
    )

    force_csv = find_capture_file(
        directory,
        logical_name=None,
        component_prefix=None,
        fallback_patterns=("fused_force_pose.csv", "force.csv", "*force*.csv"),
        require_nonempty=True,
        required=False,
    )

    gripper_name = (
        manifest.get("arucoGripperCSVFileName")
        or manifest.get("gripperCSVFileName")
        or manifest.get("gripperFileName")
    )
    gripper_csv = find_capture_file(
        directory,
        logical_name=gripper_name,
        component_prefix="aruco_gripper",
        fallback_patterns=(
            "aruco_gripper.csv",
            "aruco_gripper__*.csv",
            "*gripper*.csv",
        ),
        require_nonempty=True,
        required=False,
    )

    video_name = manifest.get("videoFileName")
    video_path = find_capture_file(
        directory,
        logical_name=video_name,
        component_prefix="video",
        fallback_patterns=(
            "video.mp4",
            "video.mov",
            "video.m4v",
            "video__*.mp4",
            "video__*.mov",
            "video__*.m4v",
            "ARPoseStreamer*.mp4",
            "ARPoseStreamer*.mov",
            "ARPoseStreamer*.m4v",
        ),
        require_nonempty=True,
        required=False,
    )
    if video_path is None:
        video_path = find_non_ultrawide_video_file(directory)

    return CaptureInputs(
        directory=directory,
        manifest_path=manifest_path,
        manifest=manifest,
        pose_csv=pose_csv,
        magnetic_csv=magnetic_csv,
        magnetic_left_csv=magnetic_left_csv,
        force_csv=force_csv,
        gripper_csv=gripper_csv,
        video_path=video_path,
    )


def build_episode(
    capture: CaptureInputs,
    image_size: int,
    action_source: str,
    trim_static: bool = False,
    static_pos_threshold: float = STATIC_POS_THRESHOLD,
    static_rot_threshold: float = STATIC_ROT_THRESHOLD,
    eef_calibration_result: Optional[Path] = None,
    magnet_wrap_recovery: bool = MAGNET_WRAP_RECOVERY_ENABLED,
    magnet_wrap_period: float = MAGNET_WRAP_PERIOD,
    magnet_wrap_sensor_index: int = MAGNET_WRAP_SENSOR_INDEX,
    magnet_wrap_axis_index: int = MAGNET_WRAP_AXIS_INDEX,
    magnet_wrap_jump_threshold: Optional[float] = None,
) -> EpisodeArrays:
    pose = load_pose_series(capture.pose_csv, capture.manifest) if capture.pose_csv else None
    magnetic = (
        load_magnetic_series(
            capture.magnetic_csv,
            capture.manifest,
            unwrap=magnet_wrap_recovery,
            wrap_period=magnet_wrap_period,
            wrap_sensor_index=magnet_wrap_sensor_index,
            wrap_axis_index=magnet_wrap_axis_index,
            wrap_jump_threshold=magnet_wrap_jump_threshold,
        )
        if capture.magnetic_csv
        else None
    )
    magnetic_left = (
        load_magnetic_series(
            capture.magnetic_left_csv,
            capture.manifest,
            unwrap=magnet_wrap_recovery,
            wrap_period=magnet_wrap_period,
            wrap_sensor_index=magnet_wrap_sensor_index,
            wrap_axis_index=magnet_wrap_axis_index,
            wrap_jump_threshold=magnet_wrap_jump_threshold,
        )
        if capture.magnetic_left_csv
        else None
    )
    force = load_force_series(capture.force_csv, capture.manifest) if capture.force_csv else None
    gripper = load_gripper_series(capture.gripper_csv, capture.manifest) if capture.gripper_csv else None
    eef_calibration = load_eef_calibration_result(eef_calibration_result)

    camera_rgb, sample_time = load_video_frames(
        capture.video_path,
        capture.manifest,
        fallback_pose=pose,
        image_size=image_size,
    )

    if len(sample_time) == 0:
        raise ValueError(f"No video frames or pose samples available in {capture.directory}")

    phone_pos, phone_quaternion = sample_pose_quaternion_on_time(pose, sample_time)
    if eef_calibration is None:
        eef_pos, eef_rot_axis_angle = pose_quaternion_to_pos_rotvec(phone_pos, phone_quaternion)
    else:
        eef_pos, eef_rot_axis_angle = transform_phone_pose_to_eef(
            position=phone_pos,
            quaternion=phone_quaternion,
            calibration=eef_calibration,
        )
    force_torque, force_valid = sample_force_on_time(force, sample_time)
    magnetic_txyz, magnetic_valid = sample_magnetic_on_time(magnetic, sample_time)
    magnetic_left_txyz, magnetic_left_valid = sample_magnetic_on_time(
        magnetic_left,
        sample_time,
    )
    gripper_width = sample_gripper_on_time(gripper, sample_time)
    if gripper is None:
        print(f"[WARN] {capture.directory.name}: no aruco_gripper.csv found; robot0_gripper_width stays zero")
    else:
        print(
            f"[INFO] {capture.directory.name}: sampled gripper width from {capture.gripper_csv.name}, "
            f"range=[{float(np.min(gripper_width)):.4f}, {float(np.max(gripper_width)):.4f}]m"
        )

    magnet_xyz, magnet_timestamp_ns, magnet_sample_count = build_magnet_arrays(
        magnetic_txyz=magnetic_txyz,
        magnetic_valid=magnetic_valid,
        sample_time=sample_time,
        magnetic_left_txyz=magnetic_left_txyz,
        magnetic_left_valid=magnetic_left_valid,
    )
    cleaned_magnet_xyz, replaced_magnet_count = replace_abnormal_magnet_readings(magnet_xyz)
    if replaced_magnet_count:
        print(
            f"[WARN] {capture.directory.name}: replaced {replaced_magnet_count} "
            f"abnormal magnet readings with nearest normal readings "
            f"(abs threshold {MAGNET_ABNORMAL_ABS_THRESHOLD:g})"
    )
    magnet_xyz = cleaned_magnet_xyz

    data = {
        "camera0_rgb": camera_rgb,
        "timestamp": sample_time,
        "robot0_eef_pos": eef_pos,
        "robot0_eef_rot_axis_angle": eef_rot_axis_angle,
        "robot0_gripper_width": gripper_width,
        "force_torque": force_torque,
        "force_valid": force_valid,
        "magnet_xyz": magnet_xyz,
        "magnet_timestamp_ns": magnet_timestamp_ns,
        "magnet_sample_count": magnet_sample_count,
        "magnetic_txyz": magnetic_txyz,
        "magnetic_valid": magnetic_valid,
        "magnetic_left_txyz": magnetic_left_txyz,
        "magnetic_left_valid": magnetic_left_valid,
    }
    if trim_static:
        data = trim_static_frames(
            capture.directory,
            data,
            pos_threshold=static_pos_threshold,
            rot_threshold=static_rot_threshold,
        )

    camera_rgb = data["camera0_rgb"]
    sample_time = data["timestamp"]
    eef_pos = data["robot0_eef_pos"]
    eef_rot_axis_angle = data["robot0_eef_rot_axis_angle"]
    gripper_width = data["robot0_gripper_width"]
    force_torque = data["force_torque"]
    force_valid = data["force_valid"]
    magnet_xyz = data["magnet_xyz"]
    magnet_timestamp_ns = data["magnet_timestamp_ns"]
    magnet_sample_count = data["magnet_sample_count"]
    magnetic_txyz = data["magnetic_txyz"]
    magnetic_valid = data["magnetic_valid"]
    magnetic_left_txyz = data["magnetic_left_txyz"]
    magnetic_left_valid = data["magnetic_left_valid"]

    pose6 = np.concatenate([eef_pos, eef_rot_axis_angle], axis=1).astype(np.float32)
    start_pose = pose6[0] if len(pose6) else np.zeros(POSE_DIM, dtype=np.float32)
    end_pose = pose6[-1] if len(pose6) else np.zeros(POSE_DIM, dtype=np.float32)
    demo_start_pose = np.repeat(start_pose[None, :], len(sample_time), axis=0).astype(np.float32)
    demo_end_pose = np.repeat(end_pose[None, :], len(sample_time), axis=0).astype(np.float32)

    action = build_action(
        eef_pos=eef_pos,
        eef_rot_axis_angle=eef_rot_axis_angle,
        gripper_width=gripper_width,
        force_torque=force_torque,
        action_source=action_source,
    )

    return EpisodeArrays(
        camera_rgb=camera_rgb.astype(np.uint8, copy=False),
        timestamp=sample_time.astype(np.float64, copy=False),
        eef_pos=eef_pos.astype(np.float32, copy=False),
        eef_rot_axis_angle=eef_rot_axis_angle.astype(np.float32, copy=False),
        gripper_width=gripper_width,
        demo_start_pose=demo_start_pose,
        demo_end_pose=demo_end_pose,
        action=action,
        force_torque=force_torque.astype(np.float32, copy=False),
        force_valid=force_valid.astype(bool, copy=False),
        magnet_xyz=magnet_xyz.astype(np.float32, copy=False),
        magnet_timestamp_ns=magnet_timestamp_ns.astype(np.int64, copy=False),
        magnet_sample_count=magnet_sample_count.astype(np.int32, copy=False),
        magnetic_txyz=magnetic_txyz.astype(np.float32, copy=False),
        magnetic_valid=magnetic_valid.astype(bool, copy=False),
        magnetic_left_txyz=magnetic_left_txyz.astype(np.float32, copy=False),
        magnetic_left_valid=magnetic_left_valid.astype(bool, copy=False),
    )


def build_action(
    eef_pos: np.ndarray,
    eef_rot_axis_angle: np.ndarray,
    gripper_width: np.ndarray,
    force_torque: np.ndarray,
    action_source: str,
) -> np.ndarray:
    action = np.zeros((len(eef_pos), ACTION_DIM), dtype=np.float32)
    if action_source == "force":
        action[:, :FORCE_DIM] = force_torque
        return action
    if action_source == "zero":
        return action
    if action_source != "next_obs":
        raise ValueError(f"Unsupported action_source: {action_source}")

    if len(eef_pos) == 0:
        return action
    next_indices = np.arange(len(eef_pos), dtype=np.int64)
    if len(next_indices) > 1:
        next_indices[:-1] = next_indices[1:]
    action[:, :3] = eef_pos[next_indices, :3]
    action[:, 3:6] = eef_rot_axis_angle[next_indices, :3]
    action[:, 6:7] = gripper_width[next_indices, :1]
    return action


def default_eef_calibration_result() -> Optional[Path]:
    return DEFAULT_EEF_CALIBRATION_RESULT if DEFAULT_EEF_CALIBRATION_RESULT.is_file() else None


def make_zarr_attrs(
    eef_calibration_result: Optional[Path],
    action_source: str,
    *,
    magnet_wrap_recovery: bool = MAGNET_WRAP_RECOVERY_ENABLED,
    magnet_wrap_period: float = MAGNET_WRAP_PERIOD,
    magnet_wrap_sensor_index: int = MAGNET_WRAP_SENSOR_INDEX,
    magnet_wrap_axis_index: int = MAGNET_WRAP_AXIS_INDEX,
    magnet_wrap_jump_threshold: Optional[float] = None,
) -> Dict[str, object]:
    path = Path(eef_calibration_result).expanduser().resolve() if eef_calibration_result else None
    jump_threshold = (
        magnet_wrap_period * 0.5
        if magnet_wrap_jump_threshold is None
        else float(magnet_wrap_jump_threshold)
    )
    return {
        "action_source": action_source,
        "rdp_source_zarr_schema_version": RDP_SOURCE_ZARR_SCHEMA_VERSION,
        "magnet_key": "magnet_xyz",
        "magnet_processing": "eval_compatible",
        "magnet_source_key": (
            "stack(right=magnetic_txyz,left=magnetic_left_txyz)[:, :, :, 1:4]"
        ),
        "magnet_board_order": ["right", "left"],
        "magnet_used_chip_count": EVAL_MAGNET_USED_CHIP_COUNT,
        "magnet_subtract_baseline": EVAL_MAGNET_SUBTRACT_BASELINE,
        "magnet_abnormal_abs_threshold": MAGNET_ABNORMAL_ABS_THRESHOLD,
        "magnet_wrap_recovery": bool(magnet_wrap_recovery),
        "magnet_wrap_period": float(magnet_wrap_period),
        "magnet_wrap_sensor_index": int(magnet_wrap_sensor_index),
        "magnet_wrap_axis_index": int(magnet_wrap_axis_index),
        "magnet_wrap_jump_threshold": float(jump_threshold),
        "magnet_wrap_recovery_stage": "raw_csv_before_video_interpolation",
        "gripper_width_source": "aruco_gripper_calibrated_width_sampled_or_zero_fallback",
        "gripper_width_unit": "m",
        "eef_pose_source": "iphone_arkit_calibrated_to_gripper" if path else "raw_iphone_arkit",
        "eef_calibration_result": str(path) if path else "",
    }


def load_eef_calibration_result(path: Optional[Path]) -> Optional[EEFCalibration]:
    if path is None or str(path).strip() == "":
        return None
    calibration_path = Path(path).expanduser().resolve()
    if not calibration_path.is_file():
        raise FileNotFoundError(f"EEF calibration result not found: {calibration_path}")

    payload = load_json(calibration_path)
    T_base_world = read_transform_matrix(payload.get("T_base_world"), "T_base_world")
    if payload.get("T_cam2gripper") is not None:
        T_cam2gripper = read_transform_matrix(payload.get("T_cam2gripper"), "T_cam2gripper")
    else:
        rotation = np.asarray(payload.get("R_cam2gripper"), dtype=np.float64)
        translation = np.asarray(payload.get("t_cam2gripper"), dtype=np.float64).reshape(3)
        if rotation.shape != (3, 3):
            raise ValueError(f"R_cam2gripper must be 3x3 in {calibration_path}")
        T_cam2gripper = np.eye(4, dtype=np.float64)
        T_cam2gripper[:3, :3] = rotation
        T_cam2gripper[:3, 3] = translation

    scale_factor = optional_float(payload.get("scale_factor"))
    if scale_factor is None:
        scale_factor = 1.0
    return EEFCalibration(
        path=calibration_path,
        scale_factor=float(scale_factor),
        T_cam2gripper=T_cam2gripper,
        T_base_world=T_base_world,
    )


def read_transform_matrix(value: object, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 matrix, got shape {matrix.shape}")
    return matrix


def pose_quaternion_to_pos_rotvec(
    position: np.ndarray,
    quaternion: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(position) == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
        )
    rotvec = Rotation.from_quat(quaternion).as_rotvec()
    return position.astype(np.float32, copy=False), rotvec.astype(np.float32, copy=False)


def transform_phone_pose_to_eef(
    position: np.ndarray,
    quaternion: np.ndarray,
    calibration: EEFCalibration,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(position) == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
        )

    rotations = Rotation.from_quat(quaternion).as_matrix()
    T_cam2gripper_inv = np.linalg.inv(calibration.T_cam2gripper)
    eef_pos = np.zeros((len(position), 3), dtype=np.float64)
    eef_rot = np.zeros((len(position), 3, 3), dtype=np.float64)
    for index, (pos, rotation) in enumerate(zip(position, rotations)):
        T_world_cam = np.eye(4, dtype=np.float64)
        T_world_cam[:3, :3] = rotation
        T_world_cam[:3, 3] = np.asarray(pos, dtype=np.float64) * calibration.scale_factor
        T_base_gripper = calibration.T_base_world @ T_world_cam @ T_cam2gripper_inv
        eef_pos[index] = T_base_gripper[:3, 3]
        eef_rot[index] = T_base_gripper[:3, :3]
    eef_rot_axis_angle = Rotation.from_matrix(eef_rot).as_rotvec()
    return eef_pos.astype(np.float32), eef_rot_axis_angle.astype(np.float32)


def build_magnet_arrays(
    magnetic_txyz: np.ndarray,
    magnetic_valid: np.ndarray,
    sample_time: np.ndarray,
    magnetic_left_txyz: Optional[np.ndarray] = None,
    magnetic_left_valid: Optional[np.ndarray] = None,
    used_chip_count: int = EVAL_MAGNET_USED_CHIP_COUNT,
    subtract_baseline: bool = EVAL_MAGNET_SUBTRACT_BASELINE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    used_chip_count = int(used_chip_count)
    if used_chip_count < 1 or used_chip_count > MAGNETIC_CHIP_COUNT:
        raise ValueError(
            f"used_chip_count must be in [1, {MAGNETIC_CHIP_COUNT}], got {used_chip_count}"
        )
    if magnetic_left_txyz is None:
        magnetic_left_txyz = np.zeros_like(magnetic_txyz)
    if magnetic_left_valid is None:
        magnetic_left_valid = np.zeros_like(magnetic_valid, dtype=bool)
    if magnetic_left_txyz.shape != magnetic_txyz.shape:
        raise ValueError(
            "Right and left magnetic arrays must have the same shape: "
            f"{magnetic_txyz.shape} != {magnetic_left_txyz.shape}"
        )
    if magnetic_left_valid.shape != magnetic_valid.shape:
        raise ValueError(
            "Right and left magnetic validity arrays must have the same shape: "
            f"{magnetic_valid.shape} != {magnetic_left_valid.shape}"
        )

    magnet_xyz = np.stack(
        [
            magnetic_txyz[:, :used_chip_count, 1:4],
            magnetic_left_txyz[:, :used_chip_count, 1:4],
        ],
        axis=1,
    ).astype(np.float32, copy=True)
    valid = np.column_stack(
        [
            np.any(magnetic_valid[:, :used_chip_count], axis=1),
            np.any(magnetic_left_valid[:, :used_chip_count], axis=1),
        ]
    )
    magnet_sample_count = valid.astype(np.int32)

    if subtract_baseline:
        magnet_xyz = subtract_first_valid_magnet_baseline(
            magnet_xyz=magnet_xyz,
            valid_rows=valid,
        )
    magnet_xyz[~valid] = 0

    timestamp_sec = np.nan_to_num(sample_time.astype(np.float64, copy=False), nan=0.0)
    magnet_timestamp_ns = np.repeat(
        np.rint(timestamp_sec[:, None] * 1e9).astype(np.int64),
        2,
        axis=1,
    )
    return magnet_xyz, magnet_timestamp_ns, magnet_sample_count


def subtract_first_valid_magnet_baseline(
    magnet_xyz: np.ndarray,
    valid_rows: np.ndarray,
) -> np.ndarray:
    magnet_xyz = np.asarray(magnet_xyz, dtype=np.float32)
    if magnet_xyz.ndim != 4 or magnet_xyz.shape[-1] != 3:
        raise ValueError(f"Expected magnet_xyz [T, S, N, 3], got {magnet_xyz.shape}")

    valid_rows = np.asarray(valid_rows, dtype=bool)
    if valid_rows.ndim == 1:
        valid_rows = np.repeat(valid_rows[:, None], magnet_xyz.shape[1], axis=1)
    if valid_rows.shape != magnet_xyz.shape[:2]:
        raise ValueError(
            f"valid_rows shape {valid_rows.shape} does not match magnet rows {magnet_xyz.shape[:2]}"
        )

    baseline = np.zeros(magnet_xyz.shape[1:], dtype=np.float32)
    for sample_idx in range(magnet_xyz.shape[1]):
        valid_indices = np.flatnonzero(valid_rows[:, sample_idx])
        for sensor_idx in range(magnet_xyz.shape[2]):
            for axis_idx in range(3):
                values = magnet_xyz[
                    valid_indices,
                    sample_idx,
                    sensor_idx,
                    axis_idx,
                ]
                finite_indices = np.flatnonzero(np.isfinite(values))
                if finite_indices.size > 0:
                    baseline[sample_idx, sensor_idx, axis_idx] = values[
                        finite_indices[0]
                    ]

    return (magnet_xyz - baseline[None, :, :, :]).astype(np.float32)


def replace_abnormal_magnet_readings(magnet_xyz: np.ndarray) -> Tuple[np.ndarray, int]:
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
    output = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray) and value.shape[:1] == (n,):
            output[key] = value[keep_mask]
        else:
            output[key] = value
    return output


def trim_static_frames(
    capture_dir: Path,
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
        raise ValueError(f"Capture {capture_dir} has no detected movement after static trimming.")
    if trim_start > 0:
        print(f"[WARN] Trimmed {trim_start} static start frames from {capture_dir.name}")
        keep_mask = np.zeros(len(eef_pos_arr), dtype=bool)
        keep_mask[trim_start:] = True
        data = apply_frame_mask(data, keep_mask)
        eef_pos_arr = data["robot0_eef_pos"]
        eef_rot_arr = data["robot0_eef_rot_axis_angle"]

    return data


def infer_video_fps(timestamps: np.ndarray, fallback_fps: float = 25.0) -> float:
    if len(timestamps) < 2:
        return fallback_fps
    duration = float(timestamps[-1] - timestamps[0])
    if duration <= 0:
        return fallback_fps
    fps = float((len(timestamps) - 1) / duration)
    if not np.isfinite(fps) or fps <= 0:
        return fallback_fps
    return min(max(fps, 1.0), 60.0)


def get_magnet_baseline(magnet_xyz: np.ndarray) -> np.ndarray:
    magnet_xyz = np.asarray(magnet_xyz, dtype=np.float32)
    if magnet_xyz.ndim == 4:
        sensor_count = int(magnet_xyz.shape[2])
    elif magnet_xyz.ndim == 3 and magnet_xyz.shape[-1] == 3:
        sensor_count = int(magnet_xyz.shape[1])
    else:
        sensor_count = MAGNETIC_CHIP_COUNT
    baseline = np.full((sensor_count, 3), np.nan, dtype=np.float32)
    if len(magnet_xyz) == 0:
        return baseline
    if magnet_xyz.ndim == 4:
        first_frame = magnet_xyz[0]
        for sensor_idx in range(sensor_count):
            for axis_idx in range(3):
                first_values = first_frame[:, sensor_idx, axis_idx]
                valid_indices = np.flatnonzero(np.isfinite(first_values))
                if valid_indices.size > 0:
                    baseline[sensor_idx, axis_idx] = first_values[valid_indices[-1]]
                    continue
                episode_values = magnet_xyz[:, :, sensor_idx, axis_idx].reshape(-1)
                valid_indices = np.flatnonzero(np.isfinite(episode_values))
                if valid_indices.size > 0:
                    baseline[sensor_idx, axis_idx] = episode_values[valid_indices[0]]
    else:
        first_frame = magnet_xyz[0]
        for sensor_idx in range(sensor_count):
            for axis_idx in range(3):
                first_value = first_frame[sensor_idx, axis_idx]
                if np.isfinite(first_value):
                    baseline[sensor_idx, axis_idx] = first_value
                    continue
                episode_values = magnet_xyz[:, sensor_idx, axis_idx]
                valid_indices = np.flatnonzero(np.isfinite(episode_values))
                if valid_indices.size > 0:
                    baseline[sensor_idx, axis_idx] = episode_values[valid_indices[0]]
    return baseline


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
    board_label: str = "selected",
) -> np.ndarray:
    magnet_frame = np.asarray(magnet_frame, dtype=np.float32)
    sensor_count = int(magnet_frame.shape[1])
    panel = np.full((panel_height, panel_width, 3), 245, dtype=np.uint8)
    row_height = max(panel_height // max(sensor_count, 1), 1)
    graph_left = 58
    colors = {
        "X": (220, 60, 60),
        "Y": (60, 170, 60),
        "Z": (60, 100, 220),
    }
    cv2.putText(
        panel,
        f"relative magnet xyz ({board_label})",
        (10, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    for sensor_idx in range(sensor_count):
        y0 = sensor_idx * row_height
        y1 = panel_height if sensor_idx == sensor_count - 1 else (sensor_idx + 1) * row_height
        top = y0 + 24
        bottom = y1 - 8
        if bottom <= top:
            continue
        center_y = (top + bottom) // 2
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
            f"d{axis}={value:.1f}" if np.isfinite(value) else f"d{axis}=nan"
            for axis, value in zip(("X", "Y", "Z"), latest_values)
        )
        cv2.putText(
            panel,
            latest_text,
            (graph_left, y0 + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )
    return panel


def save_rgb_magnet_video(
    frames: np.ndarray,
    magnet_xyz: np.ndarray,
    timestamps: np.ndarray,
    video_path: Path,
    panel_width: int = VIDEO_PANEL_WIDTH,
) -> None:
    if len(frames) == 0:
        return
    video_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    panel_width = max(int(panel_width), 280)
    if panel_width % 2 != 0:
        panel_width += 1
    if width % 2 != 0:
        frames = frames[:, :, :-1]
        width -= 1
    if height % 2 != 0:
        frames = frames[:, :-1]
        height -= 1

    magnet_xyz = np.asarray(magnet_xyz, dtype=np.float32)
    if magnet_xyz.ndim != 4 or magnet_xyz.shape[1] < 1 or magnet_xyz.shape[-1] != 3:
        raise ValueError(f"Expected magnet_xyz [T, board, sensor, xyz], got {magnet_xyz.shape}")

    # Captures may contain one or two board slots. Keep every board that has
    # signal so dual-sensor captures are visible in the exported video.
    finite_nonzero = np.isfinite(magnet_xyz) & (np.abs(magnet_xyz) > 1e-8)
    board_scores = np.count_nonzero(finite_nonzero, axis=(0, 2, 3))
    active_board_indices = [
        int(index) for index, score in enumerate(board_scores) if int(score) > 0
    ]
    if not active_board_indices:
        active_board_indices = [int(np.argmax(board_scores))]
    board_labels = {0: "right", 1: "left"}
    relative_magnets = {}
    magnet_plot_limits = {}
    for board_index in active_board_indices:
        selected_magnet = magnet_xyz[:, board_index, :, :]
        baseline = get_magnet_baseline(selected_magnet)
        relative_magnet = selected_magnet - baseline[None, :, :]
        relative_magnets[board_index] = relative_magnet
        magnet_plot_limits[board_index] = get_magnet_plot_limit(relative_magnet)
    fps = infer_video_fps(timestamps)
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width + panel_width * len(active_board_indices), height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {video_path}")

    t0 = float(timestamps[0]) if len(timestamps) else 0.0
    try:
        for frame_idx, rgb in enumerate(frames):
            magnet_panels = []
            for board_index in active_board_indices:
                magnet_panel = draw_magnet_panel(
                    magnet_frame=relative_magnets[board_index][: frame_idx + 1],
                    panel_height=height,
                    panel_width=panel_width,
                    value_limit=magnet_plot_limits[board_index],
                    board_label=board_labels.get(board_index, f"board {board_index + 1}"),
                )
                cv2.putText(
                    magnet_panel,
                    f"frame={frame_idx}  t={float(timestamps[frame_idx]) - t0:.3f}s",
                    (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (30, 30, 30),
                    1,
                    cv2.LINE_AA,
                )
                magnet_panels.append(magnet_panel)
            output_rgb = np.concatenate([rgb[:height, :width], *magnet_panels], axis=1)
            writer.write(cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def write_episode_visualizations(
    output_dir: Path,
    captures: List[CaptureInputs],
    episodes: List[EpisodeArrays],
    overwrite: bool = False,
    panel_width: int = VIDEO_PANEL_WIDTH,
) -> None:
    output_dir = output_dir.expanduser()
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for capture, episode in zip(captures, episodes):
        video_path = output_dir / f"{capture.directory.name}_camera0_magnet.mp4"
        save_rgb_magnet_video(
            frames=episode.camera_rgb,
            magnet_xyz=episode.magnet_xyz,
            timestamps=episode.timestamp,
            video_path=video_path,
            panel_width=panel_width,
        )
        print(f"[INFO] Wrote synchronized visualization video: {video_path}")


def load_video_frames(
    video_path: Optional[Path],
    manifest: dict,
    fallback_pose: Optional[PoseSeries],
    image_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if video_path is None:
        if fallback_pose is None or len(fallback_pose.time) == 0:
            return np.zeros((0, image_size, image_size, 3), dtype=np.uint8), np.zeros(0, dtype=np.float64)
        frames = np.zeros((len(fallback_pose.time), image_size, image_size, 3), dtype=np.uint8)
        return frames, fallback_pose.time.copy()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 1e-6:
        fps = 60.0

    video_start_offset = float(manifest.get("videoStartOffsetSeconds") or 0.0)
    created_at = manifest.get("createdAtUnixTime")
    created_at = float(created_at) if created_at is not None else None

    frames: List[np.ndarray] = []
    times: List[float] = []
    frame_index = 0

    while True:
        ok, bgr = cap.read()
        if not ok:
            break

        pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        if np.isfinite(pos_msec) and pos_msec >= 0:
            frame_time = pos_msec / 1000.0
        else:
            frame_time = frame_index / fps

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = resize_center_crop(rgb, image_size)
        frames.append(rgb)

        relative_time = video_start_offset + frame_time
        times.append((created_at + relative_time) if created_at is not None else relative_time)
        frame_index += 1

    cap.release()

    if not frames:
        raise ValueError(f"Video had no readable frames: {video_path}")

    return np.stack(frames, axis=0), np.asarray(times, dtype=np.float64)


def resize_center_crop(image: np.ndarray, output_size: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = output_size / min(height, width)
    resized_width = max(output_size, int(round(width * scale)))
    resized_height = max(output_size, int(round(height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    y0 = max(0, (resized_height - output_size) // 2)
    x0 = max(0, (resized_width - output_size) // 2)
    return resized[y0 : y0 + output_size, x0 : x0 + output_size]


def load_pose_series(path: Path, manifest: dict) -> PoseSeries:
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"Pose CSV has no rows: {path}")

    created_at = optional_float(manifest.get("createdAtUnixTime"))
    session_start_frame_time = optional_float(manifest.get("sessionStartFrameTime"))

    times = []
    positions = []
    quaternions = []

    for row in rows:
        relative_time = first_float(row, ("relative_time", "time"))
        frame_time = first_float(row, ("frame_time",), required=False)
        sender_time = first_float(row, ("sender_time", "timestamp", "received_time"), required=False)

        if relative_time is None and frame_time is not None and session_start_frame_time is not None:
            relative_time = frame_time - session_start_frame_time

        if created_at is not None and relative_time is not None:
            time_value = created_at + relative_time
        elif sender_time is not None:
            time_value = sender_time
        elif relative_time is not None:
            time_value = relative_time
        else:
            continue

        position = [
            first_float(row, ("x", "px", "pos_x")),
            first_float(row, ("y", "py", "pos_y")),
            first_float(row, ("z", "pz", "pos_z")),
        ]
        quaternion = [
            first_float(row, ("qx",)),
            first_float(row, ("qy",)),
            first_float(row, ("qz",)),
            first_float(row, ("qw",)),
        ]

        if any(value is None for value in position + quaternion):
            continue

        times.append(time_value)
        positions.append(position)
        quaternions.append(normalize_quaternion(quaternion))

    if not times:
        raise ValueError(f"Pose CSV did not contain usable pose rows: {path}")

    order = np.argsort(np.asarray(times, dtype=np.float64))
    return PoseSeries(
        time=np.asarray(times, dtype=np.float64)[order],
        position=np.asarray(positions, dtype=np.float64)[order],
        quaternion=np.asarray(quaternions, dtype=np.float64)[order],
    )


def sample_pose_quaternion_on_time(
    pose: Optional[PoseSeries],
    sample_time: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if pose is None or len(pose.time) == 0:
        identity_quaternion = np.zeros((len(sample_time), 4), dtype=np.float32)
        identity_quaternion[:, 3] = 1.0
        return (
            np.zeros((len(sample_time), 3), dtype=np.float32),
            identity_quaternion,
        )

    unique_time, unique_indices = np.unique(pose.time, return_index=True)
    positions = pose.position[unique_indices]
    quaternions = pose.quaternion[unique_indices]

    if len(unique_time) == 1:
        sampled_pos = np.repeat(positions[0][None, :], len(sample_time), axis=0)
        sampled_quat = np.repeat(quaternions[0][None, :], len(sample_time), axis=0)
    else:
        clipped_time = np.clip(sample_time, unique_time[0], unique_time[-1])
        sampled_pos = np.column_stack(
            [np.interp(clipped_time, unique_time, positions[:, axis]) for axis in range(3)]
        )
        rotations = Rotation.from_quat(quaternions)
        slerp = Slerp(unique_time, rotations)
        sampled_quat = slerp(clipped_time).as_quat()

    return sampled_pos.astype(np.float32), sampled_quat.astype(np.float32)


def sample_pose_on_time(pose: Optional[PoseSeries], sample_time: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sampled_pos, sampled_quat = sample_pose_quaternion_on_time(pose, sample_time)
    return pose_quaternion_to_pos_rotvec(sampled_pos, sampled_quat)


def load_gripper_series(path: Optional[Path], manifest: dict) -> Optional[GripperSeries]:
    if path is None:
        return None

    rows = read_csv_rows(path)
    if not rows:
        return None

    start_unix = optional_float(
        manifest.get("experimentStartUnixTime", manifest.get("createdAtUnixTime"))
    )
    created_at = optional_float(manifest.get("createdAtUnixTime"))
    times = []
    widths_m = []

    for row in rows:
        relative_time = first_float(
            row,
            ("experiment_time", "relative_time", "video_time"),
            required=False,
        )
        absolute_time = first_float(
            row,
            ("capture_time", "sender_time", "timestamp", "time"),
            required=False,
        )

        if start_unix is not None and relative_time is not None and relative_time < 1e8:
            time_value = start_unix + relative_time
        elif absolute_time is not None and absolute_time >= 1e8:
            time_value = absolute_time
        elif created_at is not None and relative_time is not None and relative_time < 1e8:
            time_value = created_at + relative_time
        elif relative_time is not None:
            time_value = relative_time
        elif created_at is not None and absolute_time is not None and absolute_time < 1e8:
            time_value = created_at + absolute_time
        elif absolute_time is not None:
            time_value = absolute_time
        else:
            continue

        # The ArUco CSV contains both raw marker separation and the calibrated
        # physical jaw opening. Training must use the latter; raw distance is
        # only a calibration input and can exceed the gripper stroke.
        width_m = first_float(
            row,
            (
                "offline_smoothed_m",
                "filtered_m",
                "calibrated_m",
                "filtered_distance_m",
                "calibrated_distance_m",
                "gripper_width_m",
                "width_m",
            ),
            required=False,
        )
        if width_m is None:
            width_mm = first_float(
                row,
                (
                    "offline_smoothed_mm",
                    "filtered_mm",
                    "calibrated_mm",
                    "filtered_distance_mm",
                    "calibrated_distance_mm",
                    "gripper_width_mm",
                    "width_mm",
                ),
                required=False,
            )
            if width_mm is not None:
                width_m = width_mm / 1000.0

        if width_m is None or not np.isfinite(width_m) or width_m < 0:
            continue
        if not np.isfinite(time_value):
            continue

        times.append(time_value)
        widths_m.append(width_m)

    if not times:
        return None

    order = np.argsort(np.asarray(times, dtype=np.float64))
    return GripperSeries(
        time=np.asarray(times, dtype=np.float64)[order],
        width_m=np.asarray(widths_m, dtype=np.float64)[order],
    )


def sample_gripper_on_time(
    gripper: Optional[GripperSeries],
    sample_time: np.ndarray,
) -> np.ndarray:
    values = np.zeros((len(sample_time), 1), dtype=np.float32)
    if gripper is None or len(gripper.time) == 0:
        return values

    unique_time, unique_indices = np.unique(gripper.time, return_index=True)
    source = gripper.width_m[unique_indices]
    finite = np.isfinite(unique_time) & np.isfinite(source) & (source >= 0)
    unique_time = unique_time[finite]
    source = source[finite]
    if len(unique_time) == 0:
        return values
    if len(unique_time) == 1:
        values[:, 0] = source[0]
        return values

    clipped_time = np.clip(sample_time, unique_time[0], unique_time[-1])
    values[:, 0] = np.interp(clipped_time, unique_time, source)
    return values


def unwrap_magnetic_series(
    magnetic: MagneticSeries,
    *,
    period: float = MAGNET_WRAP_PERIOD,
    sensor_index: int = MAGNET_WRAP_SENSOR_INDEX,
    axis_index: int = MAGNET_WRAP_AXIS_INDEX,
    jump_threshold: Optional[float] = None,
) -> Tuple[MagneticSeries, Dict[str, object]]:
    """Recover one signed magnetic channel that wraps at a fixed ADC period.

    The correction is performed on the raw, time-sorted samples.  For a large
    raw step, the selected value is shifted by an integer number of periods so
    that it remains closest to the previous recovered value.  Other channels
    and non-finite samples are left untouched.
    """
    values = np.asarray(magnetic.values, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != MAGNETIC_VALUE_DIM:
        raise ValueError(f"Expected magnetic values [T, chip, value], got {values.shape}")
    if not np.isfinite(period) or period <= 0:
        raise ValueError(f"Magnetic wrap period must be positive, got {period}")
    if not 0 <= sensor_index < values.shape[1]:
        raise ValueError(f"Magnetic wrap sensor index {sensor_index} is outside {values.shape[1]} chips")
    if not 0 <= axis_index < values.shape[2]:
        raise ValueError(f"Magnetic wrap axis index {axis_index} is outside {values.shape[2]} values")

    threshold = period * 0.5 if jump_threshold is None else float(jump_threshold)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError(f"Magnetic wrap jump threshold must be positive, got {threshold}")

    recovered_values = values.copy()
    raw = values[:, sensor_index, axis_index]
    recovered = recovered_values[:, sensor_index, axis_index]
    finite_indices = np.flatnonzero(np.isfinite(raw))
    wrap_count = 0
    offset_periods = 0
    previous_raw = None
    previous_recovered = None

    for sample_index in finite_indices:
        current_raw = float(raw[sample_index])
        if previous_recovered is not None and abs(current_raw - previous_raw) >= threshold:
            candidate = current_raw + offset_periods * period
            correction = int(round((previous_recovered - candidate) / period))
            if correction:
                offset_periods += correction
                wrap_count += abs(correction)
        current_recovered = current_raw + offset_periods * period
        recovered[sample_index] = current_recovered
        previous_raw = current_raw
        previous_recovered = current_recovered

    raw_steps = np.diff(raw[finite_indices]) if len(finite_indices) > 1 else np.empty(0)
    recovered_steps = np.diff(recovered[finite_indices]) if len(finite_indices) > 1 else np.empty(0)
    report: Dict[str, object] = {
        "enabled": True,
        "period": float(period),
        "sensor_index": int(sensor_index),
        "axis_index": int(axis_index),
        "jump_threshold": float(threshold),
        "wrap_count": int(wrap_count),
        "max_raw_step": float(np.max(np.abs(raw_steps))) if len(raw_steps) else 0.0,
        "max_recovered_step": float(np.max(np.abs(recovered_steps))) if len(recovered_steps) else 0.0,
    }
    return MagneticSeries(time=magnetic.time.copy(), values=recovered_values), report


def load_magnetic_series(
    path: Optional[Path],
    manifest: dict,
    *,
    unwrap: bool = MAGNET_WRAP_RECOVERY_ENABLED,
    wrap_period: float = MAGNET_WRAP_PERIOD,
    wrap_sensor_index: int = MAGNET_WRAP_SENSOR_INDEX,
    wrap_axis_index: int = MAGNET_WRAP_AXIS_INDEX,
    wrap_jump_threshold: Optional[float] = None,
) -> Optional[MagneticSeries]:
    if path is None:
        return None
    rows = read_csv_rows(path)
    if not rows:
        return None

    start_unix = optional_float(
        manifest.get("experimentStartUnixTime", manifest.get("createdAtUnixTime"))
    )
    start_monotonic = optional_float(
        manifest.get("experimentStartMonotonicTime", manifest.get("sessionStartFrameTime"))
    )
    times = []
    samples = []
    for row in rows:
        relative_time = first_float(row, ("relative_time", "experiment_time"), required=False)
        receive_time = first_float(row, ("phone_receive_time", "sender_time"), required=False)
        monotonic_time = first_float(row, ("phone_monotonic_time",), required=False)
        if start_unix is not None and relative_time is not None:
            time_value = start_unix + relative_time
        elif receive_time is not None:
            time_value = receive_time
        elif start_unix is not None and start_monotonic is not None and monotonic_time is not None:
            time_value = start_unix + monotonic_time - start_monotonic
        elif relative_time is not None:
            time_value = relative_time
        else:
            continue

        chip_values = []
        for chip in range(MAGNETIC_CHIP_COUNT):
            chip_values.append(
                [
                    first_float(row, (f"s{chip}_t",), required=False),
                    first_float(row, (f"s{chip}_x",), required=False),
                    first_float(row, (f"s{chip}_y",), required=False),
                    first_float(row, (f"s{chip}_z",), required=False),
                ]
            )
        values = np.asarray(
            [[np.nan if value is None else value for value in chip] for chip in chip_values],
            dtype=np.float64,
        )
        if not np.isfinite(values).any():
            continue
        times.append(time_value)
        samples.append(values)

    if not times:
        return None
    order = np.argsort(np.asarray(times, dtype=np.float64))
    magnetic = MagneticSeries(
        time=np.asarray(times, dtype=np.float64)[order],
        values=np.asarray(samples, dtype=np.float64)[order],
    )
    if not unwrap:
        return magnetic

    magnetic, report = unwrap_magnetic_series(
        magnetic,
        period=wrap_period,
        sensor_index=wrap_sensor_index,
        axis_index=wrap_axis_index,
        jump_threshold=wrap_jump_threshold,
    )
    if report["wrap_count"]:
        print(
            f"[INFO] {path.name}: recovered {report['wrap_count']} magnetic wrap(s) "
            f"on s{wrap_sensor_index}_{('t', 'x', 'y', 'z')[wrap_axis_index]} "
            f"(max step {report['max_raw_step']:.1f} -> {report['max_recovered_step']:.1f})"
        )
    return magnetic


def sample_magnetic_on_time(
    magnetic: Optional[MagneticSeries],
    sample_time: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    values = np.zeros(
        (len(sample_time), MAGNETIC_CHIP_COUNT, MAGNETIC_VALUE_DIM),
        dtype=np.float32,
    )
    valid = np.zeros((len(sample_time), MAGNETIC_CHIP_COUNT), dtype=bool)
    if magnetic is None or len(magnetic.time) == 0:
        return values, valid

    unique_time, unique_indices = np.unique(magnetic.time, return_index=True)
    source = magnetic.values[unique_indices]
    for chip in range(MAGNETIC_CHIP_COUNT):
        finite_rows = np.all(np.isfinite(source[:, chip, :]), axis=1)
        chip_time = unique_time[finite_rows]
        chip_values = source[finite_rows, chip, :]
        if len(chip_time) == 0:
            continue
        if len(chip_time) == 1:
            nearest = np.isclose(sample_time, chip_time[0], atol=1e-3)
            values[nearest, chip, :] = chip_values[0]
            valid[nearest, chip] = True
            continue
        in_range = (sample_time >= chip_time[0]) & (sample_time <= chip_time[-1])
        for axis in range(MAGNETIC_VALUE_DIM):
            values[in_range, chip, axis] = np.interp(
                sample_time[in_range],
                chip_time,
                chip_values[:, axis],
            )
        valid[in_range, chip] = True
    return values, valid


def load_force_series(path: Optional[Path], manifest: dict) -> Optional[ForceSeries]:
    if path is None:
        return None

    rows = read_csv_rows(path)
    if not rows:
        return None

    created_at = optional_float(manifest.get("createdAtUnixTime"))
    times = []
    values = []

    for row in rows:
        force = [
            first_float(row, ("fx", "force_x", "f_x", "force0", "force_0"), required=False),
            first_float(row, ("fy", "force_y", "f_y", "force1", "force_1"), required=False),
            first_float(row, ("fz", "force_z", "f_z", "force2", "force_2"), required=False),
            first_float(row, ("tx", "torque_x", "t_x", "mx", "moment_x", "force3", "force_3"), required=False),
            first_float(row, ("ty", "torque_y", "t_y", "my", "moment_y", "force4", "force_4"), required=False),
            first_float(row, ("tz", "torque_z", "t_z", "mz", "moment_z", "force5", "force_5"), required=False),
        ]
        if any(value is None for value in force):
            continue

        time_value = first_float(
            row,
            (
                "phone_receive_time",
                "received_time",
                "receive_time",
                "force_time",
                "aligned_time",
                "timestamp",
                "time",
                "relative_time",
            ),
            required=False,
        )
        if time_value is None:
            continue

        if created_at is not None and time_value < 1e8:
            time_value = created_at + time_value

        times.append(time_value)
        values.append(force)

    if not times:
        return None

    order = np.argsort(np.asarray(times, dtype=np.float64))
    return ForceSeries(
        time=np.asarray(times, dtype=np.float64)[order],
        values=np.asarray(values, dtype=np.float64)[order],
    )


def sample_force_on_time(force: Optional[ForceSeries], sample_time: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    values = np.zeros((len(sample_time), FORCE_DIM), dtype=np.float32)
    valid = np.zeros(len(sample_time), dtype=bool)

    if force is None or len(force.time) == 0:
        return values, valid

    unique_time, unique_indices = np.unique(force.time, return_index=True)
    force_values = force.values[unique_indices]

    if len(unique_time) == 1:
        nearest = np.isclose(sample_time, unique_time[0], atol=1e-3)
        values[nearest] = force_values[0]
        valid[nearest] = True
        return values, valid

    in_range = (sample_time >= unique_time[0]) & (sample_time <= unique_time[-1])
    for axis in range(FORCE_DIM):
        values[in_range, axis] = np.interp(sample_time[in_range], unique_time, force_values[:, axis])
    valid[in_range] = True
    return values, valid


def write_zarr(
    out_path: Path,
    episodes: List[EpisodeArrays],
    attrs: Optional[Dict[str, object]] = None,
) -> None:
    if not episodes:
        raise ValueError("No episodes to write.")

    camera_rgb = np.concatenate([ep.camera_rgb for ep in episodes], axis=0)
    timestamp = np.concatenate([ep.timestamp for ep in episodes], axis=0)
    eef_pos = np.concatenate([ep.eef_pos for ep in episodes], axis=0)
    eef_rot_axis_angle = np.concatenate([ep.eef_rot_axis_angle for ep in episodes], axis=0)
    gripper_width = np.concatenate([ep.gripper_width for ep in episodes], axis=0)
    demo_start_pose = np.concatenate([ep.demo_start_pose for ep in episodes], axis=0)
    demo_end_pose = np.concatenate([ep.demo_end_pose for ep in episodes], axis=0)
    action = np.concatenate([ep.action for ep in episodes], axis=0)
    force_torque = np.concatenate([ep.force_torque for ep in episodes], axis=0)
    force_valid = np.concatenate([ep.force_valid for ep in episodes], axis=0)
    magnet_xyz = np.concatenate([ep.magnet_xyz for ep in episodes], axis=0)
    magnet_timestamp_ns = np.concatenate([ep.magnet_timestamp_ns for ep in episodes], axis=0)
    magnet_sample_count = np.concatenate([ep.magnet_sample_count for ep in episodes], axis=0)
    magnetic_txyz = np.concatenate([ep.magnetic_txyz for ep in episodes], axis=0)
    magnetic_valid = np.concatenate([ep.magnetic_valid for ep in episodes], axis=0)
    magnetic_left_txyz = np.concatenate(
        [ep.magnetic_left_txyz for ep in episodes], axis=0
    )
    magnetic_left_valid = np.concatenate(
        [ep.magnetic_left_valid for ep in episodes], axis=0
    )
    episode_ends = np.cumsum([len(ep.timestamp) for ep in episodes], dtype=np.int64)

    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)

    root = zarr.open_group(str(out_path), mode="w")
    data_group = root.create_group("data")
    meta_group = root.create_group("meta")

    write_array(data_group, "camera0_rgb", camera_rgb, (1, camera_rgb.shape[1], camera_rgb.shape[2], 3), compressor)
    write_array(data_group, "timestamp", timestamp, (ZARR_CHUNK_ROWS,), compressor)
    write_array(data_group, "robot0_eef_pos", eef_pos, (ZARR_CHUNK_ROWS, 3), compressor)
    write_array(data_group, "robot0_eef_rot_axis_angle", eef_rot_axis_angle, (ZARR_CHUNK_ROWS, 3), compressor)
    write_array(data_group, "robot0_gripper_width", gripper_width, (ZARR_CHUNK_ROWS, 1), compressor)
    write_array(data_group, "robot0_demo_start_pose", demo_start_pose, (ZARR_CHUNK_ROWS, POSE_DIM), compressor)
    write_array(data_group, "robot0_demo_end_pose", demo_end_pose, (ZARR_CHUNK_ROWS, POSE_DIM), compressor)
    write_array(data_group, "action", action, (ZARR_CHUNK_ROWS, ACTION_DIM), compressor)
    write_array(data_group, "force_torque", force_torque, (ZARR_CHUNK_ROWS, FORCE_DIM), compressor)
    write_array(data_group, "force_valid", force_valid, (ZARR_CHUNK_ROWS,), compressor)
    write_array(
        data_group,
        "magnet_xyz",
        magnet_xyz,
        (ZARR_CHUNK_ROWS, magnet_xyz.shape[1], magnet_xyz.shape[2], magnet_xyz.shape[3]),
        compressor,
    )
    write_array(
        data_group,
        "magnet_timestamp_ns",
        magnet_timestamp_ns,
        (ZARR_CHUNK_ROWS, magnet_timestamp_ns.shape[1]),
        compressor,
    )
    write_array(
        data_group,
        "magnet_sample_count",
        magnet_sample_count,
        (ZARR_CHUNK_ROWS, magnet_sample_count.shape[1]),
        compressor,
    )
    write_array(
        data_group,
        "magnetic_txyz",
        magnetic_txyz,
        (ZARR_CHUNK_ROWS, MAGNETIC_CHIP_COUNT, MAGNETIC_VALUE_DIM),
        compressor,
    )
    write_array(
        data_group,
        "magnetic_valid",
        magnetic_valid,
        (ZARR_CHUNK_ROWS, MAGNETIC_CHIP_COUNT),
        compressor,
    )
    write_array(
        data_group,
        "magnetic_left_txyz",
        magnetic_left_txyz,
        (ZARR_CHUNK_ROWS, MAGNETIC_CHIP_COUNT, MAGNETIC_VALUE_DIM),
        compressor,
    )
    write_array(
        data_group,
        "magnetic_left_valid",
        magnetic_left_valid,
        (ZARR_CHUNK_ROWS, MAGNETIC_CHIP_COUNT),
        compressor,
    )
    write_array(meta_group, "episode_ends", episode_ends, (len(episode_ends),), None)
    if attrs:
        root.attrs.update(attrs)


def write_array(group, name: str, values: np.ndarray, chunks: Tuple[int, ...], compressor) -> None:
    group.create_dataset(
        name,
        data=values,
        shape=values.shape,
        chunks=chunks,
        dtype=values.dtype,
        compressor=compressor,
        overwrite=True,
    )


def find_capture_file(
    directory: Path,
    logical_name: Optional[str],
    component_prefix: Optional[str],
    fallback_patterns: Iterable[str],
    require_nonempty: bool,
    required: bool = True,
) -> Optional[Path]:
    candidates: List[Path] = []

    if logical_name:
        candidates.append(directory / logical_name)
        if component_prefix:
            candidates.append(directory / f"{component_prefix}__{logical_name}")

    for pattern in fallback_patterns:
        candidates.extend(sorted(directory.glob(pattern)))

    for candidate in unique_paths(candidates):
        if candidate.is_file() and (not require_nonempty or candidate.stat().st_size > 0):
            return candidate

    if required:
        raise FileNotFoundError(f"Could not find required file in {directory}: {', '.join(fallback_patterns)}")
    return None


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in unique_paths(paths):
        if path.is_file():
            return path
    return None


def find_non_ultrawide_video_file(directory: Path) -> Optional[Path]:
    candidates = []
    for pattern in ("*.mp4", "*.mov", "*.m4v"):
        for path in sorted(directory.glob(pattern)):
            if path.name.lower().startswith("ultrawide_video"):
                continue
            candidates.append(path)
    return first_existing(candidates)


def unique_paths(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    result = []
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return [{normalize_key(key): value for key, value in row.items()} for row in reader]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_key(key: Optional[str]) -> str:
    return (key or "").strip().lower()


def optional_float(value) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def first_float(row: Dict[str, str], names: Iterable[str], required: bool = True) -> Optional[float]:
    for name in names:
        value = optional_float(row.get(normalize_key(name)))
        if value is not None:
            return value
    if required:
        raise ValueError(f"Missing numeric CSV field. Tried: {', '.join(names)}")
    return None


def normalize_quaternion(values: Iterable[float]) -> List[float]:
    quat = np.asarray(list(values), dtype=np.float64)
    norm = np.linalg.norm(quat)
    if not np.isfinite(norm) or norm <= 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return (quat / norm).tolist()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
