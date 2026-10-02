#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export CAMERA_PREPROCESS_MODE="${CAMERA_PREPROCESS_MODE:-square_crop}"
export CAMERA_COLOR_MATCH_DATASET_START="${CAMERA_COLOR_MATCH_DATASET_START:-True}"
export CAMERA_COLOR_MATCH_DATASET_EPISODE="${CAMERA_COLOR_MATCH_DATASET_EPISODE:-0}"
export CAMERA_COLOR_MATCH_TIMEOUT="${CAMERA_COLOR_MATCH_TIMEOUT:-2.0}"
export CAMERA_IPHONE_STARTUP_TIMEOUT="${CAMERA_IPHONE_STARTUP_TIMEOUT:-10.0}"
export OPEN_GRIPPER_ON_START="${OPEN_GRIPPER_ON_START:-True}"
export IGNORE_POLICY_GRIPPER_COMMANDS="${IGNORE_POLICY_GRIPPER_COMMANDS:-False}"
export MOVE_TO_START_POSE_PATH="${MOVE_TO_START_POSE_PATH:-dataset/current_pose.json}"
export RELATIVE_ACTION="${RELATIVE_ACTION:-True}"
export CAMERA_BACKEND="${CAMERA_BACKEND:-iphone}"
export CAMERA_SOURCE="${CAMERA_SOURCE:-auto}"
export MOVE_TO_START_DURATION="${MOVE_TO_START_DURATION:-5.0}"
export MOVE_TO_START_FREQUENCY="${MOVE_TO_START_FREQUENCY:-20.0}"

# Use the current training dataset's gripper-label range. Do not collapse
# near-open policy outputs into a fixed full-open command.
export LATENCY_STEP="${LATENCY_STEP:-2}"
export GRIPPER_LATENCY_STEP="${GRIPPER_LATENCY_STEP:-4}"
export TCP_ACTION_UPDATE_INTERVAL="${TCP_ACTION_UPDATE_INTERVAL:-8}"
export GRIPPER_ACTION_UPDATE_INTERVAL="${GRIPPER_ACTION_UPDATE_INTERVAL:-8}"
export OBS_TEMPORAL_DOWNSAMPLE_RATIO="${OBS_TEMPORAL_DOWNSAMPLE_RATIO:-2}"
export TCP_ENSEMBLE_MODE="${TCP_ENSEMBLE_MODE:-new}"
export GRIPPER_ENSEMBLE_MODE="${GRIPPER_ENSEMBLE_MODE:-new}"
# Relative TCP observations/actions are already expressed in the current TCP frame.
# Applying the dataset-to-robot tool rotation again reverses the horizontal motion.
export POLICY_TCP_POSE_OBS_MODE="none"
export POLICY_RELATIVE_ACTION_FRAME_MODE="none"

export TASK="${TASK:-franka_polymetis_image_dp_absolute_12fps}"
export CKPT_PATH="${CKPT_PATH:-data/outputs/2026.07.24/20.39.10_train_diffusion_unet_image_real_wipe_image_dp_absolute_12fps/checkpoints/latest.ckpt}"
export DATASET_PATH="${DATASET_PATH:-dataset/traj_rdp10d_command_downsample2}"
export OUTPUT_DIR="${OUTPUT_DIR:-data/eval_outputs/franka_polymetis_iphone_aligned}"
export MAX_DURATION="${MAX_DURATION:-100}"

exec bash eval.sh
