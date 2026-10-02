#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Static eval: run policy inference and recording without sending robot commands.
export OPEN_GRIPPER_ON_START=False
export MOVE_TO_START=False
export IGNORE_GRIPPER_COMMANDS=True
export IGNORE_POLICY_TCP_COMMANDS=True
export IGNORE_POLICY_GRIPPER_COMMANDS=True
export ASK_RESET_CONFIRMATION=False

export CAMERA_PREPROCESS_MODE="${CAMERA_PREPROCESS_MODE:-square_crop}"
export CAMERA_BACKEND="${CAMERA_BACKEND:-iphone}"
export CAMERA_SOURCE="${CAMERA_SOURCE:-auto}"
export RELATIVE_ACTION="${RELATIVE_ACTION:-True}"
export MAGNET_SENSOR_ORDER="${MAGNET_SENSOR_ORDER:-4,1,2,3}"

export TASK="${TASK:-franka_polymetis_image_gelsight_emb_dp_absolute_12fps}"
export CKPT_PATH="${CKPT_PATH:-data/outputs/2026.07.26/16.43.01_train_diffusion_unet_image_real_wipe_image_gelsight_emb_dp_absolute_12fps/checkpoints/latest.ckpt}"
export DATASET_PATH="${DATASET_PATH:-dataset/traj_rdp10d_command_downsample2}"
export OUTPUT_DIR="${OUTPUT_DIR:-../data/eval_outputs/franka_polymetis_static_10s}"
export MAX_DURATION="${MAX_DURATION:-10}"
export EVAL_EPISODES="${EVAL_EPISODES:-1}"
export ENABLE_POLICY_RECORDING="${ENABLE_POLICY_RECORDING:-True}"

exec bash eval.sh
