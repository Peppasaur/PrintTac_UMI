#!/usr/bin/env bash
set -euo pipefail

GRIPPER_WIDTH_MM="${GRIPPER_WIDTH_MM:?Set GRIPPER_WIDTH_MM, for example GRIPPER_WIDTH_MM=44}"
DURATION="${DURATION:-10}"
SETTLE_SEC="${SETTLE_SEC:-0.5}"
OUTPUT_DIR="${OUTPUT_DIR:-data/eval_outputs/franka_polymetis/gripper_magnet}"
ROBOT_SERVER_HOST="${ROBOT_SERVER_HOST:-127.0.0.1}"
ROBOT_SERVER_PORT="${ROBOT_SERVER_PORT:-8092}"
DATASET_PATH="${DATASET_PATH:-dataset/traj_rdp10d_command_downsample5}"
FPS="${FPS:-12}"
CAMERA_BACKEND="${CAMERA_BACKEND:-iphone}"
CAMERA_SOURCE="${CAMERA_SOURCE:-auto}"
CAMERA_PREPROCESS_MODE="${CAMERA_PREPROCESS_MODE:-square_crop}"
CAMERA_SQUARE_CROP_BOTTOM_ROWS="${CAMERA_SQUARE_CROP_BOTTOM_ROWS:-0}"
MAGNET_PORT="${MAGNET_PORT:-/dev/ttyACM0}"
MAGNET_BAUDRATE="${MAGNET_BAUDRATE:-115200}"
MAGNET_SENSOR_ORDER="${MAGNET_SENSOR_ORDER:-4,1,2,3}"
GRIPPER_STROKE="${GRIPPER_STROKE:-0.085}"
GRASP_FORCE="${GRASP_FORCE:-20.0}"
GRIPPER_VELOCITY="${GRIPPER_VELOCITY:-0.08}"

conda run --no-capture-output -n umi python scripts/record_magnet_after_gripper_width.py \
  --gripper-width-mm "${GRIPPER_WIDTH_MM}" \
  --duration "${DURATION}" \
  --settle-sec "${SETTLE_SEC}" \
  --output-dir "${OUTPUT_DIR}" \
  --robot-server-host "${ROBOT_SERVER_HOST}" \
  --robot-server-port "${ROBOT_SERVER_PORT}" \
  --dataset-path "${DATASET_PATH}" \
  --fps "${FPS}" \
  --camera-backend "${CAMERA_BACKEND}" \
  --camera-source "${CAMERA_SOURCE}" \
  --camera-preprocess-mode "${CAMERA_PREPROCESS_MODE}" \
  --camera-square-crop-bottom-rows "${CAMERA_SQUARE_CROP_BOTTOM_ROWS}" \
  --magnet-port "${MAGNET_PORT}" \
  --magnet-baudrate "${MAGNET_BAUDRATE}" \
  --magnet-sensor-order "${MAGNET_SENSOR_ORDER}" \
  --gripper-stroke "${GRIPPER_STROKE}" \
  --grasp-force "${GRASP_FORCE}" \
  --gripper-velocity "${GRIPPER_VELOCITY}"
