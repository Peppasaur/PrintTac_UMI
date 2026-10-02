#!/usr/bin/env bash
set -euo pipefail

FR3_ROBOT_IP="${FR3_ROBOT_IP:-172.16.0.2}"
POLYMETIS_GRIPPER_HOST="${POLYMETIS_GRIPPER_HOST:-127.0.0.1}"
POLYMETIS_GRIPPER_PORT="${POLYMETIS_GRIPPER_PORT:-50052}"
GRIPPER_TYPE="${GRIPPER_TYPE:-robotiq_2f}"
ROBOTIQ_PORT="${ROBOTIQ_PORT:-/dev/ttyUSB1}"
ROBOTIQ_STROKE="${ROBOTIQ_STROKE:-0.085}"
ROBOTIQ_ACTIVATION_TIMEOUT="${ROBOTIQ_ACTIVATION_TIMEOUT:-10.0}"
ROBOTIQ_RESET_BEFORE_ACTIVATE="${ROBOTIQ_RESET_BEFORE_ACTIVATE:-0}"

echo "Starting Polymetis gripper server on ${POLYMETIS_GRIPPER_HOST}:${POLYMETIS_GRIPPER_PORT}"
echo "Gripper type: ${GRIPPER_TYPE}"
echo "Leave this terminal running. Press Ctrl+C to stop the gripper server."

if [[ "${GRIPPER_TYPE}" == "franka_hand" ]]; then
  echo "Franka Hand robot IP: ${FR3_ROBOT_IP}"
  exec conda run --no-capture-output -n polymetis launch_gripper.py \
    gripper=franka_hand \
    gripper.executable_cfg.robot_ip="${FR3_ROBOT_IP}" \
    ip="${POLYMETIS_GRIPPER_HOST}" \
    port="${POLYMETIS_GRIPPER_PORT}"
fi

if [[ "${GRIPPER_TYPE}" == "robotiq_2f" ]]; then
  echo "Robotiq serial port: ${ROBOTIQ_PORT}"
  ROBOTIQ_ARGS=()
  if [[ "${ROBOTIQ_RESET_BEFORE_ACTIVATE}" == "1" || "${ROBOTIQ_RESET_BEFORE_ACTIVATE}" == "true" ]]; then
    ROBOTIQ_ARGS+=(--reset-before-activate)
  fi
  exec conda run --no-capture-output -n polymetis python -u robotiq_polymetis_gripper_server.py \
    --host "${POLYMETIS_GRIPPER_HOST}" \
    --port "${POLYMETIS_GRIPPER_PORT}" \
    --comport "${ROBOTIQ_PORT}" \
    --stroke "${ROBOTIQ_STROKE}" \
    --activation-timeout "${ROBOTIQ_ACTIVATION_TIMEOUT}" \
    "${ROBOTIQ_ARGS[@]}"
fi

echo "Unsupported GRIPPER_TYPE=${GRIPPER_TYPE}. Use robotiq_2f or franka_hand." >&2
exit 2
