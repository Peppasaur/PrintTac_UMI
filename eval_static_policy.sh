#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Run policy inference and recording while suppressing every robot command.
# These safety settings intentionally override values inherited from the shell.
export OPEN_GRIPPER_ON_START=False
export START_GRIPPER_WIDTH_MM=""
export FIXED_GRIPPER_WIDTH_MM=""
export MOVE_TO_START=False
export IGNORE_GRIPPER_COMMANDS=True
export IGNORE_POLICY_TCP_COMMANDS=True
export IGNORE_POLICY_GRIPPER_COMMANDS=True
export ASK_RESET_CONFIRMATION=False

export DATA_DIR="${DATA_DIR:-../data}"
export OUTPUT_DIR="${OUTPUT_DIR:-${DATA_DIR}/eval_outputs/franka_polymetis_static_policy}"
export MAX_DURATION="${MAX_DURATION:-10}"
export EVAL_EPISODES="${EVAL_EPISODES:-1}"
export ENABLE_POLICY_RECORDING="${ENABLE_POLICY_RECORDING:-True}"

echo "Static policy eval: TCP, gripper, startup, and reset motion are disabled."
echo "Inference and policy recording remain enabled (duration=${MAX_DURATION}s)."

exec bash "${SCRIPT_DIR}/eval.sh"
