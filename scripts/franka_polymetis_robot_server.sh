#!/usr/bin/env bash
set -euo pipefail

RDP_ROBOT_SERVER_HOST="${RDP_ROBOT_SERVER_HOST:-127.0.0.1}"
RDP_ROBOT_SERVER_PORT="${RDP_ROBOT_SERVER_PORT:-8092}"
POLYMETIS_HOST="${POLYMETIS_HOST:-127.0.0.1}"
POLYMETIS_PORT="${POLYMETIS_PORT:-50051}"
POLYMETIS_GRIPPER_HOST="${POLYMETIS_GRIPPER_HOST:-127.0.0.1}"
POLYMETIS_GRIPPER_PORT="${POLYMETIS_GRIPPER_PORT:-50052}"
FRANKA_CONTROL_FREQUENCY="${FRANKA_CONTROL_FREQUENCY:-300}"
FRANKA_VR_FREQUENCY="${FRANKA_VR_FREQUENCY:-60}"
FRANKA_KX_SCALE="${FRANKA_KX_SCALE:-1.0}"
FRANKA_KXD_SCALE="${FRANKA_KXD_SCALE:-1.0}"
RDP_AUTO_CLEANUP="${RDP_AUTO_CLEANUP:-1}"
ENABLE_GRIPPER="${ENABLE_GRIPPER:-1}"
DEFAULT_GRIPPER_WIDTH="${DEFAULT_GRIPPER_WIDTH:-0.080}"

case "${ENABLE_GRIPPER}" in
  1|true|TRUE|True|yes|YES|on|ON)
    ENABLE_GRIPPER_NORMALIZED=1
    ;;
  0|false|FALSE|False|no|NO|off|OFF)
    ENABLE_GRIPPER_NORMALIZED=0
    ;;
  *)
    echo "ERROR: ENABLE_GRIPPER must be 1/0, true/false, yes/no, or on/off; got '${ENABLE_GRIPPER}'."
    exit 1
    ;;
esac

kill_pids() {
  local signal="$1"
  shift
  [[ "$#" -eq 0 ]] && return 0
  kill "-${signal}" "$@" >/dev/null 2>&1 || true
}

wait_pids_gone() {
  local pids=("$@")
  local pid
  for _ in {1..30}; do
    local any_alive=0
    for pid in "${pids[@]}"; do
      if kill -0 "${pid}" >/dev/null 2>&1; then
        any_alive=1
        break
      fi
    done
    [[ "${any_alive}" -eq 0 ]] && return 0
    sleep 0.1
  done
  return 1
}

cleanup_stale_http_server() {
  [[ "${RDP_AUTO_CLEANUP}" == "1" ]] || return 1

  local pids=()
  local pid args
  while read -r pid; do
    [[ -z "${pid}" ]] && continue
    args="$(ps -p "${pid}" -o args= 2>/dev/null || true)"
    if [[ "${args}" == *"reactive_diffusion_policy.real_world.robot.franka_server"* ]]; then
      pids+=("${pid}")
    else
      echo "ERROR: ${RDP_ROBOT_SERVER_HOST}:${RDP_ROBOT_SERVER_PORT} is occupied by a non-RDP process:"
      echo "  pid=${pid} ${args}"
      echo "Refusing to kill it automatically."
      return 1
    fi
  done < <(lsof -tiTCP:"${RDP_ROBOT_SERVER_PORT}" -sTCP:LISTEN 2>/dev/null || true)

  [[ "${#pids[@]}" -gt 0 ]] || return 0
  echo "Cleaning stale RDP Franka HTTP server processes on port ${RDP_ROBOT_SERVER_PORT}: ${pids[*]}"

  kill_pids INT "${pids[@]}"
  wait_pids_gone "${pids[@]}" && return 0

  echo "Stale RDP Franka HTTP server did not stop after SIGINT; sending SIGTERM..."
  kill_pids TERM "${pids[@]}"
  wait_pids_gone "${pids[@]}" && return 0

  echo "Stale RDP Franka HTTP server still alive; sending SIGKILL..."
  kill_pids KILL "${pids[@]}"
}

echo "Starting original RDP FrankaServer on ${RDP_ROBOT_SERVER_HOST}:${RDP_ROBOT_SERVER_PORT}"
echo "Arm Polymetis endpoint: ${POLYMETIS_HOST}:${POLYMETIS_PORT}"
if [[ "${ENABLE_GRIPPER_NORMALIZED}" == "1" ]]; then
  echo "Gripper Polymetis endpoint: ${POLYMETIS_GRIPPER_HOST}:${POLYMETIS_GRIPPER_PORT}"
else
  echo "Gripper disabled; reported default width: ${DEFAULT_GRIPPER_WIDTH}"
fi
echo "Control frequency: ${FRANKA_CONTROL_FREQUENCY}Hz"
echo "Auto cleanup stale HTTP server: ${RDP_AUTO_CLEANUP}"

if lsof -nP -iTCP:"${RDP_ROBOT_SERVER_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  if ! cleanup_stale_http_server; then
    echo "ERROR: ${RDP_ROBOT_SERVER_HOST}:${RDP_ROBOT_SERVER_PORT} is still listening."
    echo "Stop the existing process manually or set RDP_AUTO_CLEANUP=1."
    exit 1
  fi
fi

SERVER_PID=""

server_alive() {
  [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1
}

stop_server_group() {
  local exit_code="${1:-0}"
  trap - INT TERM EXIT

  if server_alive; then
    echo "Stopping Franka robot server ${SERVER_PID}..."
    pkill -INT -P "${SERVER_PID}" >/dev/null 2>&1 || true
    kill -INT "${SERVER_PID}" >/dev/null 2>&1 || true
    for _ in {1..30}; do
      server_alive || break
      sleep 0.1
    done
  fi

  if server_alive; then
    echo "Franka robot server did not stop after SIGINT; sending SIGTERM..."
    pkill -TERM -P "${SERVER_PID}" >/dev/null 2>&1 || true
    kill -TERM "${SERVER_PID}" >/dev/null 2>&1 || true
    for _ in {1..30}; do
      server_alive || break
      sleep 0.1
    done
  fi

  if server_alive; then
    echo "Franka robot server still alive; sending SIGKILL."
    pkill -KILL -P "${SERVER_PID}" >/dev/null 2>&1 || true
    kill -KILL "${SERVER_PID}" >/dev/null 2>&1 || true
  fi

  if [[ -n "${SERVER_PID}" ]]; then
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi

  exit "${exit_code}"
}

trap 'stop_server_group 130' INT
trap 'stop_server_group 143' TERM
trap 'stop_server_group $?' EXIT

SERVER_ARGS=(
  -m reactive_diffusion_policy.real_world.robot.franka_server
  --robot_ip "${POLYMETIS_HOST}"
  --robot_port "${POLYMETIS_PORT}"
  --gripper_ip "${POLYMETIS_GRIPPER_HOST}"
  --gripper_port "${POLYMETIS_GRIPPER_PORT}"
  --host_ip "${RDP_ROBOT_SERVER_HOST}"
  --port "${RDP_ROBOT_SERVER_PORT}"
  --frequency "${FRANKA_CONTROL_FREQUENCY}"
  --vr_frequency "${FRANKA_VR_FREQUENCY}"
  --Kx_scale "${FRANKA_KX_SCALE}"
  --Kxd_scale "${FRANKA_KXD_SCALE}"
  --default_gripper_width "${DEFAULT_GRIPPER_WIDTH}"
)

if [[ "${ENABLE_GRIPPER_NORMALIZED}" == "0" ]]; then
  SERVER_ARGS+=(--disable_gripper)
fi

if [[ "${CONDA_DEFAULT_ENV:-}" == "polymetis" ]]; then
  python "${SERVER_ARGS[@]}" &
else
  conda run --no-capture-output -n polymetis python "${SERVER_ARGS[@]}" &
fi

SERVER_PID=$!
wait "${SERVER_PID}"
