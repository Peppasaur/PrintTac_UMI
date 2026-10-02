#!/usr/bin/env bash
set -euo pipefail

FR3_ROBOT_IP="${FR3_ROBOT_IP:-172.16.0.2}"
POLYMETIS_HOST="${POLYMETIS_HOST:-127.0.0.1}"
POLYMETIS_PORT="${POLYMETIS_PORT:-50051}"
FRANKA_CARTESIAN_POS_LOWER="${FRANKA_CARTESIAN_POS_LOWER:-[0.1,-0.55,-0.05]}"
FRANKA_CARTESIAN_POS_UPPER="${FRANKA_CARTESIAN_POS_UPPER:-[1.0,0.4,1.0]}"
FRANKA_SAFETY_CARTESIAN_MARGIN="${FRANKA_SAFETY_CARTESIAN_MARGIN:-0.05}"
POLYMETIS_USE_REAL_TIME="${POLYMETIS_USE_REAL_TIME:-true}"
POLYMETIS_AUTO_CLEANUP="${POLYMETIS_AUTO_CLEANUP:-1}"

collect_pids() {
  local pattern="$1"
  local pid
  pgrep -f "${pattern}" 2>/dev/null | while read -r pid; do
    [[ -z "${pid}" || "${pid}" == "$$" || "${pid}" == "${PPID}" ]] && continue
    echo "${pid}"
  done
}

append_unique_pid() {
  local new_pid="$1"
  local existing
  [[ -n "${new_pid}" ]] || return 0
  for existing in "${pids[@]}"; do
    [[ "${existing}" == "${new_pid}" ]] && return 0
  done
  pids+=("${new_pid}")
}

kill_pids() {
  local signal="$1"
  shift
  [[ "$#" -eq 0 ]] && return 0
  kill "-${signal}" "$@" >/dev/null 2>&1 || true
  if command -v sudo >/dev/null 2>&1; then
    sudo -n kill "-${signal}" "$@" >/dev/null 2>&1 || true
  fi
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

cleanup_stale_arm_server() {
  [[ "${POLYMETIS_AUTO_CLEANUP}" == "1" ]] || return 0

  local pids=()
  local pid
  while read -r pid; do append_unique_pid "${pid}"; done < <(
    collect_pids "run_server -s ${POLYMETIS_HOST} -p ${POLYMETIS_PORT}"
  )
  while read -r pid; do append_unique_pid "${pid}"; done < <(
    collect_pids "launch_robot.py .*ip=${POLYMETIS_HOST} .*port=${POLYMETIS_PORT}"
  )
  while read -r pid; do append_unique_pid "${pid}"; done < <(
    collect_pids "franka_panda_client"
  )

  [[ "${#pids[@]}" -gt 0 ]] || return 0
  echo "Cleaning stale Polymetis arm server/client processes: ${pids[*]}"

  kill_pids INT "${pids[@]}"
  wait_pids_gone "${pids[@]}" && return 0

  echo "Stale Polymetis processes did not stop after SIGINT; sending SIGTERM..."
  kill_pids TERM "${pids[@]}"
  wait_pids_gone "${pids[@]}" && return 0

  echo "Stale Polymetis processes still alive; sending SIGKILL..."
  kill_pids KILL "${pids[@]}"
}

echo "Starting Polymetis arm server on ${POLYMETIS_HOST}:${POLYMETIS_PORT}"
echo "FR3 robot IP: ${FR3_ROBOT_IP}"
echo "Cartesian workspace lower: ${FRANKA_CARTESIAN_POS_LOWER}"
echo "Cartesian workspace upper: ${FRANKA_CARTESIAN_POS_UPPER}"
echo "Cartesian safety margin: ${FRANKA_SAFETY_CARTESIAN_MARGIN}"
echo "Use real time: ${POLYMETIS_USE_REAL_TIME}"
echo "Auto cleanup stale arm server: ${POLYMETIS_AUTO_CLEANUP}"

cleanup_stale_arm_server

exec conda run --no-capture-output -n polymetis launch_robot.py \
  robot_client=franka_hardware \
  robot_client.executable_cfg.robot_ip="${FR3_ROBOT_IP}" \
  ip="${POLYMETIS_HOST}" \
  port="${POLYMETIS_PORT}" \
  use_real_time="${POLYMETIS_USE_REAL_TIME}" \
  robot_client.use_real_time="${POLYMETIS_USE_REAL_TIME}" \
  robot_client.executable_cfg.use_real_time="${POLYMETIS_USE_REAL_TIME}" \
  "robot_client.executable_cfg.limits.cartesian_pos_lower=${FRANKA_CARTESIAN_POS_LOWER}" \
  "robot_client.executable_cfg.limits.cartesian_pos_upper=${FRANKA_CARTESIAN_POS_UPPER}" \
  "robot_client.executable_cfg.safety_controller.margins.cartesian_pos=${FRANKA_SAFETY_CARTESIAN_MARGIN}"
