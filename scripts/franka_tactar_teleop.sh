#!/usr/bin/env bash
set -euo pipefail

RDP_ROBOT_SERVER_HOST="${RDP_ROBOT_SERVER_HOST:-127.0.0.1}"
RDP_ROBOT_SERVER_PORT="${RDP_ROBOT_SERVER_PORT:-8092}"
RDP_TELEOP_HOST="${RDP_TELEOP_HOST:-10.16.1.240}"
RDP_TELEOP_PORT="${RDP_TELEOP_PORT:-8082}"

exec conda run --no-capture-output -n umi python franka_tactar_teleop.py \
  task=real_franka_env \
  task.robot_server.host_ip="${RDP_ROBOT_SERVER_HOST}" \
  task.robot_server.port="${RDP_ROBOT_SERVER_PORT}" \
  task.teleop_server.host_ip="${RDP_TELEOP_HOST}" \
  task.teleop_server.port="${RDP_TELEOP_PORT}" \
  "$@"
