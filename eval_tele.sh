#!/bin/bash
set -euo pipefail

CKPT_PATH="${CKPT_PATH:-data/outputs/2026.05.20/16.24.48_train_diffusion_unet_image_real_wipe_image_dp_absolute_12fps/checkpoints/latest.ckpt}"
CONFIG_NAME="${CONFIG_NAME:-train_diffusion_unet_real_image_workspace}"
TASK="${TASK:-franka_polymetis_image_dp_absolute_12fps}"
DATASET_PATH="${DATASET_PATH:-dataset/traj_rdp10d_downsample2}"
OUTPUT_DIR="${OUTPUT_DIR:-data/eval_outputs/franka_polymetis}"
MAX_DURATION="${MAX_DURATION:-20}"
EVAL_EPISODES="${EVAL_EPISODES:-1}"
ROBOT_SERVER_HOST="${ROBOT_SERVER_HOST:-127.0.0.1}"
ROBOT_SERVER_PORT="${ROBOT_SERVER_PORT:-8092}"
CAMERA_SOURCE="${CAMERA_SOURCE:-auto}"
CAMERA_BACKEND="${CAMERA_BACKEND:-opencv}"
CAMERA_PREPROCESS_MODE="${CAMERA_PREPROCESS_MODE:-}"
CAMERA_SQUARE_CROP_BOTTOM_ROWS="${CAMERA_SQUARE_CROP_BOTTOM_ROWS:-0}"
if [[ -z "${CAMERA_PREPROCESS_MODE}" && "${CAMERA_BACKEND}" == "iphone" ]]; then
  CAMERA_PREPROCESS_MODE="square_crop"
fi
CAMERA_COLOR_MATCH_DATASET_START="${CAMERA_COLOR_MATCH_DATASET_START:-False}"
CAMERA_COLOR_MATCH_DATASET_EPISODE="${CAMERA_COLOR_MATCH_DATASET_EPISODE:-}"
CAMERA_COLOR_MATCH_TIMEOUT="${CAMERA_COLOR_MATCH_TIMEOUT:-2.0}"
CAMERA_ZED_VIEW="${CAMERA_ZED_VIEW:-left}"
CAMERA_ZED_RESOLUTION="${CAMERA_ZED_RESOLUTION:-HD720}"
CAMERA_ZED_DEPTH_MODE="${CAMERA_ZED_DEPTH_MODE:-NEURAL}"
CAMERA_IPHONE_BIND_HOST="${CAMERA_IPHONE_BIND_HOST:-0.0.0.0}"
CAMERA_IPHONE_VIDEO_PORT="${CAMERA_IPHONE_VIDEO_PORT:-5560}"
CAMERA_IPHONE_COMBINED_PORT="${CAMERA_IPHONE_COMBINED_PORT:-5558}"
CAMERA_IPHONE_PHONE_IP="${CAMERA_IPHONE_PHONE_IP:-}"
CAMERA_IPHONE_REGISTRATION_PORT="${CAMERA_IPHONE_REGISTRATION_PORT:-5559}"
CAMERA_IPHONE_STARTUP_TIMEOUT="${CAMERA_IPHONE_STARTUP_TIMEOUT:-5.0}"
CAMERA_IPHONE_READ_TIMEOUT="${CAMERA_IPHONE_READ_TIMEOUT:-1.0}"
CAMERA_IPHONE_HELLO_INTERVAL="${CAMERA_IPHONE_HELLO_INTERVAL:-2.0}"
CAMERA_FLIP="${CAMERA_FLIP:-False}"
ASK_RESET_CONFIRMATION="${ASK_RESET_CONFIRMATION:-False}"
OPEN_GRIPPER_ON_START="${OPEN_GRIPPER_ON_START:-False}"
START_GRIPPER_WIDTH_MM="${START_GRIPPER_WIDTH_MM:-}"
MOVE_TO_START="${MOVE_TO_START:-True}"
MOVE_TO_START_POSE_PATH="${MOVE_TO_START_POSE_PATH:-}"
RESET_EPISODE="${RESET_EPISODE:-0}"
MOVE_TO_START_DURATION="${MOVE_TO_START_DURATION:-5.0}"
MOVE_TO_START_FREQUENCY="${MOVE_TO_START_FREQUENCY:-30.0}"
MOVE_TO_START_SETTLE="${MOVE_TO_START_SETTLE:-True}"
MOVE_TO_START_SETTLE_TIMEOUT="${MOVE_TO_START_SETTLE_TIMEOUT:-3.0}"
MOVE_TO_START_SETTLE_FREQUENCY="${MOVE_TO_START_SETTLE_FREQUENCY:-20.0}"
MOVE_TO_START_POS_TOLERANCE="${MOVE_TO_START_POS_TOLERANCE:-0.018}"
MOVE_TO_START_ROT_TOLERANCE_DEG="${MOVE_TO_START_ROT_TOLERANCE_DEG:-8.0}"
MOVE_TO_START_STRICT="${MOVE_TO_START_STRICT:-True}"
SAVE_PROCESSED_IMAGE="${SAVE_PROCESSED_IMAGE:-True}"
PROCESSED_IMAGE_OUTPUT_DIR="${PROCESSED_IMAGE_OUTPUT_DIR:-${OUTPUT_DIR}/debug_images}"
ENABLE_POLICY_RECORDING="${ENABLE_POLICY_RECORDING:-True}"
POLICY_RECORDING_OUTPUT_DIR="${POLICY_RECORDING_OUTPUT_DIR:-${OUTPUT_DIR}/policy_recordings}"
POLICY_RECORDING_FPS="${POLICY_RECORDING_FPS:-12}"
POLICY_RECORDING_IMAGE_WIDTH="${POLICY_RECORDING_IMAGE_WIDTH:-}"
POLICY_RECORDING_IMAGE_HEIGHT="${POLICY_RECORDING_IMAGE_HEIGHT:-}"
POLICY_RECORDING_PLOT_WIDTH="${POLICY_RECORDING_PLOT_WIDTH:-360}"
POLICY_RECORDING_PLOT_WINDOW_SEC="${POLICY_RECORDING_PLOT_WINDOW_SEC:-10.0}"
GRIPPER_ACTION_MODE="${GRIPPER_ACTION_MODE:-continuous}"
IGNORE_GRIPPER_COMMANDS="${IGNORE_GRIPPER_COMMANDS:-False}"
IGNORE_POLICY_TCP_COMMANDS="${IGNORE_POLICY_TCP_COMMANDS:-False}"
IGNORE_POLICY_GRIPPER_COMMANDS="${IGNORE_POLICY_GRIPPER_COMMANDS:-False}"
GRIPPER_STROKE="${GRIPPER_STROKE:-0.085}"
GRIPPER_COMMAND_WIDTH_OFFSET="${GRIPPER_COMMAND_WIDTH_OFFSET:-}"
FIXED_GRIPPER_WIDTH_MM="${FIXED_GRIPPER_WIDTH_MM:-}"
GRIPPER_SIGNAL_SOURCE="${GRIPPER_SIGNAL_SOURCE:-umi_marker}"
GRIPPER_RAW_MIN="${GRIPPER_RAW_MIN:-}"
GRIPPER_RAW_MAX="${GRIPPER_RAW_MAX:-}"
GRIPPER_RAW_OPEN_QUANTILE="${GRIPPER_RAW_OPEN_QUANTILE:-0.99}"
GRIPPER_RAW_RANGE_MODE="${GRIPPER_RAW_RANGE_MODE:-}"
GRIPPER_RAW_CALIBRATION_SCALE="${GRIPPER_RAW_CALIBRATION_SCALE:-}"
GRIPPER_RAW_CALIBRATION_OFFSET="${GRIPPER_RAW_CALIBRATION_OFFSET:-}"
GRIPPER_CONTROL_WIDTH_PRECISION="${GRIPPER_CONTROL_WIDTH_PRECISION:-0.001}"
GRIPPER_BINARY_THRESHOLD="${GRIPPER_BINARY_THRESHOLD:-0.5}"
GRIPPER_BINARY_HYSTERESIS="${GRIPPER_BINARY_HYSTERESIS:-0.0}"
GRIPPER_BINARY_OPEN_THRESHOLD="${GRIPPER_BINARY_OPEN_THRESHOLD:-0.98}"
GRIPPER_OBS_MODE="${GRIPPER_OBS_MODE:-}"
GRIPPER_OBS_RAW_OFFSET="${GRIPPER_OBS_RAW_OFFSET:-0.0}"
TCP_ACTION_UPDATE_INTERVAL="${TCP_ACTION_UPDATE_INTERVAL:-8}"
GRIPPER_ACTION_UPDATE_INTERVAL="${GRIPPER_ACTION_UPDATE_INTERVAL:-8}"
TCP_ENSEMBLE_MODE="${TCP_ENSEMBLE_MODE:-hato}"
GRIPPER_ENSEMBLE_MODE="${GRIPPER_ENSEMBLE_MODE:-hato}"
TCP_ENSEMBLE_TAU="${TCP_ENSEMBLE_TAU:-0.9}"
GRIPPER_ENSEMBLE_TAU="${GRIPPER_ENSEMBLE_TAU:-0.7}"
LATENT_TCP_ENSEMBLE_MODE="${LATENT_TCP_ENSEMBLE_MODE:-hato}"
LATENT_GRIPPER_ENSEMBLE_MODE="${LATENT_GRIPPER_ENSEMBLE_MODE:-hato}"
LATENT_TCP_ENSEMBLE_TAU="${LATENT_TCP_ENSEMBLE_TAU:-0.9}"
LATENT_GRIPPER_ENSEMBLE_TAU="${LATENT_GRIPPER_ENSEMBLE_TAU:-0.9}"
TCP_MOVE_TIMEOUT="${TCP_MOVE_TIMEOUT:-0.05}"
TCP_TARGET_DURATION="${TCP_TARGET_DURATION:-}"
LATENCY_STEP="${LATENCY_STEP:-}"
GRIPPER_LATENCY_STEP="${GRIPPER_LATENCY_STEP:-}"
RELATIVE_ACTION="${RELATIVE_ACTION:-}"
OBS_TEMPORAL_DOWNSAMPLE_RATIO="${OBS_TEMPORAL_DOWNSAMPLE_RATIO:-}"
AT="${AT:-}"
AT_LOAD_DIR="${AT_LOAD_DIR:-}"
CONTACT_ACTION_SCALE="${CONTACT_ACTION_SCALE:-1.0}"
CONTACT_ACTION_SCALE_MODE="${CONTACT_ACTION_SCALE_MODE:-fixed}"
CONTACT_ACTION_SCALE_THRESHOLD="${CONTACT_ACTION_SCALE_THRESHOLD:-400.0}"
CONTACT_ACTION_SCALE_OBS_KEY="${CONTACT_ACTION_SCALE_OBS_KEY:-left_gripper1_marker_offset_emb}"
CONTACT_ACTION_SCALE_DIMS="${CONTACT_ACTION_SCALE_DIMS:-12}"
CONTACT_ACTION_SCALE_MAX_TRANSLATION="${CONTACT_ACTION_SCALE_MAX_TRANSLATION:-}"
CONTACT_ACTION_SCALE_MAGNET_DIVISOR="${CONTACT_ACTION_SCALE_MAGNET_DIVISOR:-300.0}"
CONTACT_ACTION_SCALE_LOG_EVERY="${CONTACT_ACTION_SCALE_LOG_EVERY:-6}"
ENABLE_MAGNET="${ENABLE_MAGNET:-}"
MAGNET_PORT="${MAGNET_PORT:-}"
MAGNET_BAUDRATE="${MAGNET_BAUDRATE:-}"
MAGNET_REQUIRED="${MAGNET_REQUIRED:-}"
MAGNET_SENSOR_ORDER="${MAGNET_SENSOR_ORDER:-4,1,2,3}"
MAGNET_ZERO_CHANNELS="${MAGNET_ZERO_CHANNELS:-}"

MAGNET_READER_SUBTRACT_BASELINE="${MAGNET_READER_SUBTRACT_BASELINE:-True}"
MAGNET_NORMALIZE_TO_FIRST_FRAME="${MAGNET_NORMALIZE_TO_FIRST_FRAME:-False}"
MAGNET_REZERO_AFTER_POLICY_START_SEC="${MAGNET_REZERO_AFTER_POLICY_START_SEC:-}"
MAGNET_FILTER_ABNORMAL_READINGS="${MAGNET_FILTER_ABNORMAL_READINGS:-True}"
MAGNET_ABNORMAL_ABS_THRESHOLD="${MAGNET_ABNORMAL_ABS_THRESHOLD:-5000.0}"
DEBUG_POLICY_ACTIONS="${DEBUG_POLICY_ACTIONS:-False}"
DEBUG_POLICY_ACTION_EVERY="${DEBUG_POLICY_ACTION_EVERY:-1}"
POLICY_TCP_POSE_OBS_MODE="${POLICY_TCP_POSE_OBS_MODE:-none}"
POLICY_TCP_POSE_OBS_DATASET_EPISODE="${POLICY_TCP_POSE_OBS_DATASET_EPISODE:-}"
POLICY_TCP_POSE_OBS_DATASET_PATH="${POLICY_TCP_POSE_OBS_DATASET_PATH:-}"
POLICY_RELATIVE_ACTION_FRAME_MODE="${POLICY_RELATIVE_ACTION_FRAME_MODE:-none}"
export LOGURU_LEVEL="${LOGURU_LEVEL:-INFO}"

# Converted datasets can store physical width in meters even when the original
# GELLO recorder stored a direct 0..1 open-ratio command.
case "${GRIPPER_SIGNAL_SOURCE}" in
  task_config|config)
    GRIPPER_COMMAND_WIDTH_OFFSET="${GRIPPER_COMMAND_WIDTH_OFFSET:-0.000}"
    ;;
  umi_marker|umi)
    GRIPPER_RAW_RANGE_MODE="${GRIPPER_RAW_RANGE_MODE:-calibrated_marker_width}"
    # Width is already calibrated; scale=1 and offset=0 make command/obs direct.
    GRIPPER_RAW_CALIBRATION_SCALE="${GRIPPER_RAW_CALIBRATION_SCALE:-1.0}"
    GRIPPER_RAW_CALIBRATION_OFFSET="${GRIPPER_RAW_CALIBRATION_OFFSET:-0.0}"
    GRIPPER_COMMAND_WIDTH_OFFSET="${GRIPPER_COMMAND_WIDTH_OFFSET:--0.010}"
    GRIPPER_OBS_MODE="${GRIPPER_OBS_MODE:-auto}"
    ;;
  teleop_raw|teleop)
    # Raw teleop values are open ratios: 0 is closed and 1 is fully open.
    # Do not reinterpret the observed training minimum as the closed endpoint.
    GRIPPER_RAW_RANGE_MODE="${GRIPPER_RAW_RANGE_MODE:-trajectory_zero_to_max}"
    GRIPPER_RAW_MIN="${GRIPPER_RAW_MIN:-0.0}"
    GRIPPER_RAW_MAX="${GRIPPER_RAW_MAX:-1.0}"
    GRIPPER_COMMAND_WIDTH_OFFSET="${GRIPPER_COMMAND_WIDTH_OFFSET:-0.000}"
    # Teleop training records the commanded raw signal, not measured width.
    GRIPPER_OBS_MODE="${GRIPPER_OBS_MODE:-commanded}"
    ;;
  teleop_width|teleop_meters|physical_width)
    # GELLO command signal after conversion from open ratio to physical meters.
    GRIPPER_RAW_RANGE_MODE="${GRIPPER_RAW_RANGE_MODE:-calibrated_marker_width}"
    GRIPPER_RAW_CALIBRATION_SCALE="${GRIPPER_RAW_CALIBRATION_SCALE:-1.0}"
    GRIPPER_RAW_CALIBRATION_OFFSET="${GRIPPER_RAW_CALIBRATION_OFFSET:-0.0}"
    GRIPPER_COMMAND_WIDTH_OFFSET="${GRIPPER_COMMAND_WIDTH_OFFSET:-0.000}"
    GRIPPER_OBS_MODE="${GRIPPER_OBS_MODE:-commanded}"
    ;;
  *)
    echo "ERROR: GRIPPER_SIGNAL_SOURCE must be task_config, umi_marker, teleop_raw, or teleop_width; got '${GRIPPER_SIGNAL_SOURCE}'" >&2
    exit 2
    ;;
esac
GRIPPER_OBS_MODE="${GRIPPER_OBS_MODE:-auto}"

echo "Gripper signal source: ${GRIPPER_SIGNAL_SOURCE} (range_mode=${GRIPPER_RAW_RANGE_MODE:-task_config}, obs_mode=${GRIPPER_OBS_MODE})"

if [[ -n "${START_GRIPPER_WIDTH_MM}" && -n "${FIXED_GRIPPER_WIDTH_MM}" ]]; then
  echo "ERROR: START_GRIPPER_WIDTH_MM and FIXED_GRIPPER_WIDTH_MM cannot be used together." >&2
  exit 2
fi

if [[ "${CONFIG_NAME}" == *latent* || "${TASK}" == *ldp* ]]; then
  if [[ -z "${AT_LOAD_DIR}" ]]; then
    AT_LOAD_DIR="$(python -c 'import sys, torch, dill
path = sys.argv[1]
try:
    payload = torch.load(open(path, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload.get("cfg", {})
    value = cfg.get("at_load_dir", "") if hasattr(cfg, "get") else ""
    print(value or "")
except Exception:
    print("")' "${CKPT_PATH}")"
    if [[ -n "${AT_LOAD_DIR}" ]]; then
      echo "Using AT_LOAD_DIR from checkpoint cfg: ${AT_LOAD_DIR}"
    fi
  fi
  if [[ -z "${AT_LOAD_DIR}" || ! -f "${AT_LOAD_DIR}" ]]; then
    echo "ERROR: latent/RDP eval requires a trained AT checkpoint."
    echo "Set AT_LOAD_DIR=/path/to/train_vae.../checkpoints/latest.ckpt"
    echo "Current AT_LOAD_DIR='${AT_LOAD_DIR}'"
    exit 1
  fi
fi

EXTRA_OVERRIDES=()
if [[ -n "${TCP_ENSEMBLE_MODE}" ]]; then
  EXTRA_OVERRIDES+=("task.env_runner.tcp_ensemble_buffer_params.ensemble_mode=${TCP_ENSEMBLE_MODE}")
fi
if [[ -n "${GRIPPER_ENSEMBLE_MODE}" ]]; then
  EXTRA_OVERRIDES+=("task.env_runner.gripper_ensemble_buffer_params.ensemble_mode=${GRIPPER_ENSEMBLE_MODE}")
fi
if [[ -n "${TCP_ENSEMBLE_TAU}" ]]; then
  EXTRA_OVERRIDES+=("task.env_runner.tcp_ensemble_buffer_params.tau=${TCP_ENSEMBLE_TAU}")
fi
if [[ -n "${GRIPPER_ENSEMBLE_TAU}" ]]; then
  EXTRA_OVERRIDES+=("task.env_runner.gripper_ensemble_buffer_params.tau=${GRIPPER_ENSEMBLE_TAU}")
fi
if [[ -n "${AT}" ]]; then
  EXTRA_OVERRIDES+=("at=${AT}")
fi
if [[ -n "${AT_LOAD_DIR}" ]]; then
  EXTRA_OVERRIDES+=("at_load_dir=${AT_LOAD_DIR}")
fi
if [[ -n "${TCP_TARGET_DURATION}" ]]; then
  EXTRA_OVERRIDES+=("task.env_runner.env_params.tcp_target_duration=${TCP_TARGET_DURATION}")
fi
if [[ -n "${LATENCY_STEP}" ]]; then
  EXTRA_OVERRIDES+=("task.env_runner.latency_step=${LATENCY_STEP}")
fi
if [[ -n "${GRIPPER_LATENCY_STEP}" ]]; then
  EXTRA_OVERRIDES+=("task.env_runner.gripper_latency_step=${GRIPPER_LATENCY_STEP}")
fi
if [[ -n "${RELATIVE_ACTION}" ]]; then
  EXTRA_OVERRIDES+=("task.dataset.relative_action=${RELATIVE_ACTION}")
  EXTRA_OVERRIDES+=("task.env_runner.use_relative_action=${RELATIVE_ACTION}")
fi
if [[ -n "${OBS_TEMPORAL_DOWNSAMPLE_RATIO}" ]]; then
  EXTRA_OVERRIDES+=("task.env_runner.obs_temporal_downsample_ratio=${OBS_TEMPORAL_DOWNSAMPLE_RATIO}")
fi
if [[ "${POLICY_TCP_POSE_OBS_MODE}" != "none" ]]; then
  EXTRA_OVERRIDES+=("++task.env_runner.policy_tcp_pose_obs_mode=${POLICY_TCP_POSE_OBS_MODE}")
fi
if [[ "${POLICY_TCP_POSE_OBS_MODE}" != "none" || "${POLICY_RELATIVE_ACTION_FRAME_MODE}" != "none" ]]; then
  if [[ -n "${POLICY_TCP_POSE_OBS_DATASET_EPISODE}" ]]; then
    EXTRA_OVERRIDES+=("++task.env_runner.policy_tcp_pose_obs_dataset_episode=${POLICY_TCP_POSE_OBS_DATASET_EPISODE}")
  fi
  if [[ -n "${POLICY_TCP_POSE_OBS_DATASET_PATH}" ]]; then
    EXTRA_OVERRIDES+=("++task.env_runner.policy_tcp_pose_obs_dataset_path=${POLICY_TCP_POSE_OBS_DATASET_PATH}")
  fi
fi
if [[ "${POLICY_RELATIVE_ACTION_FRAME_MODE}" != "none" ]]; then
  EXTRA_OVERRIDES+=("++task.env_runner.policy_relative_action_frame_mode=${POLICY_RELATIVE_ACTION_FRAME_MODE}")
fi
if [[ "${CONTACT_ACTION_SCALE}" != "1.0" && "${CONTACT_ACTION_SCALE}" != "1" || "${CONTACT_ACTION_SCALE_MODE}" != "fixed" ]]; then
  EXTRA_OVERRIDES+=("++task.env_runner.contact_action_scale=${CONTACT_ACTION_SCALE}")
  EXTRA_OVERRIDES+=("++task.env_runner.contact_action_scale_mode=${CONTACT_ACTION_SCALE_MODE}")
  EXTRA_OVERRIDES+=("++task.env_runner.contact_action_scale_threshold=${CONTACT_ACTION_SCALE_THRESHOLD}")
  EXTRA_OVERRIDES+=("++task.env_runner.contact_action_scale_obs_key=${CONTACT_ACTION_SCALE_OBS_KEY}")
  EXTRA_OVERRIDES+=("++task.env_runner.contact_action_scale_dims=${CONTACT_ACTION_SCALE_DIMS}")
  EXTRA_OVERRIDES+=("++task.env_runner.contact_action_scale_magnet_divisor=${CONTACT_ACTION_SCALE_MAGNET_DIVISOR}")
  EXTRA_OVERRIDES+=("++task.env_runner.contact_action_scale_log_every=${CONTACT_ACTION_SCALE_LOG_EVERY}")
  if [[ -n "${CONTACT_ACTION_SCALE_MAX_TRANSLATION}" ]]; then
    EXTRA_OVERRIDES+=("++task.env_runner.contact_action_scale_max_translation=${CONTACT_ACTION_SCALE_MAX_TRANSLATION}")
  fi
fi
if [[ -n "${POLICY_RECORDING_IMAGE_WIDTH}" ]]; then
  EXTRA_OVERRIDES+=("task.env_runner.env_params.policy_recording_image_width=${POLICY_RECORDING_IMAGE_WIDTH}")
fi
if [[ -n "${POLICY_RECORDING_IMAGE_HEIGHT}" ]]; then
  EXTRA_OVERRIDES+=("task.env_runner.env_params.policy_recording_image_height=${POLICY_RECORDING_IMAGE_HEIGHT}")
fi
if [[ -n "${ENABLE_MAGNET}" ]]; then
  EXTRA_OVERRIDES+=("task.env_runner.env_params.enable_magnet=${ENABLE_MAGNET}")
fi
if [[ -n "${MAGNET_PORT}" ]]; then
  EXTRA_OVERRIDES+=("task.env_runner.env_params.magnet_port=${MAGNET_PORT}")
fi
if [[ -n "${MAGNET_BAUDRATE}" ]]; then
  EXTRA_OVERRIDES+=("task.env_runner.env_params.magnet_baudrate=${MAGNET_BAUDRATE}")
fi
if [[ -n "${MAGNET_REQUIRED}" ]]; then
  EXTRA_OVERRIDES+=("task.env_runner.env_params.magnet_required=${MAGNET_REQUIRED}")
fi
if [[ -n "${MAGNET_SENSOR_ORDER}" ]]; then
  EXTRA_OVERRIDES+=("++task.env_runner.env_params.magnet_sensor_order=[${MAGNET_SENSOR_ORDER}]")
fi
if [[ -n "${MAGNET_ZERO_CHANNELS}" ]]; then
  EXTRA_OVERRIDES+=("++task.env_runner.env_params.magnet_zero_channels=[${MAGNET_ZERO_CHANNELS}]")
fi
if [[ -n "${MAGNET_REZERO_AFTER_POLICY_START_SEC}" ]]; then
  EXTRA_OVERRIDES+=("++task.env_runner.env_params.magnet_rezero_after_policy_start_sec=${MAGNET_REZERO_AFTER_POLICY_START_SEC}")
fi
if [[ -n "${GRIPPER_RAW_MIN}" ]]; then
  EXTRA_OVERRIDES+=("task.env_runner.env_params.gripper_raw_min=${GRIPPER_RAW_MIN}")
fi
if [[ -n "${GRIPPER_RAW_MAX}" ]]; then
  EXTRA_OVERRIDES+=("task.env_runner.env_params.gripper_raw_max=${GRIPPER_RAW_MAX}")
fi
if [[ -n "${GRIPPER_RAW_RANGE_MODE}" ]]; then
  EXTRA_OVERRIDES+=("++task.env_runner.env_params.gripper_raw_range_mode=${GRIPPER_RAW_RANGE_MODE}")
fi
if [[ -n "${GRIPPER_RAW_CALIBRATION_SCALE}" ]]; then
  EXTRA_OVERRIDES+=("++task.env_runner.env_params.gripper_raw_calibration_scale=${GRIPPER_RAW_CALIBRATION_SCALE}")
fi
if [[ -n "${GRIPPER_RAW_CALIBRATION_OFFSET}" ]]; then
  EXTRA_OVERRIDES+=("++task.env_runner.env_params.gripper_raw_calibration_offset=${GRIPPER_RAW_CALIBRATION_OFFSET}")
fi
if [[ -n "${FIXED_GRIPPER_WIDTH_MM}" ]]; then
  EXTRA_OVERRIDES+=("++task.env_runner.env_params.fixed_gripper_width_mm=${FIXED_GRIPPER_WIDTH_MM}")
fi
if [[ -n "${START_GRIPPER_WIDTH_MM}" ]]; then
  EXTRA_OVERRIDES+=("++task.env_runner.start_gripper_width_mm=${START_GRIPPER_WIDTH_MM}")
fi
if [[ -n "${CAMERA_PREPROCESS_MODE}" ]]; then
  EXTRA_OVERRIDES+=("++task.env_runner.env_params.camera_preprocess_mode=${CAMERA_PREPROCESS_MODE}")
fi
EXTRA_OVERRIDES+=("++task.env_runner.env_params.camera_square_crop_bottom_rows=${CAMERA_SQUARE_CROP_BOTTOM_ROWS}")
EXTRA_OVERRIDES+=("++task.env_runner.env_params.camera_color_match_dataset_start=${CAMERA_COLOR_MATCH_DATASET_START}")
EXTRA_OVERRIDES+=("++task.env_runner.env_params.camera_color_match_timeout=${CAMERA_COLOR_MATCH_TIMEOUT}")
if [[ -n "${CAMERA_COLOR_MATCH_DATASET_EPISODE}" ]]; then
  EXTRA_OVERRIDES+=("++task.env_runner.env_params.camera_color_match_dataset_episode=${CAMERA_COLOR_MATCH_DATASET_EPISODE}")
fi

python eval_real_robot_flexiv.py \
  --config-name "${CONFIG_NAME}" \
  "task=${TASK}" \
  "task.dataset_path=${DATASET_PATH}" \
  "+ckpt_path=${CKPT_PATH}" \
  "task.env_runner.output_dir=${OUTPUT_DIR}" \
  "task.env_runner.eval_episodes=${EVAL_EPISODES}" \
  "task.env_runner.max_duration_time=${MAX_DURATION}" \
  "task.env_runner.debug_policy_actions=${DEBUG_POLICY_ACTIONS}" \
  "task.env_runner.debug_policy_action_every=${DEBUG_POLICY_ACTION_EVERY}" \
  "++task.env_runner.open_gripper_on_start=${OPEN_GRIPPER_ON_START}" \
  "task.env_runner.tcp_action_update_interval=${TCP_ACTION_UPDATE_INTERVAL}" \
  "task.env_runner.gripper_action_update_interval=${GRIPPER_ACTION_UPDATE_INTERVAL}" \
  "++task.env_runner.latent_tcp_ensemble_buffer_params.ensemble_mode=${LATENT_TCP_ENSEMBLE_MODE}" \
  "++task.env_runner.latent_gripper_ensemble_buffer_params.ensemble_mode=${LATENT_GRIPPER_ENSEMBLE_MODE}" \
  "++task.env_runner.latent_tcp_ensemble_buffer_params.tau=${LATENT_TCP_ENSEMBLE_TAU}" \
  "++task.env_runner.latent_gripper_ensemble_buffer_params.tau=${LATENT_GRIPPER_ENSEMBLE_TAU}" \
  "task.env_runner.ask_reset_confirmation=${ASK_RESET_CONFIRMATION}" \
  "task.env_runner.env_params.robot_server_ip=${ROBOT_SERVER_HOST}" \
  "task.env_runner.env_params.robot_server_port=${ROBOT_SERVER_PORT}" \
  "task.env_runner.env_params.dataset_path=${DATASET_PATH}" \
  "task.env_runner.env_params.camera_source=${CAMERA_SOURCE}" \
  "task.env_runner.env_params.camera_backend=${CAMERA_BACKEND}" \
  "task.env_runner.env_params.camera_zed_view=${CAMERA_ZED_VIEW}" \
  "++task.env_runner.env_params.camera_zed_resolution=${CAMERA_ZED_RESOLUTION}" \
  "++task.env_runner.env_params.camera_zed_depth_mode=${CAMERA_ZED_DEPTH_MODE}" \
  "++task.env_runner.env_params.camera_iphone_bind_host=${CAMERA_IPHONE_BIND_HOST}" \
  "++task.env_runner.env_params.camera_iphone_video_port=${CAMERA_IPHONE_VIDEO_PORT}" \
  "++task.env_runner.env_params.camera_iphone_combined_port=${CAMERA_IPHONE_COMBINED_PORT}" \
  "++task.env_runner.env_params.camera_iphone_phone_ip=${CAMERA_IPHONE_PHONE_IP}" \
  "++task.env_runner.env_params.camera_iphone_registration_port=${CAMERA_IPHONE_REGISTRATION_PORT}" \
  "++task.env_runner.env_params.camera_iphone_startup_timeout=${CAMERA_IPHONE_STARTUP_TIMEOUT}" \
  "++task.env_runner.env_params.camera_iphone_read_timeout=${CAMERA_IPHONE_READ_TIMEOUT}" \
  "++task.env_runner.env_params.camera_iphone_hello_interval=${CAMERA_IPHONE_HELLO_INTERVAL}" \
  "task.env_runner.env_params.camera_flip=${CAMERA_FLIP}" \
  "task.env_runner.env_params.move_to_start_on_reset=${MOVE_TO_START}" \
  "++task.env_runner.env_params.move_to_start_pose_path=${MOVE_TO_START_POSE_PATH}" \
  "task.env_runner.env_params.reset_episode=${RESET_EPISODE}" \
  "task.env_runner.env_params.move_to_start_duration=${MOVE_TO_START_DURATION}" \
  "task.env_runner.env_params.move_to_start_frequency=${MOVE_TO_START_FREQUENCY}" \
  "task.env_runner.env_params.move_to_start_settle=${MOVE_TO_START_SETTLE}" \
  "task.env_runner.env_params.move_to_start_settle_timeout=${MOVE_TO_START_SETTLE_TIMEOUT}" \
  "task.env_runner.env_params.move_to_start_settle_frequency=${MOVE_TO_START_SETTLE_FREQUENCY}" \
  "task.env_runner.env_params.move_to_start_pos_tolerance=${MOVE_TO_START_POS_TOLERANCE}" \
  "task.env_runner.env_params.move_to_start_rot_tolerance_deg=${MOVE_TO_START_ROT_TOLERANCE_DEG}" \
  "task.env_runner.env_params.move_to_start_strict=${MOVE_TO_START_STRICT}" \
  "task.env_runner.env_params.save_processed_image=${SAVE_PROCESSED_IMAGE}" \
  "task.env_runner.env_params.processed_image_output_dir=${PROCESSED_IMAGE_OUTPUT_DIR}" \
  "task.env_runner.env_params.enable_policy_recording=${ENABLE_POLICY_RECORDING}" \
  "task.env_runner.env_params.policy_recording_output_dir=${POLICY_RECORDING_OUTPUT_DIR}" \
  "task.env_runner.env_params.policy_recording_fps=${POLICY_RECORDING_FPS}" \
  "task.env_runner.env_params.policy_recording_plot_width=${POLICY_RECORDING_PLOT_WIDTH}" \
  "task.env_runner.env_params.policy_recording_plot_window_sec=${POLICY_RECORDING_PLOT_WINDOW_SEC}" \
  "task.env_runner.env_params.gripper_action_mode=${GRIPPER_ACTION_MODE}" \
  "++task.env_runner.env_params.ignore_gripper_commands=${IGNORE_GRIPPER_COMMANDS}" \
  "++task.env_runner.env_params.ignore_policy_tcp_commands=${IGNORE_POLICY_TCP_COMMANDS}" \
  "++task.env_runner.env_params.ignore_policy_gripper_commands=${IGNORE_POLICY_GRIPPER_COMMANDS}" \
  "task.env_runner.env_params.gripper_stroke=${GRIPPER_STROKE}" \
  "++task.env_runner.env_params.gripper_command_width_offset=${GRIPPER_COMMAND_WIDTH_OFFSET}" \
  "task.env_runner.env_params.gripper_raw_open_quantile=${GRIPPER_RAW_OPEN_QUANTILE}" \
  "task.env_runner.env_params.gripper_control_width_precision=${GRIPPER_CONTROL_WIDTH_PRECISION}" \
  "task.env_runner.env_params.gripper_binary_threshold=${GRIPPER_BINARY_THRESHOLD}" \
  "task.env_runner.env_params.gripper_binary_hysteresis=${GRIPPER_BINARY_HYSTERESIS}" \
  "task.env_runner.env_params.gripper_binary_open_threshold=${GRIPPER_BINARY_OPEN_THRESHOLD}" \
  "task.env_runner.env_params.gripper_obs_mode=${GRIPPER_OBS_MODE}" \
  "++task.env_runner.env_params.gripper_obs_raw_offset=${GRIPPER_OBS_RAW_OFFSET}" \
  "task.env_runner.env_params.tcp_move_timeout=${TCP_MOVE_TIMEOUT}" \
  "++task.env_runner.env_params.magnet_reader_subtract_baseline=${MAGNET_READER_SUBTRACT_BASELINE}" \
  "task.env_runner.env_params.magnet_normalize_to_first_frame=${MAGNET_NORMALIZE_TO_FIRST_FRAME}" \
  "++task.env_runner.env_params.magnet_filter_abnormal_readings=${MAGNET_FILTER_ABNORMAL_READINGS}" \
  "++task.env_runner.env_params.magnet_abnormal_abs_threshold=${MAGNET_ABNORMAL_ABS_THRESHOLD}" \
  "${EXTRA_OVERRIDES[@]}"
