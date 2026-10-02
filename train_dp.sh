#!/bin/bash
set -e

GPU_ID="${GPU_ID:-0}"
TASK="${TASK:-real_wipe_image_dp_absolute_12fps}"
CONFIG_NAME="${CONFIG_NAME:-train_diffusion_unet_real_image_workspace}"
DATASET_PATH="${DATASET_PATH:-dataset/traj_rdp10d_downsample2}"
TASK_NAME="${TASK_NAME:-${TASK}}"
LOGGING_MODE="${LOGGING_MODE:-disabled}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-False}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
RELATIVE_ACTION="${RELATIVE_ACTION:-}"
RELATIVE_GRIPPER_ACTION="${RELATIVE_GRIPPER_ACTION:-}"
ZERO_GRIPPER_ACTION="${ZERO_GRIPPER_ACTION:-}"

OVERRIDES=(
    --config-name=${CONFIG_NAME}
    task=${TASK}
    task.dataset_path=${DATASET_PATH}
    task.name=${TASK_NAME}
    logging.mode=${LOGGING_MODE}
    dataloader.batch_size=${BATCH_SIZE}
    dataloader.num_workers=${NUM_WORKERS}
    dataloader.persistent_workers=${PERSISTENT_WORKERS}
    val_dataloader.batch_size=${BATCH_SIZE}
    val_dataloader.num_workers=${NUM_WORKERS}
    val_dataloader.persistent_workers=${PERSISTENT_WORKERS}
)

if [ -n "${NUM_EPOCHS:-}" ]; then
    OVERRIDES+=(training.num_epochs=${NUM_EPOCHS})
fi

if [ -n "${RELATIVE_ACTION}" ]; then
    OVERRIDES+=(task.dataset.relative_action=${RELATIVE_ACTION})
fi

if [ -n "${RELATIVE_GRIPPER_ACTION}" ]; then
    OVERRIDES+=(task.dataset.relative_gripper_action=${RELATIVE_GRIPPER_ACTION})
fi

if [ -n "${ZERO_GRIPPER_ACTION}" ]; then
    OVERRIDES+=(task.dataset.zero_gripper_action=${ZERO_GRIPPER_ACTION})
fi

if [ "${NUM_WORKERS}" != "0" ]; then
    OVERRIDES+=(+dataloader.prefetch_factor=${PREFETCH_FACTOR})
    OVERRIDES+=(+val_dataloader.prefetch_factor=${PREFETCH_FACTOR})
fi

CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch train.py \
    "${OVERRIDES[@]}"
