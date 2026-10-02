#!/bin/bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
AT="${AT:-at_wipe_lift_12fps}"
AT_TASK="${AT_TASK:-real_wipe_image_gelsight_emb_at_absolute_12fps}"
LDP_TASK="${LDP_TASK:-real_wipe_image_gelsight_emb_ldp_absolute_12fps}"
DATASET_PATH="${DATASET_PATH:-dataset/traj_rdp10d_command_downsample2}"
LOGGING_MODE="${LOGGING_MODE:-disabled}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-False}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
RELATIVE_ACTION="${RELATIVE_ACTION:-True}"
RELATIVE_GRIPPER_ACTION="${RELATIVE_GRIPPER_ACTION:-False}"
RUN_TAG="${RUN_TAG:-$(date +%m%d%H%M%S)}"
SEARCH_PATH="${SEARCH_PATH:-../data/outputs}"
SKIP_AT="${SKIP_AT:-False}"
AT_LOAD_DIR="${AT_LOAD_DIR:-}"
AT_NUM_EPOCHS="${AT_NUM_EPOCHS:-${NUM_EPOCHS:-}}"
LDP_NUM_EPOCHS="${LDP_NUM_EPOCHS:-${NUM_EPOCHS:-}}"
AT_TASK_NAME="${AT_TASK_NAME:-${AT_TASK}_${RUN_TAG}}"
LDP_TASK_NAME="${LDP_TASK_NAME:-${LDP_TASK}_${RUN_TAG}}"

AT_OVERRIDES=(
    --config-name=train_at_workspace
    task=${AT_TASK}
    task.dataset_path=${DATASET_PATH}
    task.name=${AT_TASK_NAME}
    at=${AT}
    logging.mode=${LOGGING_MODE}
    dataloader.batch_size=${BATCH_SIZE}
    dataloader.num_workers=${NUM_WORKERS}
    dataloader.persistent_workers=${PERSISTENT_WORKERS}
    val_dataloader.batch_size=${BATCH_SIZE}
    val_dataloader.num_workers=${NUM_WORKERS}
    val_dataloader.persistent_workers=${PERSISTENT_WORKERS}
    task.dataset.relative_action=${RELATIVE_ACTION}
    task.dataset.relative_gripper_action=${RELATIVE_GRIPPER_ACTION}
)

if [ -n "${AT_NUM_EPOCHS}" ]; then
    AT_OVERRIDES+=(training.num_epochs=${AT_NUM_EPOCHS})
fi

if [ "${NUM_WORKERS}" != "0" ]; then
    AT_OVERRIDES+=(+dataloader.prefetch_factor=${PREFETCH_FACTOR})
    AT_OVERRIDES+=(+val_dataloader.prefetch_factor=${PREFETCH_FACTOR})
fi

if [ -z "${AT_LOAD_DIR}" ] && [ "${SKIP_AT}" != "True" ] && [ "${SKIP_AT}" != "true" ]; then
    echo "Stage 1: training Asymmetric Tokenizer..."
    CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py "${AT_OVERRIDES[@]}"

    echo ""
    echo "Searching for the AT checkpoint..."
    AT_LOAD_DIR=$(find "${SEARCH_PATH}" -path "*${AT_TASK_NAME}*/checkpoints/latest.ckpt" -type f | sort | tail -n 1)
fi

if [ ! -f "${AT_LOAD_DIR}" ]; then
    echo "Error: AT checkpoint not found. Set AT_LOAD_DIR=/path/to/latest.ckpt or let Stage 1 finish successfully."
    echo "Current AT_LOAD_DIR='${AT_LOAD_DIR}'"
    exit 1
fi

LDP_OVERRIDES=(
    --config-name=train_latent_diffusion_unet_real_image_workspace
    task=${LDP_TASK}
    task.dataset_path=${DATASET_PATH}
    task.name=${LDP_TASK_NAME}
    at=${AT}
    at_load_dir=${AT_LOAD_DIR}
    logging.mode=${LOGGING_MODE}
    dataloader.batch_size=${BATCH_SIZE}
    dataloader.num_workers=${NUM_WORKERS}
    dataloader.persistent_workers=${PERSISTENT_WORKERS}
    val_dataloader.batch_size=${BATCH_SIZE}
    val_dataloader.num_workers=${NUM_WORKERS}
    val_dataloader.persistent_workers=${PERSISTENT_WORKERS}
    task.dataset.relative_action=${RELATIVE_ACTION}
    task.dataset.relative_gripper_action=${RELATIVE_GRIPPER_ACTION}
)

if [ -n "${LDP_NUM_EPOCHS}" ]; then
    LDP_OVERRIDES+=(training.num_epochs=${LDP_NUM_EPOCHS})
fi

if [ "${NUM_WORKERS}" != "0" ]; then
    LDP_OVERRIDES+=(+dataloader.prefetch_factor=${PREFETCH_FACTOR})
    LDP_OVERRIDES+=(+val_dataloader.prefetch_factor=${PREFETCH_FACTOR})
fi

echo ""
echo "Stage 2: training Latent Diffusion Policy..."
echo "Using AT checkpoint: ${AT_LOAD_DIR}"
CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch train.py "${LDP_OVERRIDES[@]}"
