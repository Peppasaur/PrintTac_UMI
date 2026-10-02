#!/bin/bash
set -euo pipefail

# Train with each episode's first RGB frame and temporal tactile/state features.
CONFIG_NAME="${CONFIG_NAME:-train_diffusion_unet_first_frame_workspace}"
TASK="${TASK:-real_wipe_image_gelsight_emb_dp_absolute_12fps}"
export CONFIG_NAME
export TASK
exec bash "$(dirname "$0")/train_dp.sh" "$@"
