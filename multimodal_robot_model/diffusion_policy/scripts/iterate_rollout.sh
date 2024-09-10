#!/bin/bash

[[ $# < 1 ]] && echo "$0 <CKPT_PATH> [<SKIP>]" && exit 1

CKPT_PATH=$1
SKIP=${2:-3}

echo "[diffusion_policy/iterate_rollout.sh] CKPT_PATH: ${CKPT_PATH}"
echo "[diffusion_policy/iterate_rollout.sh] SKIP: ${SKIP}"

SCRIPT_DIR=$(cd $(dirname $0); pwd)
FIRST_OPTION="--wait_before_start"

WORLD_IDX_LIST=(0 1 2 3 4 5)
for WORLD_IDX in "${WORLD_IDX_LIST[@]}"; do
    echo "[diffusion_policy/iterate_rollout.sh] world_idx: ${WORLD_IDX}"
    python ${SCRIPT_DIR}/../bin/RolloutDiffusionPolicyUR5eCable.py \
--checkpoint ${CKPT_PATH} \
--skip ${SKIP} \
--world_idx ${WORLD_IDX} \
--win_xy_policy 0 700 ${FIRST_OPTION}
    FIRST_OPTION=""
done
