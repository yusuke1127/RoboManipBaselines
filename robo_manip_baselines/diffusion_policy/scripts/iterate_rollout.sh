#!/bin/bash

[[ $# < 1 ]] && echo "$0 <CKPT_PATH> [<TASK_NAME> <SKIP>]" && exit 1

CKPT_PATH=$1
TASK_NAME=${2:-MujocoUR5eCable}
SKIP=${3:-3}

echo "[diffusion_policy/iterate_rollout.sh] CKPT_PATH: ${CKPT_PATH}"
echo "[diffusion_policy/iterate_rollout.sh] TASK_NAME: ${TASK_NAME}"
echo "[diffusion_policy/iterate_rollout.sh] SKIP: ${SKIP}"

SCRIPT_DIR=$(cd $(dirname $0); pwd)
FIRST_OPTION="--wait_before_start"

WORLD_IDX_LIST=(0 1 2 3 4 5)
for WORLD_IDX in "${WORLD_IDX_LIST[@]}"; do
    echo "[diffusion_policy/iterate_rollout.sh] WORLD_IDX: ${WORLD_IDX}"
    python ${SCRIPT_DIR}/../bin/rollout/RolloutDiffusionPolicy${TASK_NAME}.py \
           --checkpoint ${CKPT_PATH} \
           --skip ${SKIP} \
           --world_idx ${WORLD_IDX} \
           --win_xy_policy 0 700 ${FIRST_OPTION}
    FIRST_OPTION=""
done
