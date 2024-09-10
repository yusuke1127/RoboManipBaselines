#!/bin/bash

[[ $# < 1 ]] && echo "$0 <CKPT_DIR> [<CKPT_NAME> <TASK_NAME> <SKIP>]" && exit 1

CKPT_DIR=$1
CKPT_NAME=${2:-SARNN.pth}
TASK_NAME=${3:-UR5eCable}
SKIP=${4:-6}

echo "[sarnn/iterate_rollout.sh] CKPT_DIR: ${CKPT_DIR}"
echo "[sarnn/iterate_rollout.sh] CKPT_NAME: ${CKPT_NAME}"
echo "[sarnn/iterate_rollout.sh] TASK_NAME: ${TASK_NAME}"
echo "[sarnn/iterate_rollout.sh] SKIP: ${SKIP}"

SCRIPT_DIR=$(cd $(dirname $0); pwd)
FIRST_OPTION="--wait_before_start"

WORLD_IDX_LIST=(0 1 2 3 4 5)
for WORLD_IDX in "${WORLD_IDX_LIST[@]}"; do
    echo "[sarnn/iterate_rollout.sh] WORLD_IDX: ${WORLD_IDX}"
    python ${SCRIPT_DIR}/../bin/RolloutSarnn${TASK_NAME}.py \
--checkpoint ${CKPT_DIR}/${CKPT_NAME} \
--skip ${SKIP} \
--world_idx ${WORLD_IDX} \
--win_xy_policy 0 700 ${FIRST_OPTION}
    FIRST_OPTION=""
done
