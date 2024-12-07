#!/bin/bash

[[ $# < 1 ]] && echo "$0 <CHECKPOINT> <TASK_NAME> [<CROPPED_IMG_SIZE> <SKIP>]" && exit 1

CHECKPOINT=$1
TASK_NAME=$2
CROPPED_IMG_SIZE=${3:-280}
SKIP=${4:-6}

echo "[sarnn/iterate_rollout.sh] CHECKPOINT: ${CHECKPOINT}"
echo "[sarnn/iterate_rollout.sh] TASK_NAME: ${TASK_NAME}"
echo "[sarnn/iterate_rollout.sh] CROPPED_IMG_SIZE: ${CROPPED_IMG_SIZE}"
echo "[sarnn/iterate_rollout.sh] SKIP: ${SKIP}"

SCRIPT_DIR=$(cd $(dirname $0); pwd)
FIRST_OPTION="--wait_before_start"

WORLD_IDX_LIST=(0 1 2 3 4 5)
for WORLD_IDX in "${WORLD_IDX_LIST[@]}"; do
    echo "[sarnn/iterate_rollout.sh] WORLD_IDX: ${WORLD_IDX}"
    python ${SCRIPT_DIR}/../bin/rollout/RolloutSarnn${TASK_NAME}.py \
           --checkpoint ${CHECKPOINT} \
           --cropped_img_size ${CROPPED_IMG_SIZE} \
           --skip ${SKIP} \
           --world_idx ${WORLD_IDX} \
           --win_xy_policy 0 700 ${FIRST_OPTION}
    FIRST_OPTION=""
done
