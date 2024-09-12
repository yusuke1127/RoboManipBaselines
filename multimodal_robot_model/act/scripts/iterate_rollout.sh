#!/bin/bash

[[ $# < 1 ]] && echo "$0 <CKPT_DIR> [<CKPT_NAME> <TASK_NAME> <SKIP>]" && exit 1

CKPT_DIR=$1
CKPT_NAME=${2:-policy_last.ckpt}
TASK_NAME=${3:-UR5eCable}
SKIP=${4:-3}

echo "[act/iterate_rollout.sh] CKPT_DIR: ${CKPT_DIR}"
echo "[act/iterate_rollout.sh] CKPT_NAME: ${CKPT_NAME}"
echo "[act/iterate_rollout.sh] TASK_NAME: ${TASK_NAME}"
echo "[act/iterate_rollout.sh] SKIP: ${SKIP}"

SCRIPT_DIR=$(cd $(dirname $0); pwd)
FIRST_OPTION="--wait_before_start"

WORLD_IDX_LIST=(0 1 2 3 4 5)
for WORLD_IDX in "${WORLD_IDX_LIST[@]}"; do
    echo "[act/iterate_rollout.sh] WORLD_IDX: ${WORLD_IDX}"
    python ${SCRIPT_DIR}/../bin/RolloutAct${TASK_NAME}.py \
           --ckpt_dir ${CKPT_DIR} --ckpt_name ${CKPT_NAME} \
           --chunk_size 100 --seed 42 \
           --skip ${SKIP} \
           --world_idx ${WORLD_IDX} \
           --win_xy_policy 0 700 ${FIRST_OPTION}
    FIRST_OPTION=""
done
