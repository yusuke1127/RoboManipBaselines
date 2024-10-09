#!/bin/bash

[[ $# < 1 ]] && echo "$0 <CKPT_DIR> [<CKPT_NAME> <ENV_TASK_NAME> <SKIP>]" && exit 1

CKPT_DIR=$1
CKPT_NAME=${2:-policy_last.ckpt}
ENV_TASK_NAME=${3:-MujocoUR5eCable}
SKIP=${4:-3}

echo "[mt_act/iterate_rollout.sh] CKPT_DIR: ${CKPT_DIR}"
echo "[mt_act/iterate_rollout.sh] CKPT_NAME: ${CKPT_NAME}"
echo "[mt_act/iterate_rollout.sh] ENV_TASK_NAME: ${ENV_TASK_NAME}"
echo "[mt_act/iterate_rollout.sh] SKIP: ${SKIP}"

SCRIPT_DIR=$(cd $(dirname $0); pwd)
FIRST_OPTION="--wait_before_start"

POLICY_TASK_NAME_LIST=("task0_between-two" "task1_around-red" "task2_turn-blue" "task3_around-two")
WORLD_IDX_LIST=(0 1 2 3 4 5)
for POLICY_TASK_NAME in "${POLICY_TASK_NAME_LIST[@]}"; do
    for WORLD_IDX in "${WORLD_IDX_LIST[@]}"; do
        echo "[mt_act/iterate_rollout.sh] POLICY_TASK_NAME: ${POLICY_TASK_NAME}"
        echo "[mt_act/iterate_rollout.sh] WORLD_IDX: ${WORLD_IDX}"
        python ${SCRIPT_DIR}/../bin/rollout/RolloutAct${ENV_TASK_NAME}.py \
               --ckpt_dir ${CKPT_DIR} --ckpt_name ${CKPT_NAME} --task_name ${POLICY_TASK_NAME} \
               --chunk_size 100 --seed 42 \
               --skip ${SKIP} \
               --world_idx ${WORLD_IDX} \
               --win_xy_policy 0 700 ${FIRST_OPTION}
        FIRST_OPTION=""
    done
done
