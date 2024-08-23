#!/bin/bash

[[ $# < 1 ]] && echo "$0 <CKPT_DIR> [<CKPT_NAME>]" && exit 1

CKPT_DIR=$1
CKPT_NAME=${2:-policy_best.ckpt}
CHUNK_SIZE=${3:-100}
SKIP=${4:-1}

echo "[mt_act/iterate_rollout.sh] CKPT_DIR: ${CKPT_DIR}"
echo "[mt_act/iterate_rollout.sh] CKPT_NAME: ${CKPT_NAME}"
echo "[mt_act/iterate_rollout.sh] CHUNK_SIZE: ${CHUNK_SIZE}"  # ACT:100, MT-ACT:20
echo "[mt_act/iterate_rollout.sh] SKIP: ${SKIP}"

SCRIPT_DIR=$(cd $(dirname $0); pwd)
FIRST_OPTION="--wait_before_start"

array_pole=(0 1 2 3 4 5)
array_task=("task0_between-two" "task1_around-red" "task2_turn-blue" "task3_around-two")
for TASK_NAME in "${array_task[@]}"; do
    for POLE_POS_IDX in "${array_pole[@]}"; do
        echo "[mt_act/iterate_rollout.sh] TASK_NAME: ${TASK_NAME}"
        echo "[mt_act/iterate_rollout.sh] POLE_POS_IDX: ${POLE_POS_IDX}"
        python ${SCRIPT_DIR}/../bin/rollout.py \
        --ckpt_dir ${CKPT_DIR} --ckpt_name ${CKPT_NAME} --task_name ${TASK_NAME} \
        --skip ${SKIP} \
        --policy_class ACT --chunk_size ${CHUNK_SIZE} --num_epochs 0 \
        --kl_weight 10 \
        --hidden_dim 512 \
        --dim_feedforward 3200 \
        --seed 0 \
        --multi_task \
        --win_xy_policy 0 700 --win_xy_simulation 900 0 \
        --pole-pos-idx ${POLE_POS_IDX} $FIRST_OPTION
        FIRST_OPTION=""
    done
done

