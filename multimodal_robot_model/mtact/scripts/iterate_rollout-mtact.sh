#!/bin/bash

[[ $# < 1 ]] && echo "$0 <CKPT_DIR> [<CKPT_NAME>]" && exit 1

CKPT_DIR=$1
CKPT_NAME=${2:-policy_best.ckpt}
SKIP=${3:-2}
POLE_POS_IDX=${4:-3}

echo "[act/iterate_rollout.sh] CKPT_DIR: ${CKPT_DIR}"
echo "[act/iterate_rollout.sh] CKPT_NAME: ${CKPT_NAME}"
echo "[act/iterate_rollout.sh] SKIP: ${SKIP}"
echo "[act/iterate_rollout.sh] POLE_POS_IDX: ${POLE_POS_IDX}"

SCRIPT_DIR=$(cd $(dirname $0); pwd)

array=("task0_between-two" "task1_around-red" "task2_turn-blue" "task3_around-two")
for TASK_NAME in "${array[@]}"; do
    echo "[act/iterate_rollout.sh] task_name: ${TASK_NAME}"
    python ${SCRIPT_DIR}/../bin/rollout.py \
--ckpt_dir ${CKPT_DIR} --ckpt_name ${CKPT_NAME} --task_name ${TASK_NAME} \
--skip ${SKIP} \
--policy_class ACT --chunk_size 20 --num_epochs 0 \
--kl_weight 10 \
--hidden_dim 512 \
--dim_feedforward 3200 \
--seed 0 \
--temporal_agg \
--multi_task \
--win_xy_policy 0 600 --win_xy_simulation 900 0 \
--pole-pos-idx ${POLE_POS_IDX}
done

