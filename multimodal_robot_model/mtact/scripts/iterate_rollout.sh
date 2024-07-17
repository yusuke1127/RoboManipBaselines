#!/bin/bash

[[ $# < 1 ]] && echo "$0 <CKPT_DIR> [<CKPT_NAME>]" && exit 1

CKPT_DIR=$1
CKPT_NAME=${2:-policy_best.ckpt}
SKIP=${3:-1}
CHUNK_SIZE=${4:-20}

echo "[act/iterate_rollout.sh] CKPT_DIR: ${CKPT_DIR}"
echo "[act/iterate_rollout.sh] CKPT_NAME: ${CKPT_NAME}"
echo "[act/iterate_rollout.sh] SKIP: ${SKIP}"
echo "[act/iterate_rollout.sh] CHUNK_SIZE: ${CHUNK_SIZE}"

SCRIPT_DIR=$(cd $(dirname $0); pwd)

array=(3)
for i in "${array[@]}"; do
    echo "[act/iterate_rollout.sh] pole-pos-idx: $i"
    python ${SCRIPT_DIR}/../bin/rollout.py \
--ckpt_dir ${CKPT_DIR} --ckpt_name ${CKPT_NAME} --task_name "task1_around-red" \
--skip ${SKIP} \
--policy_class ACT --chunk_size ${CHUNK_SIZE} --num_epochs 0 \
--kl_weight 10 \
--hidden_dim 512 \
--dim_feedforward 3200 \
--seed 0 \
--temporal_agg \
--multi_task \
--win_xy_policy 0 600 --win_xy_simulation 900 0 \
--pole-pos-idx $i
done
