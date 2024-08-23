#!/bin/bash

[[ $# < 1 ]] && echo "$0 <CKPT_DIR> [<CKPT_NAME>]" && exit 1

CKPT_DIR=$1
CKPT_NAME=${2:-SARNN.pth}
SKIP=${3:-6}

echo "[sarnn/iterate_rollout.sh] CKPT_DIR: ${CKPT_DIR}"
echo "[sarnn/iterate_rollout.sh] CKPT_NAME: ${CKPT_NAME}"
echo "[sarnn/iterate_rollout.sh] SKIP: ${SKIP}"

SCRIPT_DIR=$(cd $(dirname $0); pwd)
FIRST_OPTION="--wait_before_start"

array=(0 1 2 3 4 5)
for i in "${array[@]}"; do
    echo "[sarnn/iterate_rollout.sh] pole-pos-idx: $i"
    python ${SCRIPT_DIR}/../bin/rollout.py \
--filename ${CKPT_DIR}/${CKPT_NAME} \
--skip ${SKIP} \
--win_xy_policy 0 700 --win_xy_simulation 900 0 \
--pole-pos-idx $i $FIRST_OPTION
    FIRST_OPTION=""
done
