#!/bin/bash

[[ $# < 1 ]] && echo "$0 <CKPT_DIR> [<CKPT_NAME>]" && exit 1

CKPT_DIR=$1
CKPT_NAME=${2:-SARNN.pth}

echo "[sarnn/iterate_rollout.sh] CKPT_DIR: ${CKPT_DIR}"

SCRIPT_DIR=$(cd $(dirname $0); pwd)

array=(0 1 2 3 4 5)
for i in "${array[@]}"; do
    echo "[sarnn/iterate_rollout.sh] pole-pos-idx: $i"
    python ${SCRIPT_DIR}/../bin/rollout.py \
--filename ${CKPT_DIR}/${CKPT_NAME} \
--win_xy_policy 0 700 --win_xy_simulation 900 0 \
--pole-pos-idx $i
done
