#!/bin/bash

[[ $# < 1 ]] && echo "$0 <CKPT_PATH> [<SKIP>]" && exit 1

CKPT_PATH=$1
SKIP=${2:-4}

echo "[diffusion_policy/iterate_rollout.sh] CKPT_PATH: ${CKPT_PATH}"
echo "[diffusion_policy/iterate_rollout.sh] SKIP: ${SKIP}"

SCRIPT_DIR=$(cd $(dirname $0); pwd)

array=(0 1 2 3 4 5)
for i in "${array[@]}"; do
    echo "[diffusion_policy/iterate_rollout.sh] pole-pos-idx: $i"
    python ${SCRIPT_DIR}/../bin/rollout.py \
--filename ${CKPT_PATH} \
--skip ${SKIP} \
--win_xy_policy 0 700 --win_xy_simulation 900 0 \
--pole-pos-idx $i
done
