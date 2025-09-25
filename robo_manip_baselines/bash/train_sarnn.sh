#!/bin/bash

source ../../rmb_env


for _ in {1..5}; do
  python3 ./bin/Train.py ActionembSarnn --dataset_dir dataset/MujocoUR5eCable_20250609_sentence/ --num_epochs 50000 --seed $RANDOM
done

