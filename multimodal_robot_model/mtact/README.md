# Multi-Task Action Chunking Transformer (MT-ACT)

## Install

Clone roboagent.
``` console
$ cd third_party/
$ git clone https://github.com/robopen/roboagent.git
``` 

Install the required packages. 
``` console
$ pip install \
torchvision \
torch \
pyquaternion \
pyyaml \
rospkg \
pexpect \
mujoco \
dm_control \
opencv-python \
matplotlib \
einops \
packaging \
h5py \
h5py_cache
$ pip install ipython lru_cache click
$ cd ../third_party/roboagent/detr && pip install -e .
```

Install [EIPL](https://github.com/ogata-lab/eipl).
``` console
$ # Go to the top directory of this repository
$ git submodule update --init --recursive
$ cd third_party/eipl
$ pip install -r requirements.txt
$ pip install -e .
```

Overwrite some of the code in roboagent.
```console
$ mv multimodal_robot_model/mtact/roboagent-overwrite/constants.py third_party/roboagent/constants.py \
&& mv multimodal_robot_model/mtact/roboagent-overwrite/detr/main.py third_party/roboagent/detr/main.py \
&& mv multimodal_robot_model/mtact/roboagent-overwrite/detr/models/detr_vae.py third_party/roboagent/detr/models/detr_vae.py \
&& mv multimodal_robot_model/mtact/roboagent-overwrite/train.py third_party/roboagent/train.py \
&& mv multimodal_robot_model/mtact/roboagent-overwrite/utils.py third_party/roboagent/utils.py
```

Install [MultimodalRobotModel](https://github.com/isri-aist/MultimodalRobotModel) (if you only want model training, `pinocchio` is not required).
```console
$ # Go to the top directory of this repository
$ pip install -e .
```
**Note**: The above installation resulted the following error message. I ignored this error message and proceeded the next steps.
```console
ERROR: dm-control 1.0.20 has requirement mujoco>=3.1.6, but you'll have mujoco 2.3.7 which is incompatible.
```

## Dataset preparation

Put your data collected under `data` directory. Here, we assume the name of your dataset directory as `teleop_data_sample`.
Data for multiple tasks has been uploaded to [mujoco_ur5e_cable_wiring > teleop_data > 20240711](https://aist.box.com/s/9qtkspyyzcxqvrssvumahfgvi31h5cet).

```console
$ tree data/teleop_data_sample/
data/teleop_data_sample/
├── task0_between-two
│   ├── env1
│   │   ├── UR5eCableEnv_env1_000.npz
│   │   └── UR5eCableEnv_env1_001.npz
│   ├── env3
│   │   ├── UR5eCableEnv_env3_000.npz
│   │   └── UR5eCableEnv_env3_001.npz
│   └── env5
│       ├── UR5eCableEnv_env5_000.npz
│       └── UR5eCableEnv_env5_001.npz
├── task1_around-red
│   ├── env1
│   │   ├── UR5eCableEnv_env1_000.npz
│   │   └── UR5eCableEnv_env1_001.npz
│   ├── env3
│   │   ├── UR5eCableEnv_env3_000.npz
│   │   └── UR5eCableEnv_env3_001.npz
│   └── env5
│       ├── UR5eCableEnv_env5_000.npz
│       └── UR5eCableEnv_env5_001.npz
├── task2_turn-blue
│   ├── env1
│   │   ├── UR5eCableEnv_env1_000.npz
│   │   └── UR5eCableEnv_env1_001.npz
│   ├── env3
│   │   ├── UR5eCableEnv_env3_000.npz
│   │   └── UR5eCableEnv_env3_001.npz
│   └── env5
│       ├── UR5eCableEnv_env5_000.npz
│       └── UR5eCableEnv_env5_001.npz
└── task3_around-two
    ├── env1
    │   ├── UR5eCableEnv_env1_000.npz
    │   └── UR5eCableEnv_env1_001.npz
    ├── env3
    │   ├── UR5eCableEnv_env3_000.npz
    │   └── UR5eCableEnv_env3_001.npz
    └── env5
        ├── UR5eCableEnv_env5_000.npz
        └── UR5eCableEnv_env5_001.npz
```

Make numpy files in each of `train` (for training) and `test` directories (for validation).

```console
$ python ../utils/make_multi_dataset.py \
--in_dir ./data/teleop_data_sample \
--out_dir ./data/learning_data_sample \
--skip 2 \
--train_keywords env1 env5 \
--test_keywords env3 \
--nproc `nproc`
```

## Model Training

Train the model. The trained weights are saved in the `log` folder.
The training hyperparameters here (such as chunk_size) are the same as those in [act#training-models](https://github.com/isri-aist/MultimodalRobotModel/tree/master/multimodal_robot_model/act#model-training).

```console
$ python ../third_party/roboagent/train.py \
--dataset_dir ./data/learning_data_sample --ckpt_dir ./log/YEAR_DAY_TIME \
--policy_class ACT \
--kl_weight 10 \
--chunk_size 100 \
--hidden_dim 512 \
--batch_size 8 \
--dim_feedforward 3200 \
--seed 0 \
--temporal_agg \
--num_epochs 20000 \
--lr 1e-5 \
--multi_task \
--run_name multi_task_run
```

## Policy rollout
Run a trained policy in the simulator.

```console
$ python ./bin/rollout.py \
--ckpt_dir ./log/YEAR_DAY_TIME --ckpt_name policy_best.ckpt --task_name task0_between-two \
--skip 2 \
--policy_class ACT --chunk_size 100 --num_epochs 0 \
--kl_weight 10 \
--hidden_dim 512 \
--dim_feedforward 3200 \
--seed 0 \
--multi_task \
--pole-pos-idx 0
```

Repeatedly run a trained policy in different environments in the simulator.

```console
$ ./scripts/iterate_rollout.sh ./log/YEAR_DAY_TIME/ policy_last.ckpt
```
