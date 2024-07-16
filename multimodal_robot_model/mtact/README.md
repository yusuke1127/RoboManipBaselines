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
mv multimodal_robot_model/mtact/roboagent-overwrite/constants.py third_party/roboagent/constants.py
mv multimodal_robot_model/mtact/roboagent-overwrite/detr/models/detr_vae.py third_party/roboagent/detr/models/detr_vae.py
mv multimodal_robot_model/mtact/roboagent-overwrite/train.py third_party/roboagent/train.py
mv multimodal_robot_model/mtact/roboagent-overwrite/utils.py third_party/roboagent/utils.py
```

## Dataset preparation

Put your data collected under `data` directory. Here, we assume the name of your dataset directory as `teleop_data_sample`.

```console
$ tree data/teleop_data_sample/
data/teleop_data_sample/
├── task0_between-two
│   └── env3
│       ├── UR5eCableEnv_env3_000.npz
│       └── UR5eCableEnv_env3_001.npz
├── task1_around-red
│   └── env3
│       ├── UR5eCableEnv_env3_000.npz
│       └── UR5eCableEnv_env3_001.npz
├── task2_turn-blue
│   └── env3
│       ├── UR5eCableEnv_env3_000.npz
│       └── UR5eCableEnv_env3_001.npz
└── task3_around-two
    └── env3
        ├── UR5eCableEnv_env3_000.npz
        └── UR5eCableEnv_env3_001.npz
```

Make numpy files in each of `train` (for training) and `test` directories (for validation).

```console
$ skip=2
$ train_ratio=0.8
$ python ../utils/make_multi_dataset.py \
--in_dir  ./data/ \
--out_dir ./data/skip_$skip/train_ratio_`echo $train_ratio|tr -d .` \
--skip $skip \
--train_ratio $train_ratio \
--nproc `nproc`
```

## Model Training

Train the model. The trained weights are saved in the `log` folder.

```console
$ skip=2
$ train_ratio=0.8
$ python ../third_party/roboagent/train.py \
--dataset_dir ./data/skip_$skip/train_ratio_`echo $train_ratio|tr -d .` \
--ckpt_dir     ./log/skip_$skip/train_ratio_`echo $train_ratio|tr -d .` \
--policy_class ACT \
--kl_weight 10 \
--chunk_size 20 \
--hidden_dim 512 \
--batch_size 64 \
--dim_feedforward 3200 \
--seed 0 \
--temporal_agg \
--num_epochs 2000 \
--lr 1e-5 \
--multi_task \
--task_name pick_butter \
--run_name multi_task_run
```

