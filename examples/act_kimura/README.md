## Requirements
### Installing ACT

Install [ACT](https://github.com/tonyzhaozh/act) under `third_party/act/` according to [act/README.md](https://github.com/tonyzhaozh/act/blob/main/README.md).
``` console
$ cd MultimodalRobotModel/
$ git clone https://github.com/tonyzhaozh/act
$ pip install torchvision torch pyquaternion pyyaml rospkg pexpect mujoco==2.3.7 dm_control==1.0.14 opencv-python matplotlib einops packaging h5py ipython
$ cd act/detr && pip install -e .
```
Replace 14 with 7 in act/detr/models/detr_vae.py .
``` console
$ sed -ir "s/14/7/g" act/detr/models/detr_vae.py
```

### Installing EIPL
To run `./bin/2_make_dataset.py`, install [EIPL](https://github.com/ogata-lab/eipl) according to `third_party/eipl/README.md`.
```console
$ # clone submodule resources
$ cd MultimodalRobotModel
$ git submodule update --init --recursive

$ # install EIPL
$ cd third_party/eipl
$ pip install -r requirements.txt
$ pip install -e .
```

## Dataset preparation

Put your data collected under `data` directory. Here, we assume the name of your dataset directory as `teleop_data_00000000`. 

```console
$ tree data/teleop_data_00000000/
data/teleop_data_00000000/
├── env0
│   ├── UR5eCableEnv_env0_000.npz
│   └── UR5eCableEnv_env0_006.npz
├── env1
│   ├── UR5eCableEnv_env1_001.npz
│   └── UR5eCableEnv_env1_007.npz
├── env2
│   ├── UR5eCableEnv_env2_002.npz
│   └── UR5eCableEnv_env2_008.npz
├── env3
│   ├── UR5eCableEnv_env3_003.npz
│   └── UR5eCableEnv_env3_009.npz
├── env4
│   ├── UR5eCableEnv_env4_004.npz
│   └── UR5eCableEnv_env4_010.npz
└── env5
    ├── UR5eCableEnv_env5_005.npz
    └── UR5eCableEnv_env5_011.npz
```

Run `./bin/2_make_dataset.py` to make NPZ files in each of `train` (for training) and `test` directories (for validation), in `teleop_data_00000000`.

```console
# e.g.
$ cd examples/act_kimura/
$ python3 ./bin/2_make_dataset.py --in_dir ./data/teleop_data_20240414 --train_keywords env0 env1 env2 env4 env5 --test_keywords env3 --nproc `nproc`
```

## Model Training

Run `./bin/train.py` to start training the model. The trained weights are saved in the log folder.

```console
# e.g.
$ python3 ./bin/imitate_episodes.py \
--task_name sim_ur5ecable \
--ckpt_dir ./data/ckpt \
--policy_class ACT --kl_weight 10 --chunk_size 50 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
--num_epochs 20  --lr 1e-5 \
--seed 0
```

## Model Evaluation

Note: The following command will terminate abnormally with `raise NotImplementedError` because the work is in progress.

```console
# e.g.
$ python3 ./bin/imitate_episodes.py \
--task_name sim_ur5ecable \
--ckpt_dir ./data/ckpt \
--policy_class ACT --kl_weight 10 --chunk_size 50 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
--num_epochs 20  --lr 1e-5 \
--seed 0 --eval --onscreen_render
```

