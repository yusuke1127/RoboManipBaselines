# Diffusion Policy

## Install

Install package
```console
$ sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
$ pip install -r requirements.txt
$ # If urllib3 version is greater than 2, execute this command
$ pip install 'urllib3<2'
```

Install [EIPL](https://github.com/ogata-lab/eipl).
```console
$ # Go to the top directory of this repository
$ git submodule update --init --recursive
$ cd third_party/eipl
$ pip install -r requirements.txt
$ pip install -e .
```

Install [MultimodalRobotModel](https://github.com/isri-aist/MultimodalRobotModel).
```console
$ # Go to the top directory of this repository
$ pip install -e .
```

Install [r3m](https://github.com/facebookresearch/r3m).
```console
$ # Go to the top directory of this repository
$ cd third_party/r3m
$ pip install -e .
```

Install [diffusion policy](https://github.com/real-stanford/diffusion_policy).
```console
$ # Go to the top directory of this repository
$ cd third_party/diffusion_policy
$ pip install -e .
```

## Dataset preparation

Put your data collected under `data` directory. Here, we assume the name of your dataset directory as `teleop_data_sample`.

```console
$ tree data/teleop_data_sample/
data/teleop_data_sample/
├── env0
│   ├── UR5eCableEnv_env0_000.npz
│   └── UR5eCableEnv_env0_006.npz
├── env1
│   ├── UR5eCableEnv_env1_001.npz
│   └── UR5eCableEnv_env1_007.npz
├── env2
│   ├── UR5eCableEnv_env2_002.npz
│   └── UR5eCableEnv_env2_008.npz
├── env3
│   ├── UR5eCableEnv_env3_003.npz
│   └── UR5eCableEnv_env3_009.npz
├── env4
│   ├── UR5eCableEnv_env4_004.npz
│   └── UR5eCableEnv_env4_010.npz
└── env5
    ├── UR5eCableEnv_env5_005.npz
    └── UR5eCableEnv_env5_011.npz
```

Make zarr file.

```console
$ python ../utils/convert_npz_to_zarr.py ./data/teleop_data_sample
```

Replace line 115 of mujoco_diffusion_policy_cnn.yaml with the following

```console
zarr_path: data/mujoco/<your zarr file name>.zarr
```

## Model Training

Train the model. The trained weights are saved in the `log` folder.

```console
$ python ./bin/train.py --config-dir=. --config-name=mujoco_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='log/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
```

## Policy rollout

Run a trained policy in the simulator.

```console
$ python ./bin/rollout.py --filename ./log/YEAR.DAY/TIME_POLICY_mujoco/checkpoints/latest.ckpt --pole-pos-idx 1
```
