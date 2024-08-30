# Diffusion Policy

## Install

Install dependent packages.
```console
$ sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
$ sudo aptitude install libavdevice-dev libavfilter-dev # Required for Ubuntu 22.04 / Python 3.8.16
$ pip install -r requirements.txt
$ # If urllib3 version is greater than 2, execute this command
$ pip install 'urllib3<2'
```

Install [r3m](https://github.com/facebookresearch/r3m).
```console
$ # Go to the top directory of this repository
$ git submodule update --init --recursive
$ cd third_party/r3m
$ pip install -e .
```

Install [diffusion policy](https://github.com/real-stanford/diffusion_policy).
```console
$ # Go to the top directory of this repository
$ cd third_party/diffusion_policy
$ pip install -e .
```

Install [MultimodalRobotModel](https://github.com/isri-aist/MultimodalRobotModel) (if you only want model training, `pinocchio` is not required).
```console
$ # Go to the top directory of this repository
$ pip install -e .
```

### Trouble-shooting

If you encounter the following error,
```python
pip._vendor.packaging.requirements.InvalidRequirement: Expected end or semicolon (after version specifier)
    opencv-python>=3.
```
replace all `opencv-python>=3.` with `opencv-python>=3.0` in `<venv directory>/lib/python3.8/site-packages/gym-0.21.0-py3.8.egg-info/requires.txt`.

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

Make zarr file (for training).

```console
$ python ../utils/convert_npz_to_zarr.py ./data/teleop_data_sample --train_keywords env0 env5
```

If you are using `pyenv` and encounter the error `No module named '_bz2'`, apply the following solution.  
https://stackoverflow.com/a/71457141

## Model Training

Train the model. The trained weights are saved in the `log` folder.

```console
$ python ./bin/train.py \
--config-dir=./lib/config/ --config-name=mujoco_diffusion_policy_cnn.yaml \
task.dataset.zarr_path=data/teleop_data_sample/learning_data.zarr
```
To disable logging by wandb, add the option `enable_wandb=False`.

## Policy rollout

Run a trained policy in the simulator.

```console
$ python ./bin/rollout.py \
--filename ./log/YYYY.MM.DD/HH.MM.SS_train_diffusion_unet_hybrid_MujocoUR5eCable/checkpoints/200.ckpt \
--pole-pos-idx 1
```

Repeatedly run a trained policy in different environments in the simulator.

```console
$ ./scripts/iterate_rollout.sh ./log/YYYY.MM.DD/HH.MM.SS_train_diffusion_unet_hybrid_MujocoUR5eCable/checkpoints/200.ckpt
```
