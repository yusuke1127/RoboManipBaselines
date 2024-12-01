# Diffusion Policy

## Install

Install [r3m](https://github.com/facebookresearch/r3m) by the following commands.
Install [diffusion policy](https://github.com/real-stanford/diffusion_policy) by the following commands.

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
$ python ../utils/convert_npz_to_zarr.py ./data/teleop_data_sample --train_keywords env0 env5 --nproc `nproc`
```

If you are using `pyenv` and encounter the error `No module named '_bz2'`, apply the following solution.  
https://stackoverflow.com/a/71457141

## Model Training

Train the model. The trained weights are saved in the `log` folder.

```console
$ python ./bin/TrainDiffusionPolicy.py \
--config-dir=./lib --config-name=RmbDiffusionPolicy.yaml \
task.dataset.zarr_path=data/teleop_data_sample/learning_data.zarr
```
To disable logging by wandb, add the option `enable_wandb=False`.

### Trouble-shooting

If you encounter the following error,
```console
ImportError: cannot import name 'cached_download' from 'huggingface_hub'
```
downgrade `huggingface_hub` by the following command.
```console
$ pip install huggingface_hub==0.21.4
```

## Policy rollout

Run a trained policy in the simulator.

```console
$ python ./bin/rollout/RolloutDiffusionPolicyMujocoUR5eCable.py \
--checkpoint ./log/YYYY.MM.DD/HH.MM.SS_train_diffusion_unet_hybrid_MujocoUR5eCable/checkpoints/200.ckpt \
--skip 3 --world_idx 1
```
The Python script is named `RolloutDiffusionPolicy<task_name>.py`. The followings are supported as task_name: `MujocoUR5eCable`, `MujocoUR5eRing`, `MujocoUR5eParticle`, `MujocoUR5eCloth`.

Repeatedly run a trained policy in different environments in the simulator.

```console
$ ./scripts/iterate_rollout.sh ./log/YYYY.MM.DD/HH.MM.SS_train_diffusion_unet_hybrid_MujocoUR5eCable/checkpoints/200.ckpt MujocoUR5eCable 3
```
