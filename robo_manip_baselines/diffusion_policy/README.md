# Diffusion Policy

## Install
See [here](../../doc/install.md#Diffusion-policy) for installation.

## Dataset preparation
Collect demonstration data by [teleoperation](../teleop).

Generate a `zarr` format dataset for learning from teleoperation data:
```console
$ python ../utils/convert_npz_to_zarr.py \
--in_dir ../teleop/teleop_data/<demo_name> --out_dir ./data/<demo_name>.zarr \
--nproc `nproc` --skip 3
```

**Note**: If you are using `pyenv` and encounter the error `No module named '_bz2'`, apply the following solution.  
https://stackoverflow.com/a/71457141

## Model training
Train a model:
```console
$ python ./bin/TrainDiffusionPolicy.py \
--config-dir=./lib --config-name=RmbDiffusionPolicy.yaml \
task.dataset.zarr_path=./data/<demo_name>.zarr task.name=<demo_name>
```
To disable logging by WandB, add the option `--enable_wandb=False`.
You can override the `yaml` configuration file by adding the following command line arguments, for example: `task.shape_meta.obs.joint.shape=\[8\] task.shape_meta.action.shape=\[8\]`.
The checkpoint files are saved in the `log` directory.

**Note**: If you encounter the following error,
```console
ImportError: cannot import name 'cached_download' from 'huggingface_hub'
```
downgrade `huggingface_hub` by the following command.
```console
$ pip install huggingface_hub==0.21.4
```

## Policy rollout
Run a trained policy:
```console
$ python ./bin/rollout/RolloutDiffusionPolicyMujocoUR5eCable.py \
--checkpoint ./log/<demo_name>/checkpoints/200.ckpt \
--skip 3 --world_idx 0
```
