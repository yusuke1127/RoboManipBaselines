# 3D Diffusion Policy

## Install
See [here](../../../doc/install.md#3D-Diffusion-policy) for installation.

## Dataset preparation
Collect demonstration data by [teleoperation](../../teleop).

> [!NOTE]
> This policy requires pointclouds instead of images.
> Use `misc/AddPointCloudToRmbData.py` to add pointclouds to collected data.
> ```console
> # Go to the top directory of this repository
> $ cd robo_manip_baselines
> $ python ./misc/AddPointCloudToRmbData.py ./dataset/<dataset_name>
> ```

## Model training
Train a model:
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ python ./bin/Train.py DiffusionPolicy3d --dataset_dir ./dataset/<dataset_name> --checkpoint_dir ./checkpoint/DiffusionPolicy3d/<checkpoint_name>
```

## Policy rollout
Run a trained policy:
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ python ./bin/Rollout.py DiffusionPolicy3d MujocoUR5eCable --checkpoint ./checkpoint/DiffusionPolicy3d/<checkpoint_name>/policy_last.ckpt
```

## Technical Details
For more information on the technical details, please see the following paper:
```bib
@inproceedings{Ze2024DP3,
	title={3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations},
	author={Yanjie Ze and Gu Zhang and Kangning Zhang and Chenyuan Hu and Muhan Wang and Huazhe Xu},
	booktitle={Proceedings of Robotics: Science and Systems (RSS)},
	year={2024}
}
```
