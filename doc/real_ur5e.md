# Experiments with real UR5e

https://github.com/user-attachments/assets/1b894ea0-6b5a-4bf8-abaa-8cd3ec2c4f85

## Install
See [here](./install.md#Real-UR5e-environments) for installation.

## Preparation
Specify the IP address of UR5e in `robot_ip` and the serial number of RealSense in `camera_ids` in the following files:
- [RealUR5eDemoEnv.yaml](../robo_manip_baselines/envs/configs/RealUR5eDemoEnv.yaml)

## Teleoperation
```console
$ cd robo_manip_baselines/teleop/
$ python ./bin/Teleop.py RealUR5eDemo --config ../envs/configs/RealUR5eDemoEnv.yaml
```

## Policy rollout
```console
# SARNN
$ cd robo_manip_baselines/rollout/
$ python ./bin/Rollout.py Act RealUR5eDemo --config ../envs/configs/RealUR5eDemoEnv.yaml --checkpoint <checkpoint_path> --wait_before_start
```
