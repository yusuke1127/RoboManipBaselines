# [MultimodalRobotModel](https://github.com/isri-aist/MultimodalRobotModel)
Imitation Learning of Robot Manipulation Based on Multimodal Sensing

## Quick start
[This quick start](./doc/QuickStart.md) allows you to collect data in the simulation and train and rollout the ACT.

## Install
Install Pinocchio according to [here](https://stack-of-tasks.github.io/pinocchio/download.html#Install_4).
In Ubuntu 20.04, install it from robotpkg apt repository; in Ubuntu 22.04, install it with pip.

Install this package via pip by the following commands.
```console
$ pip install -e .
```

## Models
### [SARNN](./multimodal_robot_model/sarnn)
Spatial attention recurrent neural network

### [ACT](./multimodal_robot_model/act)
Action Chunking with Transformers

### [DiffusionPolicy](./multimodal_robot_model/diffusion_policy)
Diffusion Policy

### [MT-ACT](./multimodal_robot_model/mt_act)
Multi-Task Action Chunking Transformer

## Data collection by teleoperation
See [teleop](./multimodal_robot_model/teleop).

## Environments for robot manipulation
See [envs](./multimodal_robot_model/envs).

## Utilities
See [utils](./multimodal_robot_model/utils).
