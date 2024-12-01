# [MultimodalRobotModel](https://github.com/isri-aist/MultimodalRobotModel)
Software that integrates various imitation learning methods and benchmark task environments to provide baselines for robot manipulation

## Quick start
[This quick start](./doc/quick_start.md) allows you to collect data in the simulation and train and rollout the ACT.

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
See [the environment catalog](doc/environment_catalog.md) for a full list of environments.

See [envs](./multimodal_robot_model/envs) for installation procedures for each environment.

## Utilities
See [utils](./multimodal_robot_model/utils).

## License
Files that originate from this repository are subject to the BSD 2-Clause License. If a file explicitly states a different license, or if there are different license files in a directory, those licenses will take precedence. For files in third-party directories, please follow the respective licenses.
