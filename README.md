# [RoboManipBaselines](https://github.com/isri-aist/RoboManipBaselines)
Software that integrates various imitation learning methods and benchmark task environments to provide baselines for robot manipulation

[![CI-install](https://github.com/isri-aist/RoboManipBaselines/actions/workflows/install.yml/badge.svg)](https://github.com/isri-aist/RoboManipBaselines/actions/workflows/install.yml)
[![CI-pre-commit](https://github.com/isri-aist/RoboManipBaselines/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/isri-aist/RoboManipBaselines/actions/workflows/pre-commit.yml)
[![LICENSE](https://img.shields.io/github/license/isri-aist/RoboManipBaselines)](https://github.com/isri-aist/RoboManipBaselines/blob/master/LICENSE)

## Quick start
[This quick start](./doc/quick_start.md) allows you to collect data in the MuJoCo simulation and train and rollout the ACT policy.

## Install
See [the installation documentation](./doc/install.md).

## Models
### [SARNN](./robo_manip_baselines/sarnn)
Spatial attention recurrent neural network

### [ACT](./robo_manip_baselines/act)
Action Chunking with Transformers

### [DiffusionPolicy](./robo_manip_baselines/diffusion_policy)
Diffusion Policy

### [MT-ACT](./robo_manip_baselines/mt_act)
Multi-Task Action Chunking Transformer

## Data collection by teleoperation
See [teleop](./robo_manip_baselines/teleop).

## Environments for robot manipulation
See [the environment catalog](doc/environment_catalog.md) for a full list of environments.

See [envs](./robo_manip_baselines/envs) for installation procedures for each environment.

## Utilities
See [utils](./robo_manip_baselines/utils).

## License
Files that originate from this repository are subject to the BSD 2-Clause License. If a file explicitly states a different license, or if there are different license files in a directory, those licenses will take precedence. For files in third-party directories, please follow the respective licenses.
