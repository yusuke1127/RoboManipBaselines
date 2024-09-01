# [MultimodalRobotModel](https://github.com/isri-aist/MultimodalRobotModel)
Imitation Learning of Robot Manipulation Based on Multimodal Sensing

## Install
1. Install Pinocchio. In Ubuntu 20.04, install it from robotpkg apt repository; in Ubuntu 22.04, install it with pip.  
https://stack-of-tasks.github.io/pinocchio/download.html#Install_4

2. Install this package via pip.
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

## Utilities
See [utils](./multimodal_robot_model/utils).

## Demonstration data collection
See [demos](./multimodal_robot_model/demos).
