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

## Data collection in MuJoCo
### Sample data
Sample data can be downloaded [here](https://www.dropbox.com/scl/fi/15r33msj4vd1potaosirh/teleop_data_20240414.zip?rlkey=2vt1h5gde7l42vrwz4axj10da&dl=0).

### Record teleoperation data
```console
$ cd multimodal_robot_model/demos/
$ # Connect SpaceMouse to your PC.
$ python Demo_UR5eCableEnv_Teleop.py
```

To add a 3D plot of the point cloud, add the following option:
```console
$ python Demo_UR5eCableEnv_Teleop.py --enable-3d-plot
```
If you cannot zoom the point cloud view by right-clicking, try changing the matplotlib version: `pip install matplotlib=="3.6.1"`.

### Playback teleoperation data
```console
$ cd multimodal_robot_model/demos/
$ python Demo_UR5eCableEnv_Playback.py ./teleop_data/env0/UR5eCableEnv_env0_000.npz
```

https://github.com/isri-aist/MultimodalRobotModel/assets/6636600/df5665c2-97fc-4913-8891-27fbfb4fad52
