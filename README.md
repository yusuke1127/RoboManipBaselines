# [MultimodalRobotModel](https://github.com/isri-aist/MultimodalRobotModel)
Imitation Learning of Robot Manipulation Based on Multimodal Sensing

## Install
1. Install Pinocchio from the robotpkg apt repository.  
https://stack-of-tasks.github.io/pinocchio/download.html#Install_4

2. Install this package via pip.
```bash
$ pip install -e .
```

## Models
### [SARNN](./multimodal_robot_model/sarnn)
Spatial attention recurrent neural network

## Data collection
### Sample data
Sample data can be downloaded [here](https://www.dropbox.com/scl/fi/15r33msj4vd1potaosirh/teleop_data_20240414.zip?rlkey=2vt1h5gde7l42vrwz4axj10da&dl=0).

### Record teleoperation data
```bash
$ cd multimodal_robot_model/demos/
$ # Connect SpaceMouse to your PC.
$ python Demo_UR5eCableEnv_Teleop.py
```

### Playback teleoperation data
```bash
$ cd multimodal_robot_model/demos/
$ python Demo_UR5eCableEnv_Playback.py ./teleop_data/env0/UR5eCableEnv_env0_000.npz
```

https://github.com/isri-aist/MultimodalRobotModel/assets/6636600/df5665c2-97fc-4913-8891-27fbfb4fad52
