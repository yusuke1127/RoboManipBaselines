# [MultimodalRobotModel](https://github.com/isri-aist/MultimodalRobotModel)
Imitation Learning of Robot Manipulation Based on Multimodal Sensing

## Install
1. Install Pinocchio from the robotpkg apt repository.  
https://stack-of-tasks.github.io/pinocchio/download.html#Install_4

2. Install this package via pip.
```bash
$ pip install -e .
```

## Sample
### Record teleoperation data
```bash
$ cd multimodal_robot_model/demos/
# Connect SpaceMouse to your PC.
$ python Demo_UR5eCableEnv_Teleop.py
```

### Playback teleoperation data
```bash
$ cd multimodal_robot_model/demos/
$ python Demo_UR5eCableEnv_Playback.py ./teleop_data/env0/UR5eCableEnv_env0_000.npz
```

https://github.com/isri-aist/MultimodalRobotModel/assets/6636600/df5665c2-97fc-4913-8891-27fbfb4fad52
