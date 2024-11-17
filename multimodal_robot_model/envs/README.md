# Environments for robot manipulation

## Install
### [Real UR5e environments](./real)
Install [ur_rtde](https://sdurobotics.gitlab.io/ur_rtde/installation/installation.html) by the following commands.
```console
$ pip install ur_rtde
```

Install [Python wrapper of Intel RealSense SDK](https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/readme.md) by the following commands.
```console
$ pip install pyrealsense2
```

Install [gello_software](https://github.com/wuphilipp/gello_software) by the following commands.
```console
$ # Go to the top directory of this repository
$ git submodule update --init --recursive
$ cd third_party/gello_software
$ pip install -r requirements.txt
$ pip install -e .
```
Since only interface classes accessing Robotiq gripper and RealSense camera are used from gello_software, errors in other parts can be ignored.

### [MuJoCo environments](./mujoco)
Follow the installation procedure [here](../../README.md#Install).

### [Isaac environments](./isaac)
Isaac Gym supports only Python 3.6, 3.7 and 3.8.
In Ubuntu 22.04, use Python 3.8 with [pyenv](https://github.com/pyenv/pyenv).

Download and unpack the Isaac Gym package from [here](https://developer.nvidia.com/isaac-gym).

Install Isaac Gym according to `IsaacGym_Preview_4_Package/isaacgym/doc/install.html` by the following commands.
```console
$ cd IsaacGym_Preview_4_Package/isaacgym/python
$ pip install -e .
```

Confirm that the sample program can be executed.
```console
$ cd IsaacGym_Preview_4_Package/isaacgym/python/examples
$ python joint_monkey.py
```

Isaac Gym and MuJoCo version 3 are conflicted by a file of the same name, `libsdf.so`, which triggers the following error: `undefined symbol: _ZN32pxrInternal_v0_19__pxrReserved__17 SdfValueTypeNamesE`. Downgrade MuJoCo by the following command.
```console
$ pip install mujoco==2.3.7
```
