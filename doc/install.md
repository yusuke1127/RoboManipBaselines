# Install

## Common installation for all policies
Install RoboManipBaselines:
```console
$ git clone git@github.com:isri-aist/RoboManipBaselines.git --recursive
$ cd RoboManipBaselines
$ pip install -e .
```

**Note**: If you have problems installing the Pinocchio library (`pin` module) from `pip` in certain environments (e.g. Ubuntu 20.04), you can also install it via `apt`. See [here](https://stack-of-tasks.github.io/pinocchio/download.html#Install) for details.

This common installation enables data collection by teleoperation in the MuJoCo environments.

## Installation of each policy
Complete [the common installation](#common-installation-for-all-policies) first.

### SARNN
```console
# Go to the top directory of this repository.
$ pip install -e .[sarnn]

# Go to the top directory of this repository.
$ cd third_party/eipl
$ pip install -e .
```

### ACT
```console
# Go to the top directory of this repository.
$ pip install -e .[act]

# Go to the top directory of this repository.
$ cd third_party/act/detr
$ pip install -e .
```

### Diffusion policy
```console
$ sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf

# Go to the top directory of this repository.
$ pip install -e .[diffusion-policy]

# Go to the top directory of this repository.
$ cd third_party/r3m
$ pip install -e .

# Go to the top directory of this repository.
$ cd third_party/diffusion_policy
$ pip install -e .
```

**Note**: If you encounter the following error,
```python
pip._vendor.packaging.requirements.InvalidRequirement: Expected end or semicolon (after version specifier)
    opencv-python>=3.
```
replace all `opencv-python>=3.` with `opencv-python>=3.0` in `<venv directory>/lib/python3.8/site-packages/gym-0.21.0-py3.8.egg-info/requires.txt`.
