# Quick start
This quick start allows you to collect data in the MuJoCo simulation and train and rollout the ACT policy.

## Install
Install RoboManipBaselines:
```console
$ git clone git@github.com:isri-aist/RoboManipBaselines.git --recursive
$ cd RoboManipBaselines
$ pip install -e .[act]
```

Install ACT from a third party:
```console
$ cd third_party/act/detr
$ pip install -e .
```

## Data collection by teleoperation
**Note**: Instead of collecting data by teleoperation, you can download the public dataset `TeleopMujocoUR5eCable_Dataset30` from [here](./dataset_list.md#Demonstrations-in-MuJoCo-environments).

Operate the robot in the simulation and save the data:
```console
$ cd robo_manip_baselines/teleop
$ # Connect a SpaceMouse 3D mouse to your PC
$ python bin/TeleopMujocoUR5eCable.py --world_idx_list 0 5
```
In our experience, models can be trained stably with roughly 30 data sets.
The teleoperation data is saved in the `robo_manip_baselines/teleop/teleop_data/MujocoUR5eCable_<date_suffix>` directory (e.g., `MujocoUR5eCable_20240101_120000`) in npz format.

## Model training
Train the ACT:
```console
$ cd robo_manip_baselines/act
$ python ../utils/make_dataset.py \
--in_dir ../teleop/teleop_data/MujocoUR5eCable_20240101_120000 \
--out_dir ./data/MujocoUR5eCable_20240101_120000 \
--train_ratio 0.8 --nproc `nproc` --skip 3
$ python ./bin/TrainAct.py \
--dataset_dir ./data/MujocoUR5eCable_20240101_120000 \
--log_dir ./log/MujocoUR5eCable_20240101_120000
```
**Note**: The following error will occur if the chunk_size is larger than the time series length of the training data.
In such a case, either set the `--skip` option in `make_dataset.py` to a small value, or set the `--chunk_size` option in `TrainAct.py` to a small value.
```console
RuntimeError: The size of tensor a (70) must match the size of tensor b (102) at non-singleton dimension 0
```

## Policy rollout
Rollout the ACT in the simulation:
```console
$ cd robo_manip_baselines/act
$ python ./bin/rollout/RolloutActMujocoUR5eCable.py \
--checkpoint ./log/MujocoUR5eCable_20240101_120000/policy_last.ckpt \
--skip 3 --world_idx 0
```
