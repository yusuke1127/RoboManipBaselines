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
Operate the robot in the simulation and save the data by the following commands.
In our experience, models can be trained stably with roughly 30 data sets.
The teleoperation data is saved in the `robo_manip_baselines/teleop/teleop_data/MujocoUR5eCable` directory in npz format.
```console
$ cd robo_manip_baselines/teleop
$ # Connect a SpaceMouse 3D mouse to your PC
$ python bin/TeleopMujocoUR5eCable.py --world_idx_list 0 5
```

## Model training
Train the ACT by the following commands.
```console
$ cd robo_manip_baselines/act
$ python ../utils/make_dataset.py --in_dir ../teleop/teleop_data/MujocoUR5eCable --out_dir ./data/MujocoUR5eCable --train_ratio 0.8 --nproc `nproc` --skip 3
$ python ./bin/TrainAct.py --dataset_dir ./data/MujocoUR5eCable --ckpt_dir ./log/MujocoUR5eCable
```
Note that the following error will occur if the chunk_size is larger than the time series length of the training data.
In such a case, either set the `--skip` option in `make_dataset.py` to a small value, or set the `--chunk_size` option in `TrainAct.py` to a small value.
```console
RuntimeError: The size of tensor a (70) must match the size of tensor b (102) at non-singleton dimension 0
```

## Policy rollout
Rollout the ACT in the simulation by the following commands.
```console
$ cd robo_manip_baselines/act
$ python ./bin/rollout/RolloutActMujocoUR5eCable.py \
--ckpt_dir ./log/YEAR_DAY_TIME --ckpt_name policy_last.ckpt \
--skip 3 --world_idx 0
```
