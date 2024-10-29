# Quick start
This quick start allows you to collect data in the simulation and train and rollout the ACT.

## Install
Install according to [here](../multimodal_robot_model/act/README.md#Install) by the following commands.
```console
# Clone this repository
$ git clone git@github.com:isri-aist/MultimodalRobotModel.git --recursive

# Install EIPL
$ # Go to the top directory of this repository
$ git submodule update --init --recursive
$ cd third_party/eipl
$ pip install -r requirements.txt
$ pip install -e .

# Install this package
$ # Go to the top directory of this repository
$ pip install -e .

# Install ACT
$ # Go to the top directory of this repository
$ git submodule update --init --recursive
$ cd third_party/act
$ pip install torchvision torch pyquaternion pyyaml rospkg pexpect mujoco==3.1.6 \
 dm_control==1.0.14 opencv-python matplotlib einops packaging h5py ipython
$ cd detr
$ pip install -e .
```

Install Pinocchio according to [here](https://stack-of-tasks.github.io/pinocchio/download.html#Install_4).
In Ubuntu 20.04, install it from robotpkg apt repository; in Ubuntu 22.04, install it with pip.

## Data collection by teleoperation
Operate the robot in the simulation and save the data by the following commands.
In our experience, models can be trained stably with roughly 30 data sets.
The teleoperation data is saved in the `multimodal_robot_model/teleop/teleop_data/MujocoUR5eCable` directory in npz format.
```console
$ cd multimodal_robot_model/teleop
$ # Connect a SpaceMouse 3D mouse to your PC
$ python bin/TeleopMujocoUR5eCable.py --world_idx_list 0 5
```

## Model Training
Train the ACT by the following commands.
```console
$ cd multimodal_robot_model/act
$ ln -sf `realpath ../teleop/teleop_data/MujocoUR5eCable` data/teleop_data_sample
$ python ../utils/make_dataset.py --in_dir ./data/teleop_data_sample --out_dir ./data/learning_data_sample --train_ratio 0.8 --nproc `nproc` --skip 3
$ python ./bin/train.py \
--dataset_dir ./data/learning_data_sample --ckpt_dir ./log/YEAR_DAY_TIME --task_name sim_ur5ecable \
--policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
--num_epochs 1000 --lr 1e-5 \
--seed 0
```
Note that the following error will occur if the chunk_size is larger than the time series length of the training data.
In such a case, either set the `--skip` option in `make_dataset.py` to a small value, or set the `--chunk_size` option in `train.py` to a small value.
```console
RuntimeError: The size of tensor a (70) must match the size of tensor b (102) at non-singleton dimension 0
```

## Policy rollout
Rollout the ACT in the simulation by the following commands.
```console
$ cd multimodal_robot_model/act
$ python ./bin/rollout/RolloutActMujocoUR5eCable.py \
--ckpt_dir ./log/YEAR_DAY_TIME --ckpt_name policy_last.ckpt \
--chunk_size 100 --seed 42 --skip 3 --world_idx 0
```
