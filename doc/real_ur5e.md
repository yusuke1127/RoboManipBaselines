# Experiments with real UR5e

https://github.com/user-attachments/assets/1b894ea0-6b5a-4bf8-abaa-8cd3ec2c4f85

## Install
See [here](./install.md#Real-UR5e-environments) for installation.

## Preparation
Specify the IP address of UR5e in `robot_ip` and the serial number of RealSense in `camera_ids` in the following files:
- [TeleopRealUR5eDemo.py](../robo_manip_baselines/teleop/bin/TeleopRealUR5eDemo.py)
- [RolloutSarnnRealUR5eDemo.py](../robo_manip_baselines/sarnn/bin/rollout/RolloutSarnnRealUR5eDemo.py)
- [RolloutActRealUR5eDemo.py](../robo_manip_baselines/act/bin/rollout/RolloutActRealUR5eDemo.py)
- [RolloutDiffusionPolicyRealUR5eDemo.py](../robo_manip_baselines/diffusion_policy/bin/rollout/RolloutDiffusionPolicyRealUR5eDemo.py)

## Teleoperation
```console
# Demonstration in the 1st environment
$ cd robo_manip_baselines/teleop/
$ python ./bin/TeleopRealUR5eDemo.py --demo_name <demo_name> --world_idx_list 0

# Demonstration in the 6th environment
$ cd robo_manip_baselines/teleop/
$ python ./bin/TeleopRealUR5eDemo.py --demo_name <demo_name> --world_idx_list 5
```

## Policy rollout
```console
# SARNN
$ cd robo_manip_baselines/sarnn/
$ python ./bin/rollout/RolloutSarnnRealUR5eDemo.py --checkpoint <path_to_SARNN.pth> --cropped_img_size <cropped_img_size> --skip 6 --scale_dt 1 --world_idx 0 --wait_before_start --win_xy_policy 0 570

# ACT
$ cd robo_manip_baselines/act/
$ python ./bin/rollout/RolloutActRealUR5eDemo.py --checkpoint <path_to_policy_last.ckpt> --skip 1 --scale_dt 1 --world_idx 0 --wait_before_start --win_xy_policy 0 570

# Diffusion policy
$ cd robo_manip_baselines/diffusion_policy/
$ python ./bin/rollout/RolloutDiffusionPolicyRealUR5eDemo.py --checkpoint <path_to_ckpt_file> --skip 1 --scale_dt 1 --world_idx 0 --wait_before_start --win_xy_policy 0 570
```
