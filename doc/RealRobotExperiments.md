# Real robot experiments

## UR5e
https://github.com/user-attachments/assets/1b894ea0-6b5a-4bf8-abaa-8cd3ec2c4f85

### Preparation
Specify the IP address of UR5e in `robot_ip` and the serial number of RealSense in `camera_ids` in the following files:
- [TeleopRealUR5eDemo.py](../multimodal_robot_model/teleop/bin/TeleopRealUR5eDemo.py)
- [RolloutSarnnRealUR5eDemo.py](../multimodal_robot_model/sarnn/bin/rollout/RolloutSarnnRealUR5eDemo.py)
- [RolloutActRealUR5eDemo.py](../multimodal_robot_model/act/bin/rollout/RolloutActRealUR5eDemo.py)
- [RolloutDiffusionPolicyRealUR5eDemo.py](../multimodal_robot_model/diffusion_policy/bin/rollout/RolloutDiffusionPolicyRealUR5eDemo.py)

### Teleoperation
```console
# Demonstration in the 1st environment
$ cd multimodal_robot_model/teleop/
$ python ./bin/TeleopRealUR5eDemo.py --demo_name <demo_name> --world_idx_list 0

# Demonstration in the 6th environment
$ cd multimodal_robot_model/teleop/
$ python ./bin/TeleopRealUR5eDemo.py --demo_name <demo_name> --world_idx_list 5
```

### Policy rollout
```console
# SARNN
$ cd multimodal_robot_model/sarnn/
$ python ./bin/rollout/RolloutSarnnRealUR5eDemo.py --checkpoint <path_to_SARNN.pth> --cropped_img_size <cropped_img_size> --skip 6 --scale_dt 1 --world_idx 0 --wait_before_start --win_xy_policy 0 570

# ACT
$ cd multimodal_robot_model/act/
$ python ./bin/rollout/RolloutActRealUR5eDemo.py --ckpt_dir <ckpt_directory> --ckpt_name policy_last.ckpt --chunk_size 100 --seed 42 --skip 1 --scale_dt 1 --world_idx 0 --wait_before_start --win_xy_policy 0 570

# Diffusion policy
$ cd multimodal_robot_model/diffusion_policy/
$ python ./bin/rollout/RolloutDiffusionPolicyRealUR5eDemo.py --checkpoint <path_to_ckpt_file> --skip 1 --scale_dt 1 --world_idx 0 --wait_before_start --win_xy_policy 0 570
```
