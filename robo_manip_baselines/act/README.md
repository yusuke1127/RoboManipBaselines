# Action Chunking with Transformers (ACT)

## Install
See [here](../../doc/install.md#ACT) for installation.

## Dataset preparation
Collect demonstration data by [teleoperation](../teleop).

## Model training
Train a model:
```console
$ python ./bin/TrainAct.py --dataset_dir ../teleop/teleop_data/<name> --checkpoint_dir ./checkpoint/<name>
```

**Note**: The following error will occur if the chunk_size is larger than the time series length of the training data.
In such a case, either set the `--skip` option to a small value, or set the `--chunk_size` option to a small value.
```console
RuntimeError: The size of tensor a (70) must match the size of tensor b (102) at non-singleton dimension 0
```

The `--state_keys`, `--action_keys`, and `--camera_names` options allow changing the policy input and output.

Examples of `--state_keys`:
```console
# Set measured joint position as observed state (default)
--state_keys measured_joint_pos

# Leave observed state empty
--state_keys # no arguments
```

Examples of `--action_keys`:
```console
# Set command joint position as action (default)
--action_keys command_joint_pos

# Set command joint position relative to previous step as action
--action_keys command_joint_pos_rel

# Set command end-effector pose as action
--action_keys command_eef_pose command_gripper_joint_pos

# Set command end-effector pose relative to previous step as action
--action_keys command_eef_pose_rel command_gripper_joint_pos
```

## Policy rollout
Run a trained policy:
```console
$ python ./bin/rollout/RolloutActMujocoUR5eCable.py --checkpoint ./checkpoint/<name>/policy_last.ckpt --world_idx 0
```

## Technical Details
For more information on the technical details, please see the following paper:
```bib
@INPROCEEDINGS{ACT_RSS23,
  author = {Tony Z. Zhao and Vikash Kumar and Sergey Levine and Chelsea Finn},
  title = {Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware},
  booktitle = {Proceedings of Robotics: Science and Systems},
  year = {2023},
  month = {July},
  doi = {10.15607/RSS.2023.XIX.016}
}
```
