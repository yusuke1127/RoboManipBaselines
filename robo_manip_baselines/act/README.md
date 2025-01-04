# Action Chunking with Transformers (ACT)

## Install
See [here](../../doc/install.md#ACT) for installation.

## Dataset preparation
Collect demonstration data by [teleoperation](../teleop).

Generate a `npy` format dataset for learning from teleoperation data:
```console
$ python ../utils/make_dataset.py \
--in_dir ../teleop/teleop_data/<demo_name> --out_dir ./data/<demo_name> \
--train_ratio 0.8 --nproc `nproc` --skip 3
```

Visualize the generated data (optional):
```console
$ python ../utils/check_data.py --in_dir ./data/<demo_name> --idx 0
```

## Model training
Train a model:
```console
$ python ./bin/TrainAct.py --dataset_dir ./data/<demo_name> --log_dir ./log/<demo_name>
```
The checkpoint file `SARNN.pth` is saved in the directory specified by the `--log_dir` option.

**Note**: The following error will occur if the chunk_size is larger than the time series length of the training data.
In such a case, either set the `--skip` option in `make_dataset.py` to a small value, or set the `--chunk_size` option in `TrainAct.py` to a small value.
```console
RuntimeError: The size of tensor a (70) must match the size of tensor b (102) at non-singleton dimension 0
```

## Policy rollout
Run a trained policy:
```console
$ python ./bin/rollout/RolloutActMujocoUR5eCable.py \
--checkpoint ./log/<demo_name>/policy_last.ckpt \
--skip 3 --world_idx 0
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
