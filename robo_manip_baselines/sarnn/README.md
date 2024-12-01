# Spatial attention recurrent neural network (SARNN)

## Install
See [here](../../doc/install.md#SARNN) for installation.

## Dataset preparation
Collect demonstration data by [teleoperation](../teleop).

Generate a `npy` format dataset for learning from teleoperation data:
```console
$ python ../utils/make_dataset.py \
--in_dir ../teleop/teleop_data/<demo_name> --out_dir ./data/<demo_name> \
--train_ratio 0.8 --nproc `nproc` --skip 6 --cropped_img_size 280 --resized_img_size 64
```
The `--cropped_img_size` option should be specified appropriately for each task.

Visualize the generated data (optional):
```console
$ python ../utils/check_data.py --in_dir ./data/<demo_name> --idx 0
```

## Model training
Train a model:
```console
$ python ./bin/TrainSarnn.py \
--data_dir ./data/<demo_name> --log_dir ./log/<demo_name> \
--no_side_image --no_wrench --with_mask
```
The checkpoint file `SARNN.pth` is saved in the directory specified by the `--log_dir` option.

Visualize an animation of prediction (optional):
```console
$ python ./bin/test.py --data_dir ./data/<demo_name> --filename ./log/<demo_name>/SARNN.pth --no_side_image --no_wrench
```

Visualize the internal representation of the RNN in prediction (optional):
```console
$ python ./bin/test_pca.py --data_dir ./data/<demo_name> --filename ./log/<demo_name>/SARNN.pth --no_side_image --no_wrench
```

## Policy rollout
Run a trained policy:
```console
$ python ./bin/rollout/RolloutSarnnMujocoUR5eCable.py \
--checkpoint ./log/<demo_name>/SARNN.pth \
--cropped_img_size 280 --skip 6 --world_idx 0
```
The `--cropped_img_size` option must be the same as for dataset generation.
