# Spatial attention recurrent neural network (SARNN)

## Requirements

Install [EIPL](https://github.com/ogata-lab/eipl).
```console
$ # Go to the top directory of this repository
$ git submodule update --init --recursive
$ cd third_party/eipl
$ pip install -r requirements.txt
$ pip install -e .
```

Install [MultimodalRobotModel](https://github.com/isri-aist/MultimodalRobotModel) (if you only want model training, `pinocchio` is not required).
```console
$ # Go to the top directory of this repository
$ pip install -e .
```

## Dataset preparation

Put your data collected under `data` directory. Here, we assume the name of your dataset directory as `teleop_data_00000000`. 

```console
$ tree data/teleop_data_00000000/
data/teleop_data_00000000/
├── env0
│   ├── UR5eCableEnv_env0_000.npz
│   └── UR5eCableEnv_env0_006.npz
├── env1
│   ├── UR5eCableEnv_env1_001.npz
│   └── UR5eCableEnv_env1_007.npz
├── env2
│   ├── UR5eCableEnv_env2_002.npz
│   └── UR5eCableEnv_env2_008.npz
├── env3
│   ├── UR5eCableEnv_env3_003.npz
│   └── UR5eCableEnv_env3_009.npz
├── env4
│   ├── UR5eCableEnv_env4_004.npz
│   └── UR5eCableEnv_env4_010.npz
└── env5
    ├── UR5eCableEnv_env5_005.npz
    └── UR5eCableEnv_env5_011.npz
```

Run `./bin/2_make_dataset.py` to make NPZ files in each of `train` (for training) and `test` directories (for validation), in `teleop_data_00000000`.

```console
$ python ./bin/2_make_dataset.py --in_dir ./data/teleop_data_00000000 --nproc `nproc` --cropped_img_size 128
```

## Model Training

Run `./bin/train.py` to start training the model. The trained weights are saved in the log folder.

```console
$ python ./bin/train.py --data_dir ./data/ --no_side_image --no_wrench --with_mask
```

## Test

Specifying a weight file as the argument of `./bin/test.py` will save a gif animation of the predicted image, attention points, and predicted joint angles in the output folder.

```console
$ python ./bin/test.py --data_dir ./data/ --filename ./log/YEAR_DAY_TIME/SARNN.pth --no_side_image --no_wrench
```

## Visualization of internal representation using PCA

Specifying a weight file as the argument of `./bin/test_pca_sarnn.py` will save the internal representation of the RNN as a gif animation.

```console
$ python ./bin/test_pca_sarnn.py --data_dir ./data/ --filename ./log/YEAR_DAY_TIME/SARNN.pth --no_side_image --no_wrench
```

## Run a trained policy on the simulation

```console
$ python ./bin/Demo_UR5eCableEnv_RolloutPolicy.py --data_dir ./data/ --filename ./log/YEAR_DAY_TIME/SARNN.pth --pole-pos-idx 1
```
