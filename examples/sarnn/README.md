# Spatial attention recurrent neural network (SARNN) on UR5eCableEnv

## Requirements: installing EIPL

Install [EIPL](https://github.com/ogata-lab/eipl) according to `third_party/eipl/README.md`.
```console
$ # clone submodule resources
$ cd MultimodalRobotModel
$ git submodule update --init --recursive

$ # install EIPL
$ cd third_party/eipl
$ pip install -r requirements.txt
$ pip install -e .
```

## Dataset preparation

Put your data collected on UR5eCableEnv under `data` directory. Here, we assume the name of your dataset directory as  `your_dataset`. In `your_dataset`, make `train` directory for training and `test` directory for validation. Then, place NPZ files in each of `train` and `test` directories.

In this example, all NPZ files put under `train` and `test` directories are loaded.

```console
$ tree data
data
└── your_dataset
    ├── train
    │   ├── env0
    │   │   └── train_data_env0.npz
    │   ├── train_data_a.npz
    │   └── train_data_b.npz
    └── test
        ├── env1
        │   └── test_data_env1.npz
        └── test_data.npz
```

For instance, in the folder structure above, the NPZ files to be trained are `train_data_a.npz`, `train_data_b.npz` and `env0/train_data_env0.npz`. The NPZ files to be validated are `test_data.npz` and `env1/test_data_env1.npz`.

### Preprocessing

Create limitation files to normalize the collected joint and wrench data.

Run `bin/preprocess_data.py` to examine the maximum and minimum values of joint and wrench from all of the NPZ files under `train` and `test` directories.

The limitation files are saved as `joint_limits.npy` and `wrench_limits.npy`

```console
$ # --data_dir: directory of your dataset
$ python bin/preprocess_data.py --data_dir ./data/your_dataset
```

After the preprocessing, the dataset directory will consist of the following:

```console
$ tree data
data
└── your_dataset
    ├── train
    │   ├── env0
    │   │   └── train_data_env0.npz
    │   ├── train_data_a.npz
    │   └── train_data_b.npz
    ├── test
    │   ├── env1
    │   │   └── test_data_env1.npz
    │   └── test_data.npz
    ├── joint_limits.npy
    └── wrench_limits.npy
```

## Training SARNN

```console
$ # --data_dir option is necessary.
$ python bin/train.py --data_dir ./data/your_dataset
```

The all options can be displayed with `--help`.

## Validating SARNN
Under construction
