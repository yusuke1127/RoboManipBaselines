from argparse import ArgumentParser
from pathlib import Path
from typing import Dict

import numpy as np
from numpy import ndarray


# argument options
parser = ArgumentParser()
parser.add_argument(
    "--data_dir",
    required=True,
    type=str,
    help="Target directory including .npz files",
)
# parser.add_argument(
#     "--skip_step",
#     type=int,
#     default=1,
#     help="Skips for downsampling time series",
# )
args = parser.parse_args()

# get .npz files
data_dir = Path(args.data_dir)
train_files = (data_dir / "train").glob("**/*.npz")
test_files = (data_dir / "test").glob("**/*.npz")
npz_files = list(train_files) + list(test_files)

# find the number of joints
npz_dict: Dict[str, ndarray]

npz_file = npz_files[0]
npz_dict = np.load(npz_file)
num_joints = npz_dict["joint"].shape[-1]

# initialize limits
joint_limits = np.empty((num_joints, 2), dtype=np.float32)
joint_limits[:, 0] = np.inf
joint_limits[:, 1] = -np.inf

wrench_limits = np.empty((num_joints - 1, 2), dtype=np.float32)
wrench_limits[:, 0] = np.inf
wrench_limits[:, 1] = -np.inf

# update limits from all of .npz files
for npz_file in npz_files:
    npz_dict = np.load(npz_file)

    # update joint limits
    joint = npz_dict["joint"]
    min_ = np.min(joint, axis=0)
    idx = min_ < joint_limits[:, 0]
    joint_limits[idx, 0] = min_[idx]
    max_ = np.max(joint, axis=0)
    idx = max_ > joint_limits[:, 1]
    joint_limits[idx, 1] = max_[idx]

    # update wrench limits
    wrench = npz_dict["wrench"]
    min_ = np.min(wrench, axis=0)
    idx = min_ < wrench_limits[:, 0]
    wrench_limits[idx, 0] = min_[idx]
    max_ = np.max(wrench, axis=0)
    idx = max_ > wrench_limits[:, 1]
    wrench_limits[idx, 1] = max_[idx]

print("Joint limits:")
for i in range(len(joint_limits)):
    print(
        f"  - Index {i} = "
        f"({joint_limits[i, 0]:.2f}, {joint_limits[i, 1]:.2f})"
    )

print("Wrench limits:")
for i in range(len(wrench_limits)):
    print(
        f"  - Index {i} = "
        f"({wrench_limits[i, 0]:.2f}, {wrench_limits[i, 1]:.2f})"
    )

np.save(data_dir / "joint_limits.npy", joint_limits)
np.save(data_dir / "wrench_limits.npy", wrench_limits)
