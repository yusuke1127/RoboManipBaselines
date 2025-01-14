#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import argparse
import os

import matplotlib.animation as anim
import matplotlib.pylab as plt
import numpy as np
from tqdm import tqdm


def normalization(data, indata_range, outdata_range):
    eps = 1e-6
    data = (data - indata_range[0]) / (indata_range[1] - indata_range[0] + eps)
    data = data * (outdata_range[1] - outdata_range[0]) + outdata_range[0]
    return data


parser = argparse.ArgumentParser()
parser.add_argument("--idx", type=int, default=0)
parser.add_argument("--in_dir", type=str, default="./data/")
parser.add_argument("--measured_joints", action="store_true")
args = parser.parse_args()

idx = int(args.idx)
print("in_dir: ", args.in_dir)
if args.measured_joints:
    joints = np.load(os.path.join(args.in_dir, "test/joints.npy"))
    joint_bounds = np.load(os.path.join(args.in_dir, "joint_bounds.npy"))
else:
    joints = np.load(os.path.join(args.in_dir, "test/actions.npy"))
    joint_bounds = np.load(os.path.join(args.in_dir, "action_bounds.npy"))
front_images = np.load(os.path.join(args.in_dir, "test/front_images.npy"))
side_images = np.load(os.path.join(args.in_dir, "test/side_images.npy"))
wrenches = np.load(os.path.join(args.in_dir, "test/wrenches.npy"))
N = front_images.shape[1]
assert N == side_images.shape[1]
assert N == wrenches.shape[1]
assert N == joints.shape[1]

# normalized joints
minmax = [0.1, 0.9]
norm_joints = normalization(joints, joint_bounds, minmax)

# print data information
print("load test data, index number is {}".format(idx))
print(
    "Joint: shape={}, min={:.3g}, max={:.3g}".format(
        joints.shape, joints.min(), joints.max()
    )
)
print(
    "Norm joint: shape={}, min={:.3g}, max={:.3g}".format(
        norm_joints.shape, norm_joints.min(), norm_joints.max()
    )
)

# plot images and normalized joints
fig, ax = plt.subplots(1, 3, figsize=(14, 5), dpi=60)
pbar = tqdm(total=joints.shape[1] + 1, desc=anim.FuncAnimation.__name__)


def anim_update(i):
    for j in range(3):
        ax[j].cla()

    # plot image
    ax[0].imshow(front_images[idx, i])
    ax[0].axis("off")
    ax[0].set_title("Image")

    # plot joint angle
    ax[1].set_ylim(-1.0, 2.0)
    ax[1].set_xlim(0, N)
    ax[1].plot(joints[idx], linestyle="dashed", c="k")

    for joint_idx in range(joints.shape[2]):
        ax[1].plot(np.arange(i + 1), joints[idx, : i + 1, joint_idx])
    ax[1].set_xlabel("Step")
    ax[1].set_title("Joint angles")

    # plot normalized joint angle
    ax[2].set_ylim(0.0, 1.0)
    ax[2].set_xlim(0, N)
    ax[2].plot(norm_joints[idx], linestyle="dashed", c="k")

    for joint_idx in range(joints.shape[2]):
        ax[2].plot(np.arange(i + 1), norm_joints[idx, : i + 1, joint_idx])
    ax[2].set_xlabel("Step")
    ax[2].set_title("Normalized joint angles")
    pbar.update(1)


ani = anim.FuncAnimation(fig, anim_update, interval=int(N / 10), frames=N)
save_file_name = "./output/check_data_{}.gif".format(idx)
ani.save(save_file_name)
pbar.close()
print("save file: ", save_file_name)
