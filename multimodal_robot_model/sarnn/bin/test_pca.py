#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import os
import sys
from tqdm import tqdm
import torch
import argparse
import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as anim
from sklearn.decomposition import PCA
from eipl.utils import restore_args, tensor2numpy, normalization


# argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--filename",
    type=str,
    required=True,
    help=".pth file that PyTorch loads as checkpoint for model",
)
parser.add_argument(
    "--data_dir",
    type=str,
    required=True,
    help="directory that stores test data, that has been generated, and will be loaded",
)
parser.add_argument("--no_side_image", action="store_true")
parser.add_argument("--no_wrench", action="store_true")
args = parser.parse_args()

# restore parameters
dir_name = os.path.split(args.filename)[0]
params = restore_args(os.path.join(dir_name, "args.json"))
# idx = args.idx

# load dataset
minmax = [params["vmin"], params["vmax"]]
front_images = np.load(os.path.join(args.data_dir, "test/front_images.npy"))
if not args.no_side_image:
    side_images = np.load(os.path.join(args.data_dir, "test/side_images.npy"))
joints = np.load(os.path.join(args.data_dir, "test/joints.npy"))
joint_bounds = np.load(os.path.join(args.data_dir, "joint_bounds.npy"))
if not args.no_wrench:
    wrenches = np.load(os.path.join(args.data_dir, "test/wrenches.npy"))
    wrench_bounds = np.load(os.path.join(args.data_dir, "wrench_bounds.npy"))

# define model
joint_dim = joints.shape[-1]
if (not args.no_side_image) and (not args.no_wrench):
    from multimodal_robot_model.sarnn import SARNNwithSideimageAndWrench

    model = SARNNwithSideimageAndWrench(
        rec_dim=params["rec_dim"],
        joint_dim=joint_dim,
        wrench_dim=6,
        k_dim=params["k_dim"],
        heatmap_size=params["heatmap_size"],
        temperature=params["temperature"],
        im_size=[64, 64],
    )
elif args.no_side_image and args.no_wrench:
    from eipl.model import SARNN

    model = SARNN(
        rec_dim=params["rec_dim"],
        joint_dim=joint_dim,
        k_dim=params["k_dim"],
        heatmap_size=params["heatmap_size"],
        temperature=params["temperature"],
        im_size=[64, 64],
    )
else:
    raise AssertionError(
        f"Not asserted (no_side_image, no_wrench): {(args.no_side_image, args.no_wrench)}"
    )

if params["compile"]:
    model = torch.compile(model)

# load weight
ckpt = torch.load(args.filename, map_location=torch.device("cpu"))
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Inference
states = []
state = None
nloop = front_images.shape[1]
for loop_ct in range(nloop):
    # load data and normalization
    front_img_t = front_images[:, loop_ct].transpose(0, 3, 1, 2)
    front_img_t = normalization(front_img_t, (0, 255), minmax)
    front_img_t = torch.Tensor(front_img_t)
    if not args.no_side_image:
        side_img_t = side_images[:, loop_ct].transpose(0, 3, 1, 2)
        side_img_t = normalization(side_img_t, (0, 255), minmax)
        side_img_t = torch.Tensor(side_img_t)
    joint_t = normalization(joints[:, loop_ct], joint_bounds, minmax)
    joint_t = torch.Tensor(joint_t)
    if not args.no_wrench:
        wrench_t = normalization(wrenches[:, loop_ct], wrench_bounds, minmax)
        wrench_t = torch.Tensor(wrench_t)

    # predict rnn
    if (not args.no_side_image) and (not args.no_wrench):
        _, _, _, _, _, _, _, _, state = model(
            front_img_t, side_img_t, joint_t, wrench_t, state
        )
    elif args.no_side_image and args.no_wrench:
        _, _, _, _, state = model(front_img_t, joint_t, state)
    else:
        raise AssertionError(
            f"Not asserted (no_side_image, no_wrench): {(args.no_side_image, args.no_wrench)}"
        )
    states.append(state[0])

states = torch.permute(torch.stack(states), (1, 0, 2))
states = tensor2numpy(states)
# Reshape the state from [N,T,D] to [-1,D] for PCA of RNN.
# N is the number of datasets
# T is the sequence length
# D is the dimension of the hidden state
N, T, D = states.shape
states = states.reshape(-1, D)

# PCA
loop_ct = float(360) / T
pca_dim = 3
pca = PCA(n_components=pca_dim).fit(states)
pca_val = pca.transform(states)
# Reshape the states from [-1, pca_dim] to [N,T,pca_dim] to
# visualize each state as a 3D scatter.
pca_val = pca_val.reshape(N, T, pca_dim)

# plot front_images
fig = plt.figure(dpi=120)
ax = fig.add_subplot(projection="3d")
pbar = tqdm(total=pca_val.shape[1] + 1, desc=anim.FuncAnimation.__name__)


def anim_update(i):
    ax.cla()
    angle = int(loop_ct * i)
    ax.view_init(30, angle)

    c_list = ["C0", "C1", "C2", "C3", "C4", "C5"]
    for n, color in enumerate(c_list):
        ax.scatter(
            pca_val[n, 1:, 0], pca_val[n, 1:, 1], pca_val[n, 1:, 2], color=color, s=3.0
        )

    ax.scatter(pca_val[n, 0, 0], pca_val[n, 0, 1], pca_val[n, 0, 2], color="k", s=30.0)
    pca_ratio = pca.explained_variance_ratio_ * 100
    ax.set_xlabel("PC1 ({:.1f}%)".format(pca_ratio[0]))
    ax.set_ylabel("PC2 ({:.1f}%)".format(pca_ratio[1]))
    ax.set_zlabel("PC3 ({:.1f}%)".format(pca_ratio[2]))
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="z", labelsize=8)
    pbar.update(1)


ani = anim.FuncAnimation(fig, anim_update, interval=int(np.ceil(T / 10)), frames=T)
save_file_name = "./output/PCA_SARNN_{}.gif".format(params["tag"])
ani.save(save_file_name)
pbar.close()
print("save file: ", save_file_name)

# If an error occurs in generating the gif or mp4 animation, change the writer (front_imagemagick/ffmpeg).
# ani.save("./output/PCA_SARNN_{}.gif".format(params["tag"]), writer="front_imagemagick")
# ani.save("./output/PCA_SARNN_{}.mp4".format(params["tag"]), writer="ffmpeg")
