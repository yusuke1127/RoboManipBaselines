#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import os
import sys
sys.path.append("../../third_party/eipl/")
import random
import torch
import numpy as np
import argparse
from pathlib import Path

from tqdm import tqdm
import torch.optim as optim
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from eipl.utils import EarlyStopping, check_args, set_logdir, normalization

# argument parser
parser = argparse.ArgumentParser(
    description="Learning spatial autoencoder with recurrent neural network"
)
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--model", type=str, default="sarnn")
parser.add_argument("--epoch", type=int, default=10000)
parser.add_argument("--batch_size", type=int, default=5)
parser.add_argument("--rec_dim", type=int, default=50)
parser.add_argument("--k_dim", type=int, default=5)
parser.add_argument("--random_seed", type=int, required=False)
parser.add_argument("--front_img_loss", type=float, default=0.1)
parser.add_argument("--side_img_loss", type=float, default=0.1)
parser.add_argument("--joint_loss", type=float, default=1.0)
parser.add_argument("--wrench_loss", type=float, default=1.0)
parser.add_argument("--front_pt_loss", type=float, default=0.1)
parser.add_argument("--side_pt_loss", type=float, default=0.1)
parser.add_argument("--no_side_image", action="store_true")
parser.add_argument("--no_wrench", action="store_true")
parser.add_argument("--with_mask", action="store_true")
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--heatmap_size", type=float, default=0.1)
parser.add_argument("--temperature", type=float, default=1e-4)
parser.add_argument("--stdev", type=float, default=0.1)
parser.add_argument("--log_dir", default="log/")
parser.add_argument("--vmin", type=float, default=0.0)
parser.add_argument("--vmax", type=float, default=1.0)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--compile", action="store_true")
parser.add_argument("--tag", help="Tag name for snap/log sub directory")
args = parser.parse_args()

# check args
args = check_args(args)

# fix seed
if args.random_seed is not None:
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# calculate the noise level (variance) from the normalized range
stdev = args.stdev * (args.vmax - args.vmin)

# set device id
if args.device >= 0:
    device = "cuda:{}".format(args.device)
else:
    device = "cpu"

# load dataset
data_dir = Path(args.data_dir)
minmax = [args.vmin, args.vmax]

train_data_dir = data_dir / "train"
joint_bounds = np.load(data_dir / "joint_bounds.npy")
joints_raw = np.load(sorted(train_data_dir.glob("**/joints.npy"))[0])
joints = normalization(joints_raw, joint_bounds, minmax)
if not args.no_wrench:
    wrench_bounds = np.load(data_dir / "wrench_bounds.npy")
    wrenches_raw = np.load(sorted(train_data_dir.glob("**/wrenches.npy"))[0])
    wrenches = normalization(wrenches_raw, wrench_bounds, minmax)
front_images_raw = np.load(sorted(train_data_dir.glob("**/front_images.npy"))[0])
front_images = normalization(front_images_raw.transpose(0, 1, 4, 2, 3), (0, 255), minmax)
if not args.no_side_image:
    side_images_raw = np.load(sorted(train_data_dir.glob("**/side_images.npy"))[0])
    side_images = normalization(side_images_raw.transpose(0, 1, 4, 2, 3), (0, 255), minmax)
masks = np.load(sorted(train_data_dir.glob("**/masks.npy"))[0])
if (not args.no_side_image) and (not args.no_wrench):
    assert not args.with_mask, "with_mask option is not supported for the model with side_image and wrench."
    from multimodal_robot_model.sarnn import MultimodalDatasetWithSideimageAndWrench
    train_dataset = MultimodalDatasetWithSideimageAndWrench(
        front_images,
        side_images,
        joints,
        wrenches,
        device=device,
        stdev=stdev
    )
elif args.no_side_image and args.no_wrench:
    if args.with_mask:
        from multimodal_robot_model.sarnn import MultimodalDatasetWithMask
        train_dataset = MultimodalDatasetWithMask(
            front_images,
            joints,
            masks,
            device=device,
            stdev=stdev
        )
    else:
        from eipl.data import MultimodalDataset
        train_dataset = MultimodalDataset(
            front_images,
            joints,
            device=device,
            stdev=stdev
        )
else:
    raise AssertionError(f"Not asserted (no_side_image, no_wrench): {(args.no_side_image, args.no_wrench)}")
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=False,
)

test_data_dir = data_dir / "test"
joints_raw = np.load(sorted(test_data_dir.glob("**/joints.npy"))[0])
joints = normalization(joints_raw, joint_bounds, minmax)
if not args.no_wrench:
    wrenches_raw = np.load(sorted(test_data_dir.glob("**/wrenches.npy"))[0])
    wrenches = normalization(wrenches_raw, wrench_bounds, minmax)
front_images_raw = np.load(sorted(test_data_dir.glob("**/front_images.npy"))[0])
front_images = normalization(front_images_raw.transpose(0, 1, 4, 2, 3), (0, 255), minmax)
if not args.no_side_image:
    side_images_raw = np.load(sorted(test_data_dir.glob("**/side_images.npy"))[0])
    side_images = normalization(side_images_raw.transpose(0, 1, 4, 2, 3), (0, 255), minmax)
masks = np.load(sorted(test_data_dir.glob("**/masks.npy"))[0])
if (not args.no_side_image) and (not args.no_wrench):
    test_dataset = MultimodalDatasetWithSideimageAndWrench(
        front_images,
        side_images,
        joints,
        wrenches,
        device=device,
        stdev=None
    )
elif args.no_side_image and args.no_wrench:
    if args.with_mask:
        from multimodal_robot_model.sarnn import MultimodalDatasetWithMask
        test_dataset = MultimodalDatasetWithMask(
            front_images,
            joints,
            masks,
            device=device,
            stdev=stdev
        )
    else:
        from eipl.data import MultimodalDataset
        test_dataset = MultimodalDataset(
            front_images,
            joints,
            device=device,
            stdev=None
        )
else:
    raise AssertionError(f"Not asserted (no_side_image, no_wrench): {(args.no_side_image, args.no_wrench)}")
test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=False,
)

# define model
if (not args.no_side_image) and (not args.no_wrench):
    from multimodal_robot_model.sarnn import SARNNwithSideimageAndWrench
    model = SARNNwithSideimageAndWrench(
        rec_dim=args.rec_dim,
        joint_dim=7,
        wrench_dim=6,
        k_dim=args.k_dim,
        heatmap_size=args.heatmap_size,
        temperature=args.temperature,
        im_size=[64, 64],
    )
elif args.no_side_image and args.no_wrench:
    from eipl.model import SARNN
    model = SARNN(
        rec_dim=args.rec_dim,
        joint_dim=7,
        k_dim=args.k_dim,
        heatmap_size=args.heatmap_size,
        temperature=args.temperature,
        im_size=[64, 64],
    )
else:
    raise AssertionError(f"Not asserted (no_side_image, no_wrench): {(args.no_side_image, args.no_wrench)}")

# torch.compile makes PyTorch code run faster
if args.compile:
    torch.set_float32_matmul_precision("high")
    model = torch.compile(model)

# set optimizer
optimizer = optim.Adam(model.parameters(), eps=1e-07, lr=args.lr)

# load trainer/tester class
if (not args.no_side_image) and (not args.no_wrench):
    from multimodal_robot_model.sarnn import fullBPTTtrainerWithSideimageAndWrench, Loss
    loss_weights = [
        {
            Loss.FRONT_IMG: args.front_img_loss,
            Loss.SIDE_IMG: args.side_img_loss,
            Loss.JOINT: args.joint_loss,
            Loss.WRENCH: args.wrench_loss,
            Loss.FRONT_PT: args.front_pt_loss,
            Loss.SIDE_PT: args.side_pt_loss
        }[loss] for loss in Loss
    ]
    trainer = fullBPTTtrainerWithSideimageAndWrench(
        model, optimizer, loss_weights=loss_weights, device=device
    )
elif args.no_side_image and args.no_wrench:
    loss_weights = [args.front_img_loss, args.joint_loss, args.front_pt_loss]
    if args.with_mask:
        from multimodal_robot_model.sarnn import fullBPTTtrainerWithMask
        trainer = fullBPTTtrainerWithMask(model, optimizer, loss_weights=loss_weights, device=device)
    else:
        from eipl.tutorials.airec.sarnn.libs.fullBPTT import fullBPTTtrainer
        trainer = fullBPTTtrainer(model, optimizer, loss_weights=loss_weights, device=device)
else:
    raise AssertionError(f"Not asserted (no_side_image, no_wrench): {(args.no_side_image, args.no_wrench)}")

### training main
log_dir_path = set_logdir("./" + args.log_dir, args.tag)
save_name = os.path.join(log_dir_path, "SARNN.pth")
writer = SummaryWriter(log_dir=log_dir_path, flush_secs=30)
early_stop = EarlyStopping(patience=1000)

with tqdm(range(args.epoch)) as pbar_epoch:
    for epoch in pbar_epoch:
        # train and test
        train_loss = trainer.process_epoch(train_loader)
        with torch.no_grad():
            test_loss = trainer.process_epoch(test_loader, training=False)
        writer.add_scalar("Loss/train_loss", train_loss, epoch)
        writer.add_scalar("Loss/test_loss", test_loss, epoch)

        # early stop
        save_ckpt, _ = early_stop(test_loss)

        if save_ckpt:
            trainer.save(epoch, [train_loss, test_loss], save_name)

        # print process bar
        pbar_epoch.set_postfix(
            OrderedDict(train_loss=train_loss, test_loss=test_loss)
        )
        pbar_epoch.update()
