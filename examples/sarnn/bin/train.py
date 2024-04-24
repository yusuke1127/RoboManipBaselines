#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import os
import sys
import torch
import numpy as np
import argparse
from pathlib import Path

from tqdm import tqdm
import torch.optim as optim
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from eipl.model import SARNN
from eipl.data import MultimodalDataset, SampleDownloader
from eipl.utils import EarlyStopping, check_args, set_logdir, resize_img

# load own library
from libs.dataset import UR5eCableEnvDataset
from libs.fullBPTT import fullBPTTtrainer

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
parser.add_argument("--img_loss", type=float, default=0.1)
parser.add_argument("--joint_loss", type=float, default=1.0)
parser.add_argument("--pt_loss", type=float, default=0.1)
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

# calculate the noise level (variance) from the normalized range
stdev = args.stdev * (args.vmax - args.vmin)

# set device id
device = torch.device(
    f"cuda:{args.device}"
    if torch.cuda.is_available() and args.device >= 0
    else "cpu"
)

# load dataset
data_dir = Path(args.data_dir)
minmax = [args.vmin, args.vmax]

joint_limits = np.load(data_dir / "joint_limits.npy")
wrench_limits = np.load(data_dir / "wrench_limits.npy")

train_data_dir = data_dir / "train"
train_data_files = sorted(train_data_dir.glob("**/*.npz"))
train_dataset = UR5eCableEnvDataset(
    train_data_files,
    joint_limits,
    wrench_limits,
    stdev=stdev,
    skip=5,
)
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=False,
)

test_data_dir = data_dir / "test"
test_data_files = sorted(test_data_dir.glob("**/*.npz"))
test_dataset = UR5eCableEnvDataset(
    test_data_files,
    joint_limits,
    wrench_limits,
    stdev=stdev,
    skip=5,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=False,
)

# define model
joint_dim = train_dataset.joint.size(-1)
model = SARNN(
    rec_dim=args.rec_dim,
    joint_dim=joint_dim,
    k_dim=args.k_dim,
    heatmap_size=args.heatmap_size,
    temperature=args.temperature,
)

# torch.compile makes PyTorch code run faster
if args.compile:
    torch.set_float32_matmul_precision("high")
    model = torch.compile(model)

# set optimizer
optimizer = optim.Adam(model.parameters(), eps=1e-07)

# load trainer/tester class
loss_weights = [args.img_loss, args.joint_loss, args.pt_loss]
trainer = fullBPTTtrainer(
    model, optimizer, loss_weights=loss_weights, device=device
)

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
