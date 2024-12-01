#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import torch
import torch.nn as nn
from torch.utils.data import Dataset

try:
    from torchvision.transforms import v2 as transforms
except ImportError:
    from torchvision import transforms


class RmbSarnnDatasetWithSideImageAndWrench(Dataset):
    """
    This class is used to train models that deal with multimodal data (e.g., front_images, side_images, joints, wrenches), such as CNNRNN/SARNN.

    Args:
        front images (numpy array): Set of front images in the dataset, expected to be a 5D array [data_num, seq_num, channel, height, width].
        side images (numpy array): Set of side images in the dataset, expected to be a 5D array [data_num, seq_num, channel, height, width].
        joints (numpy array): Set of joints in the dataset, expected to be a 3D array [data_num, seq_num, joint_dim].
        wrenches (numpy array): Set of wrenches in the dataset, expected to be a 3D array [data_num, seq_num, wrench_dim].
        stdev (float, optional): Set the standard deviation for normal distribution to generate noise.
    """

    def __init__(
        self, front_images, side_images, joints, wrenches, device="cpu", stdev=None
    ):
        """
        The constructor of Multimodal Dataset class. Initializes the front images, side images, joints, wrenches, and transformation.

        Args:
            front images (numpy array): The front images data, expected to be a 5D array [data_num, seq_num, channel, height, width].
            side images (numpy array): The side images data, expected to be a 5D array [data_num, seq_num, channel, height, width].
            joints (numpy array): The joints data, expected to be a 3D array [data_num, seq_num, joint_dim].
            wrenches (numpy array): The wrenches data, expected to be a 3D array [data_num, seq_num, wrench_dim].
            stdev (float, optional): The standard deviation for the normal distribution to generate noise. Defaults to 0.02.
        """
        self.stdev = stdev
        self.device = device
        self.front_images = torch.Tensor(front_images).to(self.device)
        self.side_images = torch.Tensor(side_images).to(self.device)
        self.joints = torch.Tensor(joints).to(self.device)
        self.wrenches = torch.Tensor(wrenches).to(self.device)
        self.transform = nn.Sequential(
            transforms.RandomErasing(),
            transforms.ColorJitter(brightness=0.4),
            transforms.ColorJitter(contrast=[0.6, 1.4]),
            transforms.ColorJitter(hue=[0.0, 0.04]),
            transforms.ColorJitter(saturation=[0.6, 1.4]),
        ).to(self.device)

    def __len__(self):
        """
        Returns the number of the data.
        """
        return len(self.front_images)

    def __getitem__(self, idx):
        """
        Extraction and preprocessing of front images, side images, joints and wrenches at the specified indexes.

        Args:
            idx (int): The index of the element.

        Returns:
            dataset (list): A list containing lists of transformed and noise added front image, side image, joint and wrench (x_front_img, x_side_img, x_joint, x_wrench) and the original front image, side image, joint and wrench (y_front_img, y_side_img, y_joint, y_wrench).
        """
        x_front_img = self.front_images[idx]
        x_side_img = self.side_images[idx]
        x_joint = self.joints[idx]
        x_wrench = self.wrenches[idx]
        y_front_img = self.front_images[idx]
        y_side_img = self.side_images[idx]
        y_joint = self.joints[idx]
        y_wrench = self.wrenches[idx]

        if self.stdev is not None:
            x_front_img = self.transform(y_front_img) + torch.normal(
                mean=0, std=0.02, size=x_front_img.shape, device=self.device
            )
            x_side_img = self.transform(y_side_img) + torch.normal(
                mean=0, std=0.02, size=x_side_img.shape, device=self.device
            )
            x_joint = y_joint + torch.normal(
                mean=0, std=self.stdev, size=y_joint.shape, device=self.device
            )
            x_wrench = y_wrench + torch.normal(
                mean=0, std=self.stdev, size=y_wrench.shape, device=self.device
            )

        return [
            [x_front_img, x_side_img, x_joint, x_wrench],
            [y_front_img, y_side_img, y_joint, y_wrench],
        ]
