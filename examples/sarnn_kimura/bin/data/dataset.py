#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from torchvision.transforms import v2 as transforms
except ImportError:
    from torchvision import transforms


class ImageDataset(Dataset):
    """
    This class is used to train models that deal only with images, such as autoencoders.
    Data augmentation is applied to the given image data by adding lightning, contrast, horizontal and vertical shift, and gaussian noise.

    Arguments:
        data (numpy.array): Set the data type (train/test). If the last three dimensions are HWC or CHW, `data` allows any number of dimensions.
        stdev (float, optional): Set the standard deviation for normal distribution to generate noise.
    """

    def __init__(self, data, device="cpu", stdev=None):
        """
        Reshapes and transforms the data.

        Arguments:
            data (numpy.array): The images data, expected to be a 5D array [data_num, seq_num, channel, height, width].
            stdev (float, optional): The standard deviation for the normal distribution to generate gaussian noise.
        """

        self.stdev = stdev
        self.device = device
        _image_flatten = data.reshape(((-1,) + data.shape[-3:]))
        self.image_flatten = torch.Tensor(_image_flatten).to(self.device)

        self.transform_affine = transforms.Compose(
            [
                transforms.RandomAffine(degrees=(0, 0), translate=(0.15, 0.15)),
                transforms.RandomAutocontrast(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        ).to(self.device)

        self.transform_noise = transforms.Compose(
            [
                transforms.ColorJitter(
                    contrast=[0.6, 1.4], brightness=0.4, saturation=[0.6, 1.4], hue=0.04
                )
            ]
        ).to(self.device)

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            length (int): The length of the dataset.
        """

        return len(self.image_flatten)

    def __getitem__(self, idx):
        """
        Extracts a single image from the dataset and returns two images: the original image and the image with noise added.

        Args:
            idx (int): The index of the element.

        Returns:
            image_list (list): A list containing the transformed and noise added image (x_img) and the affine transformed image (y_img).
        """
        img = self.image_flatten[idx]

        if self.stdev is not None:
            y_img = self.transform_affine(img)
            x_img = self.transform_noise(y_img) + torch.normal(
                mean=0, std=self.stdev, size=y_img.shape, device=self.device
            )
        else:
            y_img = img
            x_img = img

        return [x_img, y_img]


class MultimodalDataset(Dataset):
    """
    This class is used to train models that deal with multimodal data (e.g., front_images, side_images, joints, wrenches), such as CNNRNN/SARNN.

    Args:
        front images (numpy array): Set of front images in the dataset, expected to be a 5D array [data_num, seq_num, channel, height, width].
        side images (numpy array): Set of side images in the dataset, expected to be a 5D array [data_num, seq_num, channel, height, width].
        joints (numpy array): Set of joints in the dataset, expected to be a 3D array [data_num, seq_num, joint_dim].
        wrenches (numpy array): Set of wrenches in the dataset, expected to be a 3D array [data_num, seq_num, wrench_dim].
        stdev (float, optional): Set the standard deviation for normal distribution to generate noise.
    """

    def __init__(self, front_images, side_images, joints, wrenches, device="cpu", stdev=None):
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

        return [[x_front_img, x_side_img, x_joint, x_wrench], [y_front_img, y_side_img, y_joint, y_wrench]]


class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        if self.batch_sampler is None:
            self.sampler = _RepeatSampler(self.sampler)
        else:
            self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return (
            len(self.sampler)
            if self.batch_sampler is None
            else len(self.batch_sampler.sampler)
        )

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
