from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from numpy import ndarray
from torch import Tensor
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from eipl.utils import normalization


STATE_KEYS = (
    "joint",
    "wrench",
    "front_image",
    "side_image",
)

cable_env_data_t = Tuple[
    Tuple[Tensor, Tensor],  # input: (image, joint)
    Tuple[Tensor, Tensor],  # target: (image, joint)
    Tensor,  # mask
]

resize_to_64x64 = transforms.Resize((64, 64))


class UR5eCableEnvDataset(Dataset):
    """"""

    def __init__(
        self,
        npz_files: List[Path],
        joint_limits: ndarray,
        wrench_limits: ndarray,
        stdev: float = None,
        skip: int = 1,
    ) -> None:

        self.stdev = stdev
        self.transform = transforms.Compose(
            [
                transforms.RandomErasing(),
                transforms.ColorJitter(brightness=0.4),
                transforms.ColorJitter(contrast=[0.6, 1.4]),
                transforms.ColorJitter(hue=[0.0, 0.04]),
                transforms.ColorJitter(saturation=[0.6, 1.4]),
            ]
        )

        # dictionary of:
        #   - key: each of STATE_KEYS
        #   - value: list of pre-aligned time series
        sequence_dict: Dict[str, List[Tensor]] = {}
        for key in STATE_KEYS:
            sequence_dict[key] = [Tensor()] * len(npz_files)

        # pre-aligned masks of time series
        mask_list = [Tensor()] * len(npz_files)

        # load npz files
        for i, npz_file in enumerate(tqdm(npz_files, desc="Load NPZ")):
            npz_dict = np.load(npz_file)
            length = int(np.ceil(len(npz_dict["time"]) / skip))

            for key in STATE_KEYS:
                array = np.array(npz_dict[key][::skip], dtype=np.float32)

                # normalize data by boundaries
                if key == "joint":
                    array = normalization(array, joint_limits.T, [-1.0, 1.0])
                elif key == "wrench":
                    array = normalization(array, wrench_limits.T, [-1.0, 1.0])
                else:  # ("front_image", "side_image")
                    array = normalization(array, [0.0, 255.0], [0.0, 1.0])

                sequence_dict[key][i] = torch.tensor(array)

            # binary mask for handling valid lengths of time series
            mask_list[i] = torch.ones((length,), dtype=torch.bool)

        # `torch.nn.utils.rnn.pad_sequence` aligns the time series with
        # variable time lengths.
        self.joint = pad_sequence(sequence_dict["joint"], batch_first=True)
        self.wrench = pad_sequence(sequence_dict["wrench"], batch_first=True)
        front_image = pad_sequence(
            sequence_dict["front_image"], batch_first=True
        )
        side_image = pad_sequence(
            sequence_dict["side_image"], batch_first=True
        )
        # permute images: (N, T, H, W, C) -> (N, T, C, H, W)
        self.front_image = resize_to_64x64(
            torch.permute(front_image, (0, 1, 4, 2, 3))
        )
        self.side_image = resize_to_64x64(
            torch.permute(side_image, (0, 1, 4, 2, 3))
        )
        # Values after finishing time series are filled as zero.
        self.mask = pad_sequence(mask_list, batch_first=True)

    def __len__(self) -> int:
        return len(self.joint)

    def __getitem__(self, index) -> cable_env_data_t:
        warnings.warn(
            "UR5eCableEnvDataset handles only joint, front_image, and mask."
        )
        y_front_img = self.front_image[index]
        y_joint = self.joint[index]

        x_front_img = (
            self.transform(y_front_img)
            + torch.randn_like(y_front_img) * self.stdev
        )
        x_joint = y_joint + torch.randn_like(y_joint) * self.stdev

        mask = self.mask[index]

        return (x_front_img, x_joint), (y_front_img, y_joint), mask

    def num_valid_elements(self) -> Tensor:
        return self.mask.sum()

    def time_length(self, index=None) -> Tensor:
        if index is None:
            return self.mask.sum(dim=1)
        else:
            return self.mask[index].sum()
