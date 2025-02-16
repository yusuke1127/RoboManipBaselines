import argparse
import datetime
import glob
import os
import pickle
import random
import sys
from abc import ABCMeta, abstractmethod
from copy import deepcopy

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .DataKey import DataKey
from .DataUtils import get_skipped_data_seq
from .MathUtils import set_random_seed


class TrainBase(metaclass=ABCMeta):
    def __init__(self):
        self.setup_args()

        set_random_seed(self.args.seed)

        self.setup_dataset()

        self.setup_policy()

    def setup_args(self, parser=None, argv=None):
        if parser is None:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )

        parser.add_argument(
            "--dataset_dir",
            type=str,
            required=True,
            help="dataset directory",
        )
        parser.add_argument(
            "--checkpoint_dir",
            type=str,
            default=None,
            help="checkpoint directory",
        )

        parser.add_argument(
            "--state_keys",
            type=str,
            nargs="*",
            default=[DataKey.MEASURED_JOINT_POS],
            choices=DataKey.MEASURED_DATA_KEYS,
            help="state data keys",
        )
        parser.add_argument(
            "--action_keys",
            type=str,
            nargs="+",
            default=[DataKey.COMMAND_JOINT_POS],
            choices=DataKey.COMMAND_DATA_KEYS,
            help="action data keys",
        )
        parser.add_argument(
            "--camera_names",
            type=str,
            nargs="+",
            default=["front"],
            help="camera names",
        )

        parser.add_argument(
            "--train_ratio", type=float, default=0.8, help="ratio of train data"
        )
        parser.add_argument(
            "--val_ratio", type=float, default=None, help="ratio of validation data"
        )

        parser.add_argument(
            "--state_aug_std",
            type=float,
            default=0.0,
            help="Standard deviation of random noise added to state",
        )
        parser.add_argument(
            "--action_aug_std",
            type=float,
            default=0.0,
            help="Standard deviation of random noise added to action",
        )
        parser.add_argument(
            "--image_aug_std",
            type=float,
            default=0.0,
            help="Standard deviation of random noise added to images",
        )
        parser.add_argument(
            "--image_aug_color",
            action="store_true",
            help="Whether to enable color augmentation for images",
        )
        parser.add_argument(
            "--image_aug_affine",
            action="store_true",
            help="Whether to enable affine augmentation for images",
        )

        parser.add_argument(
            "--skip",
            type=int,
            default=3,
            help="skip interval of data sequence (set 1 for no skip)",
        )

        parser.add_argument("--seed", type=int, default=42, help="seed")

        if argv is None:
            argv = sys.argv
        self.args = parser.parse_args(argv[1:])

        # Set checkpoint directory if it is not specified
        if self.args.checkpoint_dir is None:
            dataset_dirname = os.path.basename(os.path.normpath(self.args.dataset_dir))
            checkpoint_dirname = "{}_{}_{:%Y%m%d_%H%M%S}".format(
                dataset_dirname, self.policy_name, datetime.datetime.now()
            )
            self.args.checkpoint_dir = os.path.normpath(
                os.path.join(self.policy_dir, "checkpoint", checkpoint_dirname)
            )

    def setup_dataset(self):
        # Get file list
        all_filenames = glob.glob(f"{self.args.dataset_dir}/**/*.hdf5", recursive=True)
        random.shuffle(all_filenames)
        train_num = max(
            int(np.clip(self.args.train_ratio, 0.0, 1.0) * len(all_filenames)), 1
        )
        if self.args.val_ratio is None:
            val_num = max(len(all_filenames) - train_num, 1)
        else:
            val_num = max(
                int(np.clip(self.args.val_ratio, 0.0, 1.0) * len(all_filenames)), 1
            )
        train_filenames = all_filenames[:train_num]
        val_filenames = all_filenames[-1 * val_num :]

        # Construct dataset stats
        self.model_meta_info = self.make_model_meta_info(all_filenames)

        # Construct dataloader
        self.train_dataloader = self.make_dataloader(train_filenames, shuffle=True)
        self.val_dataloader = self.make_dataloader(val_filenames, shuffle=False)

        # Setup tensorboard
        self.writer = SummaryWriter(self.args.checkpoint_dir)

        # Print dataset information
        print(
            f"[{self.__class__.__name__}] Load dataset from {self.args.dataset_dir}\n"
            f"  - train size: {len(self.train_dataloader.dataset)}, files: {len(train_filenames)}\n"
            f"  - val size: {len(self.val_dataloader.dataset)}, files: {len(val_filenames)}"
        )
        print(
            f"  - aug std state: {self.model_meta_info['state']['aug_std']}, action: {self.model_meta_info['action']['aug_std']}, image: {self.model_meta_info['image']['aug_std']}"
        )
        image_transforms_str = ""
        image_transform_list = self.train_dataloader.dataset.image_transforms.transforms
        for image_transform_idx, image_transform in enumerate(image_transform_list):
            image_transforms_str += f"<{image_transform.__class__.__name__}>"
            if image_transform_idx < len(image_transform_list) - 1:
                image_transforms_str += " -> "
        print(f"  - image transforms: {image_transforms_str}")

    @abstractmethod
    def setup_policy(self):
        pass

    def print_policy_info(self):
        print(
            f"[{self.__class__.__name__}] Construct policy.\n"
            f"  - state dim: {len(self.model_meta_info['state']['example'])}, action dim: {len(self.model_meta_info['action']['example'])}, camera num: {len(self.args.camera_names)}\n"
            f"  - state keys: {self.args.state_keys}\n"
            f"  - action keys: {self.args.action_keys}\n"
            f"  - camera names: {self.args.camera_names}\n"
            f"  - skip: {self.args.skip}"
        )

    def make_model_meta_info(self, all_filenames):
        # Load all state and action
        all_state = []
        all_action = []
        rgb_image_example = None
        depth_image_example = None
        for filename in all_filenames:
            with h5py.File(filename, "r") as h5file:
                if len(self.args.state_keys) == 0:
                    episode_len = h5file[DataKey.TIME][:: self.args.skip].shape[0]
                    state = np.zeros((episode_len, 0), dtype=np.float32)
                else:
                    state = np.concatenate(
                        [
                            get_skipped_data_seq(
                                h5file[state_key][()], state_key, self.args.skip
                            )
                            for state_key in self.args.state_keys
                        ],
                        axis=1,
                    )
                all_state.append(state)

                action = np.concatenate(
                    [
                        get_skipped_data_seq(
                            h5file[action_key][()], action_key, self.args.skip
                        )
                        for action_key in self.args.action_keys
                    ],
                    axis=1,
                )
                all_action.append(action)

                if rgb_image_example is None:
                    rgb_image_example = {
                        camera_name: h5file[DataKey.get_rgb_image_key(camera_name)][()]
                        for camera_name in self.args.camera_names
                    }
                if depth_image_example is None:
                    depth_image_example = {
                        camera_name: h5file[DataKey.get_depth_image_key(camera_name)][
                            ()
                        ]
                        for camera_name in self.args.camera_names
                    }
        all_state = np.concatenate(all_state, dtype=np.float32)
        all_action = np.concatenate(all_action, dtype=np.float32)

        return {
            "state": {
                "keys": self.args.state_keys,
                "mean": all_state.mean(axis=0),
                "std": np.clip(all_state.std(axis=0), 1e-3, 1e10),
                "aug_std": self.args.state_aug_std,
                "example": all_state[0],
            },
            "action": {
                "keys": self.args.action_keys,
                "mean": all_action.mean(axis=0),
                "std": np.clip(all_action.std(axis=0), 1e-3, 1e10),
                "aug_std": self.args.action_aug_std,
                "example": all_action[0],
            },
            "image": {
                "camera_names": self.args.camera_names,
                "aug_std": self.args.image_aug_std,
                "aug_color": self.args.image_aug_color,
                "aug_affine": self.args.image_aug_affine,
                "rgb_example": rgb_image_example,
                "depth_example": depth_image_example,
            },
            "data": {"skip": self.args.skip},
        }

    def make_dataloader(self, filenames, shuffle=True):
        dataset = self.DatasetClass(filenames, self.model_meta_info)

        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=4,
            prefetch_factor=4,
        )

        return dataloader

    def run(self):
        os.makedirs(self.args.checkpoint_dir, exist_ok=True)

        # Save model meta info
        model_meta_info_path = os.path.join(
            self.args.checkpoint_dir, "model_meta_info.pkl"
        )
        with open(model_meta_info_path, "wb") as f:
            pickle.dump(self.model_meta_info, f)
        print(
            f"[{self.__class__.__name__}] Save model meta info: {model_meta_info_path}"
        )

        # Train loop
        print(
            f"[{self.__class__.__name__}] Train with saving checkpoints: {self.args.checkpoint_dir}"
        )
        self.train_loop()

    @abstractmethod
    def train_loop(self):
        pass

    def detach_batch_result(self, batch_result):
        for k, v in batch_result.items():
            batch_result[k] = v.item()
        return batch_result

    def log_epoch_summary(self, batch_result_list, label, epoch):
        epoch_summary = {"epoch": epoch}
        for k in batch_result_list[0]:
            epoch_summary[k] = np.mean(
                [batch_result[k] for batch_result in batch_result_list]
            )
        for k, v in epoch_summary.items():
            self.writer.add_scalar(f"{k}/{label}", v, epoch)
        return epoch_summary

    def update_best_ckpt(self, best_ckpt_info, epoch_summary):
        if epoch_summary["loss"] < best_ckpt_info["loss"]:
            best_ckpt_info = {
                "epoch": epoch_summary["epoch"],
                "loss": epoch_summary["loss"],
                "state_dict": deepcopy(self.policy.state_dict()),
            }
        return best_ckpt_info

    def save_current_ckpt(self, ckpt_suffix):
        ckpt_path = os.path.join(
            self.args.checkpoint_dir, f"{self.policy_name}_{ckpt_suffix}.ckpt"
        )
        torch.save(self.policy.state_dict(), ckpt_path)

    def save_best_ckpt(self, best_ckpt_info):
        ckpt_path = os.path.join(
            self.args.checkpoint_dir, f"{self.policy_name}_best.ckpt"
        )
        torch.save(best_ckpt_info["state_dict"], ckpt_path)
        print(
            f"[{self.__class__.__name__}] Best val loss is {best_ckpt_info['loss']:.3f} at epoch {best_ckpt_info['epoch']}"
        )

    def close(self):
        self.writer.close()
