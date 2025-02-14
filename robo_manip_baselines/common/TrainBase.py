import argparse
import datetime
import glob
import os
import pickle
import random
import sys
from abc import ABCMeta, abstractmethod

import h5py
import numpy as np

from .DataKey import DataKey
from .DataUtils import get_skipped_data_seq
from .MathUtils import set_random_seed


class TrainBase(metaclass=ABCMeta):
    def __init__(self):
        self.setup_args()

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
            type=int,
            default=0.0,
            help="Standard deviation of random noise added to state",
        )
        parser.add_argument(
            "--action_aug_std",
            type=int,
            default=0.0,
            help="Standard deviation of random noise added to action",
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
                os.path.join(
                    os.path.dirname(__file__), "../checkpoint/", checkpoint_dirname
                )
            )

    def setup_dataset(self):
        set_random_seed(0)

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
        self.train_dataloader = self.make_dataloader(train_filenames)
        self.val_dataloader = self.make_dataloader(val_filenames)
        print(
            f"[TrainBase] Load dataset from {self.args.dataset_dir}\n"
            f"  - train episodes: {len(train_filenames)}, val episodes: {len(val_filenames)}"
        )

    @abstractmethod
    def setup_policy(self):
        pass

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
                "rgb_example": rgb_image_example,
                "depth_example": depth_image_example,
            },
            "data": {"skip": self.args.skip},
        }

    @abstractmethod
    def make_dataloader(self, filenames):
        pass

    def run(self):
        os.makedirs(self.args.checkpoint_dir, exist_ok=True)

        # Save model meta info
        model_meta_info_path = os.path.join(
            self.args.checkpoint_dir, "model_meta_info.pkl"
        )
        with open(model_meta_info_path, "wb") as f:
            pickle.dump(self.model_meta_info, f)
        print(f"[TrainBase] Save model meta info: {model_meta_info_path}")

        # Train loop
        print(f"[TrainBase] Train with saving checkpoints: {self.args.checkpoint_dir}")
        self.train_loop()

    @abstractmethod
    def train_loop(self):
        pass
