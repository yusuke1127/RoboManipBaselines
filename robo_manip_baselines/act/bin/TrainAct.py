import argparse
import datetime
import glob
import os
import pickle
import random
import sys
from copy import deepcopy

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../third_party/act"))
from detr.models.detr_vae import DETRVAE
from policy import ACTPolicy
from utils import compute_dict_mean, detach_dict, set_seed

from robo_manip_baselines.act import RmbActDataset
from robo_manip_baselines.common import DataKey, get_skipped_data_seq


class TrainAct(object):
    def __init__(self):
        self.setup_args()

        self.setup_dataset()

        self.setup_policy()

    def setup_args(self):
        parser = argparse.ArgumentParser(
            description="Train ACT",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        parser.add_argument(
            "--dataset_dir",
            required=True,
            type=str,
            help="dataset directory",
        )
        parser.add_argument(
            "--checkpoint_dir",
            default=None,
            type=str,
            help="checkpoint directory",
        )
        parser.add_argument(
            "--state_keys",
            default=[DataKey.MEASURED_JOINT_POS],
            nargs="*",
            choices=DataKey.MEASURED_DATA_KEYS,
            type=str,
            help="state data keys",
        )
        parser.add_argument(
            "--action_keys",
            default=[DataKey.COMMAND_JOINT_POS],
            nargs="+",
            choices=DataKey.COMMAND_DATA_KEYS,
            type=str,
            help="action data keys",
        )
        parser.add_argument(
            "--camera_names",
            default=["front"],
            nargs="+",
            type=str,
            help="camera names",
        )
        parser.add_argument(
            "--train_ratio", default=0.8, type=float, help="ratio of train data"
        )
        parser.add_argument(
            "--skip",
            default=3,
            type=int,
            help="skip interval of data sequence (set 1 for no skip)",
        )
        parser.add_argument("--batch_size", default=8, type=int, help="batch size")
        parser.add_argument("--seed", default=0, type=int, help="seed")
        parser.add_argument(
            "--num_epochs", default=1000, type=int, help="number of epochs"
        )
        parser.add_argument("--lr", default=1e-5, type=float, help="learning rate")

        # for ACT
        parser.add_argument("--kl_weight", default=10, type=int, help="KL weight")
        parser.add_argument(
            "--chunk_size", default=100, type=int, help="action chunking size"
        )
        parser.add_argument(
            "--hidden_dim", default=512, type=int, help="hidden dimension of ACT policy"
        )
        parser.add_argument(
            "--dim_feedforward",
            default=3200,
            type=int,
            help="feedforward dimension of ACT policy",
        )

        self.args = parser.parse_args()

        # Set checkpoint directory if it is not specified
        if self.args.checkpoint_dir is None:
            dataset_dirname = os.path.basename(os.path.normpath(self.args.dataset_dir))
            checkpoint_dirname = "{}_TrainAct_{:%Y%m%d_%H%M%S}".format(
                dataset_dirname, datetime.datetime.now()
            )
            self.args.checkpoint_dir = os.path.normpath(
                os.path.join(
                    os.path.dirname(__file__), "../checkpoint/", checkpoint_dirname
                )
            )

    def setup_dataset(self):
        set_seed(1)

        # Get file list
        all_filenames = glob.glob(f"{self.args.dataset_dir}/**/*.hdf5", recursive=True)
        random.shuffle(all_filenames)
        train_num = int(len(all_filenames) * self.args.train_ratio)
        train_filenames = all_filenames[:train_num]
        val_filenames = all_filenames[train_num:]

        # Construct dataset stats
        self.dataset_stats = self.make_dataset_stats(all_filenames)

        # Construct dataloader
        self.train_dataloader = self.make_dataloader(train_filenames)
        self.val_dataloader = self.make_dataloader(val_filenames)
        print(
            f"[TrainAct] Load dataset from {self.args.dataset_dir}\n"
            f"  - train episodes: {len(train_filenames)}, val episodes: {len(val_filenames)}"
        )

    def make_dataset_stats(self, all_filenames):
        # Load all state and action
        all_state = []
        all_action = []
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
                action = np.concatenate(
                    [
                        get_skipped_data_seq(
                            h5file[action_key][()], action_key, self.args.skip
                        )
                        for action_key in self.args.action_keys
                    ],
                    axis=1,
                )
                all_state.append(state)
                all_action.append(action)
        all_state = np.concatenate(all_state, dtype=np.float32)
        all_action = np.concatenate(all_action, dtype=np.float32)

        # Calculate stats
        state_mean = all_state.mean(axis=0)
        state_std = np.clip(all_state.std(axis=0), 1e-2, np.inf)
        action_mean = all_action.mean(axis=0)
        action_std = np.clip(all_action.std(axis=0), 1e-2, np.inf)

        return {
            # Normalization
            "state_mean": state_mean,
            "state_std": state_std,
            "action_mean": action_mean,
            "action_std": action_std,
            # Example
            "example_state": all_state[0],
            "example_action": all_action[0],
            # Args
            "state_keys": self.args.state_keys,
            "action_keys": self.args.action_keys,
            "camera_names": self.args.camera_names,
            "skip": self.args.skip,
        }

    def make_dataloader(self, filenames):
        dataset = RmbActDataset(
            filenames,
            self.args.state_keys,
            self.args.action_keys,
            self.args.camera_names,
            self.dataset_stats,
            self.args.skip,
            self.args.chunk_size,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=1,
            prefetch_factor=1,
        )

        return dataloader

    def setup_policy(self):
        set_seed(self.args.seed)

        # Set dimensions of state and action
        state_dim = self.train_dataloader.dataset[0][0].shape[0]
        action_dim = self.train_dataloader.dataset[0][1].shape[1]
        DETRVAE.set_state_dim(state_dim)
        DETRVAE.set_action_dim(action_dim)
        print(
            "[TrainAct] Construct ACT policy.\n"
            f"  - state dim: {state_dim}, action dim: {action_dim}\n"
            f"  - state keys: {self.args.state_keys}\n"
            f"  - action keys: {self.args.action_keys}\n"
            f"  - camera names: {self.args.camera_names}\n"
            f"  - skip: {self.args.skip}"
        )

        # Set policy config
        lr_backbone = 1e-5
        backbone = "resnet18"
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        self.policy_config = {
            "lr": self.args.lr,
            "num_queries": self.args.chunk_size,
            "kl_weight": self.args.kl_weight,
            "hidden_dim": self.args.hidden_dim,
            "dim_feedforward": self.args.dim_feedforward,
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "enc_layers": enc_layers,
            "dec_layers": dec_layers,
            "nheads": nheads,
            "camera_names": self.args.camera_names,
        }

        # Construct policy
        self.policy = ACTPolicy(self.policy_config)
        self.policy.cuda()

        # Construct optimizer
        self.optimizer = self.policy.configure_optimizers()

    def run(self):
        os.makedirs(self.args.checkpoint_dir, exist_ok=True)

        # Save dataset stats
        dataset_stats_path = os.path.join(self.args.checkpoint_dir, "dataset_stats.pkl")
        with open(dataset_stats_path, "wb") as f:
            pickle.dump(self.dataset_stats, f)
        print(f"[TrainAct] Save dataset stats: {dataset_stats_path}")

        # Save policy config
        policy_config_path = os.path.join(self.args.checkpoint_dir, "policy_config.pkl")
        with open(policy_config_path, "wb") as f:
            pickle.dump(self.policy_config, f)
        print(f"[TrainAct] Save policy config: {policy_config_path}")

        # Train
        print(f"[TrainAct] Train with saving checkpoints: {self.args.checkpoint_dir}")
        best_ckpt_info = self.train_bc()
        best_epoch, min_val_loss, best_state_dict = best_ckpt_info

        # Save best checkpoint
        ckpt_path = os.path.join(self.args.checkpoint_dir, "policy_best.ckpt")
        torch.save(best_state_dict, ckpt_path)
        print(
            f"[TrainAct] Save the best checkpoint. val loss is {min_val_loss:.3f} at epoch {best_epoch}"
        )

    def train_bc(self, print_summary=False):
        train_history = []
        validation_history = []
        min_val_loss = np.inf
        best_ckpt_info = None
        for epoch in tqdm(range(self.args.num_epochs)):
            # Run validation step
            with torch.inference_mode():
                self.policy.eval()
                epoch_dicts = []
                for batch_idx, data in enumerate(self.val_dataloader):
                    forward_dict = self.forward_pass(data)
                    epoch_dicts.append(forward_dict)
                epoch_summary = compute_dict_mean(epoch_dicts)
                validation_history.append(epoch_summary)

                epoch_val_loss = epoch_summary["loss"]
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (
                        epoch,
                        min_val_loss,
                        deepcopy(self.policy.state_dict()),
                    )
            if print_summary:
                summary_string = "[TrainAct][val]"
                for k, v in epoch_summary.items():
                    summary_string += f" {k}: {v.item():.3f}"
                print(summary_string)

            # Run train step
            self.policy.train()
            self.optimizer.zero_grad()
            for batch_idx, data in enumerate(self.train_dataloader):
                forward_dict = self.forward_pass(data)
                loss = forward_dict["loss"]
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                train_history.append(detach_dict(forward_dict))
            epoch_summary = compute_dict_mean(
                train_history[(batch_idx + 1) * epoch : (batch_idx + 1) * (epoch + 1)]
            )
            if print_summary:
                summary_string = "[TrainAct][train]"
                for k, v in epoch_summary.items():
                    summary_string += f" {k}: {v.item():.3f}"
                print(summary_string)

            # Save current checkpoint
            if epoch % 100 == 0:
                ckpt_path = os.path.join(
                    self.args.checkpoint_dir,
                    f"policy_epoch_{epoch}_seed_{self.args.seed}.ckpt",
                )
                torch.save(self.policy.state_dict(), ckpt_path)
                self.plot_history(train_history, validation_history, epoch)

        # Save last checkpoint
        ckpt_path = os.path.join(self.args.checkpoint_dir, "policy_last.ckpt")
        torch.save(self.policy.state_dict(), ckpt_path)
        self.plot_history(train_history, validation_history, self.args.num_epochs)

        return best_ckpt_info

    def forward_pass(self, data):
        state_tensor, action_tensor, image_tensor, is_pad_tensor = data
        return self.policy(
            state_tensor.cuda(),
            image_tensor.cuda(),
            action_tensor.cuda(),
            is_pad_tensor.cuda(),
        )

    def plot_history(self, train_history, validation_history, epoch):
        for key in train_history[0]:
            plot_path = os.path.join(
                self.args.checkpoint_dir, f"train_val_{key}_seed_{self.args.seed}.png"
            )
            plt.figure()
            train_values = [summary[key].item() for summary in train_history]
            val_values = [summary[key].item() for summary in validation_history]
            plt.plot(
                np.linspace(0, epoch - 1, len(train_history)),
                train_values,
                label="train",
            )
            plt.plot(
                np.linspace(0, epoch - 1, len(validation_history)),
                val_values,
                label="validation",
            )
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.legend()
            plt.title(key)
            plt.savefig(plot_path)
            plt.close()


if __name__ == "__main__":
    train = TrainAct()
    train.run()
