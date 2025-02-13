import argparse
import os
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
from utils import compute_dict_mean, detach_dict

from robo_manip_baselines.act import RmbActDataset
from robo_manip_baselines.common import (
    DataKey,
    TrainBase,
    get_skipped_data_seq,
    set_random_seed,
)


class TrainAct(TrainBase):
    def setup_args(self):
        parser = argparse.ArgumentParser(
            description="Train ACT",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        parser.add_argument("--batch_size", type=int, default=8, help="batch size")
        parser.add_argument(
            "--num_epochs", type=int, default=1000, help="number of epochs"
        )
        parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")

        parser.add_argument("--kl_weight", type=int, default=10, help="KL weight")
        parser.add_argument(
            "--chunk_size", type=int, default=100, help="action chunking size"
        )
        parser.add_argument(
            "--hidden_dim", type=int, default=512, help="hidden dimension of ACT policy"
        )
        parser.add_argument(
            "--dim_feedforward",
            type=int,
            default=3200,
            help="feedforward dimension of ACT policy",
        )

        super().setup_args(parser)

    def make_dataloader(self, filenames):
        dataset = RmbActDataset(
            filenames,
            self.model_meta_info,
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
        set_random_seed(self.args.seed)

        # Set dimensions of state and action
        state_dim = train_dataloader.dataset[0][0].shape[0]
        action_dim = self.train_dataloader.dataset[0][1].shape[1]
        DETRVAE.set_state_dim(state_dim)
        DETRVAE.set_action_dim(action_dim)
        print(
            "[TrainAct] Construct ACT policy.\n"
            f"  - state dim: {state_dim}, action dim: {action_dim}, camera num: {len(self.args.camera_names)}\n"
            f"  - state keys: {self.args.state_keys}\n"
            f"  - action keys: {self.args.action_keys}\n"
            f"  - camera names: {self.args.camera_names}\n"
            f"  - skip: {self.args.skip}, chunk size: {self.args.chunk_size}"
        )

        # Set policy config
        policy_config = {
            "lr": self.args.lr,
            "num_queries": self.args.chunk_size,
            "kl_weight": self.args.kl_weight,
            "hidden_dim": self.args.hidden_dim,
            "dim_feedforward": self.args.dim_feedforward,
            "lr_backbone": 1e-5,
            "backbone": "resnet18",
            "enc_layers": 4,
            "dec_layers": 7,
            "nheads": 8,
            "camera_names": self.args.camera_names,
        }
        self.model_meta_info["policy_config"] = policy_config

        # Construct policy
        self.policy = ACTPolicy(policy_config)
        self.policy.cuda()

        # Construct optimizer
        self.optimizer = self.policy.configure_optimizers()

    def train_loop(self, print_summary=False):
        train_history = []
        validation_history = []
        min_val_loss = np.inf
        best_ckpt_info = None
        for epoch in tqdm(range(self.args.num_epochs)):
            # Run validation step
            with torch.inference_mode():
                self.policy.eval()
                epoch_dicts = []
                for data in self.val_dataloader:
                    epoch_dict = self.infer_policy(data)
                    epoch_dicts.append(detach_dict(epoch_dict))
                epoch_summary = compute_dict_mean(epoch_dicts)
                validation_history.append(epoch_summary)

                epoch_val_loss = epoch_summary["loss"]
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = {
                        "epoch": epoch,
                        "val_loss": min_val_loss,
                        "state_dict": deepcopy(self.policy.state_dict()),
                    }
            if print_summary:
                summary_string = "[TrainAct][val]"
                for k, v in epoch_summary.items():
                    summary_string += f" {k}: {v.item():.3f}"
                print(summary_string)

            # Run train step
            self.policy.train()
            self.optimizer.zero_grad()
            for batch_idx, data in enumerate(self.train_dataloader):
                epoch_dict = self.infer_policy(data)
                loss = epoch_dict["loss"]
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                train_history.append(detach_dict(epoch_dict))
            epoch_summary = compute_dict_mean(train_history[-1 * (batch_idx + 1) :])
            if print_summary:
                summary_string = "[TrainAct][train]"
                for k, v in epoch_summary.items():
                    summary_string += f" {k}: {v.item():.3f}"
                print(summary_string)

            if epoch % 100 == 0:
                # Save current checkpoint
                ckpt_path = os.path.join(
                    self.args.checkpoint_dir, f"ACT_epoch{epoch:0>3}.ckpt"
                )
                torch.save(self.policy.state_dict(), ckpt_path)

                # Plot current status
                self.plot_history(train_history, validation_history, epoch)

        # Save last checkpoint
        ckpt_path = os.path.join(self.args.checkpoint_dir, "ACT_last.ckpt")
        torch.save(self.policy.state_dict(), ckpt_path)

        # Plot last status
        self.plot_history(train_history, validation_history, self.args.num_epochs)

        # Save best checkpoint
        best_epoch, min_val_loss, best_state_dict = best_ckpt_info
        ckpt_path = os.path.join(self.args.checkpoint_dir, "ACT_best.ckpt")
        torch.save(best_ckpt_info["state_dict"], ckpt_path)
        print(
            f"[TrainAct] Best val loss is {best_ckpt_info['val_loss']:.3f} at epoch {best_ckpt_info['epoch']}"
        )

    def infer_policy(self, data):
        state_tensor, action_tensor, image_tensor, is_pad_tensor = data
        return self.policy(
            state_tensor.cuda(),
            image_tensor.cuda(),
            action_tensor.cuda(),
            is_pad_tensor.cuda(),
        )

    def plot_history(self, train_history, validation_history, epoch):
        for key in train_history[0]:
            plot_path = os.path.join(self.args.checkpoint_dir, f"plot_{key}.png")
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
