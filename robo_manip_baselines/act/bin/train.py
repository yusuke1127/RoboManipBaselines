import torch
import numpy as np
import sys
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from robo_manip_baselines.act import load_data

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../third_party/act"))
from utils import compute_dict_mean, set_seed, detach_dict
from policy import ACTPolicy
from detr.models.detr_vae import DETRVAE


class TrainAct(object):
    def __init__(self):
        self.setup_args()

        self.setup_dataset()

        self.setup_policy()

    def setup_args(self):
        parser = argparse.ArgumentParser(description="Train ACT")

        parser.add_argument(
            "--dataset_dir",
            action="store",
            type=str,
            help="dataset_dir",
            required=False,
            default="./data/",
        )
        parser.add_argument(
            "--ckpt_dir",
            action="store",
            type=str,
            help="ckpt_dir",
            required=False,
            default="./log/",
        )
        parser.add_argument(
            "--camera_names",
            action="store",
            type=lambda x: list(map(str, x.split(","))),
            help="camera_names",
            required=False,
            default=["front"],
        )
        parser.add_argument(
            "--batch_size", action="store", type=int, help="batch_size", required=True
        )
        parser.add_argument(
            "--seed", action="store", type=int, help="seed", required=True
        )
        parser.add_argument(
            "--num_epochs", action="store", type=int, help="num_epochs", required=True
        )
        parser.add_argument(
            "--lr", action="store", type=float, help="lr", required=True
        )

        # for ACT
        parser.add_argument(
            "--kl_weight", action="store", type=int, help="KL Weight", required=False
        )
        parser.add_argument(
            "--chunk_size", action="store", type=int, help="chunk_size", required=False
        )
        parser.add_argument(
            "--hidden_dim", action="store", type=int, help="hidden_dim", required=False
        )
        parser.add_argument(
            "--dim_feedforward",
            action="store",
            type=int,
            help="dim_feedforward",
            required=False,
        )

        self.args = parser.parse_args()

    def setup_dataset(self):
        set_seed(1)

        is_sim = True
        batch_size_train = self.args.batch_size
        batch_size_val = self.args.batch_size
        self.train_dataloader, self.val_dataloader, self.stats, _ = load_data(
            self.args.dataset_dir,
            is_sim,
            self.args.camera_names,
            batch_size_train,
            batch_size_val,
        )

    def setup_policy(self):
        set_seed(self.args.seed)

        state_dim = self.train_dataloader.dataset[0][1].shape[0]
        action_dim = self.train_dataloader.dataset[0][2].shape[1]
        DETRVAE.set_state_dim(state_dim)
        DETRVAE.set_action_dim(action_dim)

        lr_backbone = 1e-5
        backbone = "resnet18"
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {
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

        self.policy = ACTPolicy(policy_config)
        self.policy.cuda()

        self.optimizer = self.policy.configure_optimizers()

    def run(self):
        # save dataset stats
        if not os.path.isdir(self.args.ckpt_dir):
            os.makedirs(self.args.ckpt_dir)
        stats_path = os.path.join(self.args.ckpt_dir, "dataset_stats.pkl")
        with open(stats_path, "wb") as f:
            pickle.dump(self.stats, f)
        print(f"[TrainAct] Save dataset stats: {stats_path}")

        # train
        best_ckpt_info = self.train_bc()
        best_epoch, min_val_loss, best_state_dict = best_ckpt_info

        # save best checkpoint
        ckpt_path = os.path.join(self.args.ckpt_dir, "policy_best.ckpt")
        torch.save(best_state_dict, ckpt_path)
        print(f"[TrainAct] Best ckpt, val loss {min_val_loss:.3f} @ epoch{best_epoch}")

    def train_bc(self):
        train_history = []
        validation_history = []
        min_val_loss = np.inf
        best_ckpt_info = None
        for epoch in tqdm(range(self.args.num_epochs)):
            # validation
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
            summary_string = f"[TrainAct] val loss: {epoch_val_loss:.3f}"
            for k, v in epoch_summary.items():
                summary_string += f", {k}: {v.item():.3f}"
            print(summary_string)

            # training
            self.policy.train()
            self.optimizer.zero_grad()
            for batch_idx, data in enumerate(self.train_dataloader):
                forward_dict = self.forward_pass(data)
                # backward
                loss = forward_dict["loss"]
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                train_history.append(detach_dict(forward_dict))
            epoch_summary = compute_dict_mean(
                train_history[(batch_idx + 1) * epoch : (batch_idx + 1) * (epoch + 1)]
            )
            epoch_train_loss = epoch_summary["loss"]
            summary_string = f"[TrainAct] train loss: {epoch_train_loss:.3f}"
            for k, v in epoch_summary.items():
                summary_string += f", {k}: {v.item():.3f}"
            print(summary_string)

            if epoch % 100 == 0:
                ckpt_path = os.path.join(
                    self.args.ckpt_dir,
                    f"policy_epoch_{epoch}_seed_{self.args.seed}.ckpt",
                )
                torch.save(self.policy.state_dict(), ckpt_path)
                self.plot_history(train_history, validation_history, epoch)

        ckpt_path = os.path.join(self.args.ckpt_dir, "policy_last.ckpt")
        torch.save(self.policy.state_dict(), ckpt_path)

        best_epoch, min_val_loss, best_state_dict = best_ckpt_info
        ckpt_path = os.path.join(
            self.args.ckpt_dir, f"policy_epoch_{best_epoch}_seed_{self.args.seed}.ckpt"
        )
        torch.save(best_state_dict, ckpt_path)
        print(
            f"[TrainAct] Training finished: seed {self.args.seed}, val loss {min_val_loss:.3f} at epoch {best_epoch}"
        )

        # save training curves
        self.plot_history(train_history, validation_history, self.args.num_epochs)

        return best_ckpt_info

    def forward_pass(self, data):
        image_data, joint_data, action_data, is_pad = data
        image_data, joint_data, action_data, is_pad = (
            image_data.cuda(),
            joint_data.cuda(),
            action_data.cuda(),
            is_pad.cuda(),
        )
        return self.policy(
            joint_data, image_data, action_data, is_pad
        )  # TODO remove None

    def plot_history(self, train_history, validation_history, epoch):
        # save training curves
        for key in train_history[0]:
            plot_path = os.path.join(
                self.args.ckpt_dir, f"train_val_{key}_seed_{self.args.seed}.png"
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
            # plt.ylim([-0.1, 1])
            plt.tight_layout()
            plt.legend()
            plt.title(key)
            plt.savefig(plot_path)
        print(f"[TrainAct] Saved plots to {self.args.ckpt_dir}")


if __name__ == "__main__":
    train = TrainAct()
    train.run()
