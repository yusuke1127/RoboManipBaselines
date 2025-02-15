import argparse
import os
import sys
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../third_party/act"))
from detr.models.detr_vae import DETRVAE
from policy import ACTPolicy

from robo_manip_baselines.act import RmbActDataset
from robo_manip_baselines.common import (
    TrainBase,
    set_random_seed,
)


class TrainAct(TrainBase):
    policy_name = "ACT"
    policy_dir = os.path.join(os.path.dirname(__file__), "..")
    DatasetClass = RmbActDataset

    def setup_args(self):
        parser = argparse.ArgumentParser(
            description="Train ACT policy",
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

    def make_model_meta_info(self, all_filenames):
        model_meta_info = super().make_model_meta_info(all_filenames)

        model_meta_info["data"]["chunk_size"] = self.args.chunk_size

        return model_meta_info

    def setup_policy(self):
        set_random_seed(self.args.seed)

        # Set dimensions of state and action
        state_dim = len(self.model_meta_info["state"]["example"])
        action_dim = len(self.model_meta_info["action"]["example"])
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

    def train_loop(self):
        best_ckpt_info = {"val_loss": np.inf}
        for epoch in tqdm(range(self.args.num_epochs)):
            # Run validation step
            with torch.inference_mode():
                self.policy.eval()
                epoch_result_list = []
                for data in self.val_dataloader:
                    epoch_result = self.policy(*[d.cuda() for d in data])
                    epoch_result_list.append(self.detach_epoch_result(epoch_result))
                epoch_summary = self.calc_epoch_summary(epoch_result_list)
                for k, v in epoch_summary.items():
                    self.writer.add_scalar(f"{k}/val", v, epoch)

                # Check best
                epoch_val_loss = epoch_summary["loss"]
                if epoch_val_loss < best_ckpt_info["val_loss"]:
                    best_ckpt_info = {
                        "epoch": epoch,
                        "val_loss": epoch_val_loss,
                        "state_dict": deepcopy(self.policy.state_dict()),
                    }

            # Run train step
            self.policy.train()
            self.optimizer.zero_grad()
            epoch_result_list = []
            for data in self.train_dataloader:
                epoch_result = self.policy(*[d.cuda() for d in data])
                loss = epoch_result["loss"]
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                epoch_result_list.append(self.detach_epoch_result(epoch_result))
            epoch_summary = self.calc_epoch_summary(epoch_result_list)
            for k, v in epoch_summary.items():
                self.writer.add_scalar(f"{k}/train", v, epoch)

            if epoch % 100 == 0:
                # Save current checkpoint
                ckpt_path = os.path.join(
                    self.args.checkpoint_dir, f"ACT_epoch{epoch:0>3}.ckpt"
                )
                torch.save(self.policy.state_dict(), ckpt_path)

        # Save last checkpoint
        ckpt_path = os.path.join(self.args.checkpoint_dir, "ACT_last.ckpt")
        torch.save(self.policy.state_dict(), ckpt_path)

        # Save best checkpoint
        ckpt_path = os.path.join(self.args.checkpoint_dir, "ACT_best.ckpt")
        torch.save(best_ckpt_info["state_dict"], ckpt_path)
        print(
            f"[TrainAct] Best val loss is {best_ckpt_info['val_loss']:.3f} at epoch {best_ckpt_info['epoch']}"
        )


if __name__ == "__main__":
    train = TrainAct()
    train.run()
    train.close()
