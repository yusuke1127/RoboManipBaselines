import os
import sys
import argparse
import pickle
import numpy as np
import matplotlib.pylab as plt
import cv2
import torch

sys.path.append(
    os.path.join(os.path.dirname(__file__), "../../../third_party/roboagent")
)
from policy import ACTPolicy
from robo_manip_baselines.mt_act import TASKS, TEXT_EMBEDDINGS
from robo_manip_baselines.common.rollout import RolloutBase


class RolloutMtAct(RolloutBase):
    def setup_args(self, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser()

        parser.add_argument(
            "--ckpt_dir",
            action="store",
            type=str,
            help="checkpoint directory",
            required=True,
        )
        parser.add_argument(
            "--ckpt_name",
            default="policy_best.ckpt",
            type=str,
            help="ACT policy checkpoint file name (*.ckpt)",
        )
        parser.add_argument(
            "--task_name",
            choices=TASKS,
            action="store",
            type=str,
            help="task name",
            required=True,
        )
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

        # Add dummy arguments
        parser.add_argument(
            "--policy_class",
            choices=["ACT"],
            type=str,
            help="Do not set this manually as it is a dummy argument.",
            required=True,
        )
        parser.add_argument(
            "--num_epochs",
            choices=[-1],
            type=int,
            help="Do not set this manually as it is a dummy argument.",
            required=True,
        )

        argv = sys.argv
        argv += ["--policy_class", "ACT", "--num_epochs", "-1", "--multi_task"]
        super().setup_args(parser, argv)

        if self.args.skip is None:
            self.args.skip = 3
        if self.args.skip_draw is None:
            self.args.skip_draw = self.args.skip

    def setup_policy(self):
        # Set task embedding
        task_idx = TASKS.index(self.args.task_name)
        self.task_emb = np.asarray(TEXT_EMBEDDINGS[task_idx])
        self.task_emb = torch.from_numpy(self.task_emb).unsqueeze(0).float().cuda()

        # Define policy
        self.policy_config = {
            "num_queries": self.args.chunk_size,
            "kl_weight": self.args.kl_weight,
            "hidden_dim": self.args.hidden_dim,
            "dim_feedforward": self.args.dim_feedforward,
            "lr_backbone": 1e-5,
            "backbone": "resnet18",
            "enc_layers": 4,
            "dec_layers": 7,
            "nheads": 8,
            "camera_names": ["front"],
        }
        self.policy = ACTPolicy(self.policy_config)

        def forward_fook(_layer, _input, _output):
            # Output of MultiheadAttention is a tuple (attn_output, attn_output_weights)
            # https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
            _layer.correlation_mat = _output[1][0].detach().cpu().numpy()

        for layer in self.policy.model.transformer.encoder.layers:
            layer.self_attn.correlation_mat = None
            layer.self_attn.register_forward_hook(forward_fook)

        # Load weight
        ckpt_path = os.path.join(self.args.ckpt_dir, self.args.ckpt_name)
        try:
            print(f"[RolloutMtAct] Load {ckpt_path}")
            self.policy.load_state_dict(torch.load(ckpt_path))
        except RuntimeError as e:
            if "size mismatch" in str(e.self.args):
                sys.stderr.write(f"\n{sys.stderr.name} {self.args.chunk_size=}\n\n")
            raise
        self.policy.cuda()
        self.policy.eval()

        # Load data statistics
        stats_path = os.path.join(self.args.ckpt_dir, "dataset_stats.pkl")
        with open(stats_path, "rb") as f:
            self.stats = pickle.load(f)

        # Set variables
        self.joint_dim = 7
        self.joint_scales = [1.0] * 6 + [0.01]
        self.pred_action_list = np.empty((0, self.joint_dim))
        self.all_actions_history = []

    def setup_plot(self):
        fig_ax = plt.subplots(
            2,
            max(2, self.policy_config["enc_layers"]),
            figsize=(13.5, 6.0),
            dpi=60,
            squeeze=False,
        )
        super().setup_plot(fig_ax=fig_ax)

    def infer_policy(self):
        if self.auto_time_idx % self.args.skip != 0:
            return False

        # Preprocess
        self.front_image = self.info["rgb_images"]["front"]
        front_image_input = self.front_image.transpose(2, 0, 1)
        front_image_input = front_image_input.astype(np.float32) / 255.0
        front_image_input = (
            torch.Tensor(np.expand_dims(front_image_input, 0)).cuda().unsqueeze(0)
        )
        joint_input = self.motion_manager.get_joint_pos(self.obs)
        joint_input = (joint_input - self.stats["joint_mean"]) / self.stats["joint_std"]
        joint_input = torch.Tensor(np.expand_dims(joint_input, 0)).cuda()

        # Infer
        all_actions = self.policy(
            joint_input, front_image_input, task_emb=self.task_emb
        )[0]
        self.all_actions_history.append(all_actions.cpu().detach().numpy())
        if len(self.all_actions_history) > self.args.chunk_size:
            self.all_actions_history.pop(0)

        # Postprocess (temporal ensembling)
        k = 0.01
        exp_weights = np.exp(-k * np.arange(len(self.all_actions_history)))
        exp_weights = exp_weights / exp_weights.sum()
        action = np.zeros(self.joint_dim)
        for action_idx, _all_actions in enumerate(reversed(self.all_actions_history)):
            action += exp_weights[::-1][action_idx] * _all_actions[action_idx]
        self.pred_action = action * self.stats["action_std"] + self.stats["action_mean"]
        self.pred_action_list = np.concatenate(
            [self.pred_action_list, np.expand_dims(self.pred_action, 0)]
        )

        return True

    def draw_plot(self):
        if self.auto_time_idx % self.args.skip_draw != 0:
            return

        for _ax in np.ravel(self.ax):
            _ax.cla()
            _ax.axis("off")

        # Draw observed image
        self.ax[0, 0].imshow(self.front_image)
        self.ax[0, 0].set_title("Observed image", fontsize=20)

        # Plot joint
        xlim = 500 // self.args.skip
        self.ax[0, 1].set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        self.ax[0, 1].set_xlim(0, xlim)
        for joint_idx in range(self.pred_action_list.shape[1]):
            self.ax[0, 1].plot(
                np.arange(self.pred_action_list.shape[0]),
                self.pred_action_list[:, joint_idx] * self.joint_scales[joint_idx],
            )
        self.ax[0, 1].set_xlabel("Step", fontsize=20)
        self.ax[0, 1].set_title("Joint", fontsize=20)
        self.ax[0, 1].tick_params(axis="x", labelsize=16)
        self.ax[0, 1].tick_params(axis="y", labelsize=16)
        self.ax[0, 1].axis("on")

        # Draw attention images
        for layer_idx, layer in enumerate(self.policy.model.transformer.encoder.layers):
            if layer.self_attn.correlation_mat is None:
                continue
            self.ax[1, layer_idx].imshow(
                layer.self_attn.correlation_mat[3:, 1].reshape((15, 20))
            )
            self.ax[1, layer_idx].set_title(f"Attention ({layer_idx})", fontsize=20)

        self.fig.tight_layout()
        self.canvas.draw()
        cv2.imshow(
            "Policy image",
            cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGB2BGR),
        )
