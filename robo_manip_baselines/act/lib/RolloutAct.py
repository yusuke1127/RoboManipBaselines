import argparse
import os
import sys

import cv2
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../third_party/act"))
from detr.models.detr_vae import DETRVAE
from policy import ACTPolicy

from robo_manip_baselines.common import denormalize_data, normalize_data
from robo_manip_baselines.common.rollout import RolloutBase


class RolloutAct(RolloutBase):
    def setup_args(self, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )

        parser.add_argument(
            "--checkpoint",
            type=str,
            help="checkpoint file of ACT (*.ckpt)",
            required=True,
        )

        super().setup_args(parser)

    def setup_policy(self):
        # Load model meta info
        self.load_model_meta_info()

        # Set skip if not specified
        if self.args.skip is None:
            self.args.skip = self.model_meta_info["data"]["skip"]
        if self.args.skip_draw is None:
            self.args.skip_draw = self.args.skip

        # Set dimensions of state and action
        self.state_keys = self.model_meta_info["state"]["keys"]
        self.action_keys = self.model_meta_info["action"]["keys"]
        self.camera_names = self.model_meta_info["image"]["camera_names"]
        self.state_dim = len(self.model_meta_info["state"]["example"])
        self.action_dim = len(self.model_meta_info["action"]["example"])
        self.policy_config = self.model_meta_info["policy_config"]
        DETRVAE.set_state_dim(self.state_dim)
        DETRVAE.set_action_dim(self.action_dim)
        print(
            "[RolloutAct] Construct ACT policy.\n"
            f"  - state dim: {self.state_dim}, action dim: {self.action_dim}, camera num: {len(self.camera_names)}\n"
            f"  - state keys: {self.state_keys}\n"
            f"  - action keys: {self.action_keys}\n"
            f"  - camera names: {self.camera_names}\n"
            f"  - skip: {self.args.skip}, chunk size: {self.policy_config['num_queries']}"
        )

        # Construct policy
        self.policy = ACTPolicy(self.policy_config)

        # Register fook to visualize attention images
        def forward_fook(_layer, _input, _output):
            # Output of MultiheadAttention is a tuple (attn_output, attn_output_weights)
            # https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
            _layer.correlation_mat = _output[1][0].detach().cpu().numpy()

        for layer in self.policy.model.transformer.encoder.layers:
            layer.self_attn.correlation_mat = None
            layer.self_attn.register_forward_hook(forward_fook)

        # Load weight
        print(f"[RolloutAct] Load {self.args.checkpoint}")
        self.policy.load_state_dict(torch.load(self.args.checkpoint, weights_only=True))
        self.policy.cuda()
        self.policy.eval()

        # Set variables
        self.policy_action_list = np.empty((0, self.action_dim))
        self.all_actions_history = []

    def setup_plot(self):
        fig_ax = plt.subplots(
            2,
            max(len(self.camera_names) + 1, self.policy_config["enc_layers"]),
            figsize=(13.5, 6.0),
            dpi=60,
            squeeze=False,
        )
        super().setup_plot(fig_ax=fig_ax)

    def infer_policy(self):
        if self.rollout_time_idx % self.args.skip != 0:
            return False

        # Get images
        images = np.stack(
            [self.info["rgb_images"][camera_name] for camera_name in self.camera_names],
            axis=0,
        )
        images = np.einsum("k h w c -> k c h w", images)
        images = images / 255.0
        images = torch.tensor(images[np.newaxis], dtype=torch.float32).cuda()

        # Get state
        if len(self.state_keys) == 0:
            state = np.zeros(0, dtype=np.float32)
        else:
            state = np.concatenate(
                [
                    self.motion_manager.get_measured_data(state_key, self.obs)
                    for state_key in self.state_keys
                ]
            )

        state = normalize_data(state, self.model_meta_info["state"])
        state = torch.tensor(state[np.newaxis], dtype=torch.float32).cuda()

        # Infer
        all_actions = self.policy(state, images)[0]
        self.all_actions_history.append(all_actions.cpu().detach().numpy())
        if len(self.all_actions_history) > self.policy_config["num_queries"]:
            self.all_actions_history.pop(0)

        # Apply temporal ensembling to action
        k = 0.01
        exp_weights = np.exp(-k * np.arange(len(self.all_actions_history)))
        exp_weights = exp_weights / exp_weights.sum()
        action = np.zeros(self.action_dim)
        for action_idx, _all_actions in enumerate(reversed(self.all_actions_history)):
            action += exp_weights[::-1][action_idx] * _all_actions[action_idx]
        self.policy_action = denormalize_data(action, self.model_meta_info["action"])
        self.policy_action_list = np.concatenate(
            [self.policy_action_list, self.policy_action[np.newaxis]]
        )

        return True

    def draw_plot(self):
        if self.rollout_time_idx % self.args.skip_draw != 0:
            return

        for _ax in np.ravel(self.ax):
            _ax.cla()
            _ax.axis("off")

        # Draw observed image
        for camera_idx, camera_name in enumerate(self.camera_names):
            self.ax[0, camera_idx].imshow(self.info["rgb_images"][camera_name])
            self.ax[0, camera_idx].set_title(f"{camera_name} image", fontsize=20)

        # Plot joint
        joint_ax = self.ax[0, len(self.camera_names)]
        joint_ax.plot(
            np.arange(self.policy_action_list.shape[0]),
            self.policy_action_list * self.action_plot_scale,
        )
        joint_ax.set_title("scaled action", fontsize=20)
        joint_ax.set_xlim(0, 500 // self.args.skip)
        joint_ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        joint_ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        joint_ax.set_xlabel("step", fontsize=16)
        joint_ax.tick_params(axis="x", labelsize=16)
        joint_ax.tick_params(axis="y", labelsize=16)
        joint_ax.axis("on")

        # Draw attention images
        attention_shape = (15, 20 * len(self.camera_names))
        for layer_idx, layer in enumerate(self.policy.model.transformer.encoder.layers):
            if layer.self_attn.correlation_mat is None:
                continue
            self.ax[1, layer_idx].imshow(
                layer.self_attn.correlation_mat[2:, 1].reshape(attention_shape)
            )
            self.ax[1, layer_idx].set_title(
                f"attention image ({layer_idx})", fontsize=20
            )

        self.fig.tight_layout()
        self.canvas.draw()
        cv2.imshow(
            "Policy image",
            cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGB2BGR),
        )
