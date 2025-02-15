import argparse

import cv2
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from torchvision.transforms import v2

from robo_manip_baselines.common import denormalize_data, normalize_data
from robo_manip_baselines.common.rollout import RolloutBase

from .MlpPolicy import MlpPolicy


class RolloutMlp(RolloutBase):
    def setup_args(self, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )

        parser.add_argument(
            "--checkpoint", type=str, help="checkpoint file", required=True
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
        print(
            f"[{self.__class__.__name__}] Construct policy.\n"
            f"  - state dim: {self.state_dim}, action dim: {self.action_dim}, camera num: {len(self.camera_names)}\n"
            f"  - state keys: {self.state_keys}\n"
            f"  - action keys: {self.action_keys}\n"
            f"  - camera names: {self.camera_names}\n"
            f"  - skip: {self.args.skip}"
        )

        # Construct policy
        self.policy = MlpPolicy(
            self.state_dim,
            self.action_dim,
            len(self.camera_names),
            self.policy_config["hidden_dim_list"],
            self.policy_config["state_feature_dim"],
        )

        # Load weight
        print(f"[{self.__class__.__name__}] Load {self.args.checkpoint}")
        self.policy.load_state_dict(torch.load(self.args.checkpoint, weights_only=True))
        self.policy.cuda()
        self.policy.eval()

        # Setup image transforms
        self.image_transforms = v2.ToDtype(torch.float32, scale=True)

        # Set variables
        self.policy_action_list = np.empty((0, self.action_dim))

    def setup_plot(self):
        fig_ax = plt.subplots(
            2,
            len(self.camera_names),
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
        images = torch.tensor(images, dtype=torch.uint8)
        images = self.image_transforms(images)[np.newaxis].cuda()

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
        action = self.policy(state, images)[0]
        action = action.cpu().detach().numpy()
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
        joint_ax = self.ax[1, 0]
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

        self.fig.tight_layout()
        self.canvas.draw()
        cv2.imshow(
            "Policy image",
            cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGB2BGR),
        )
