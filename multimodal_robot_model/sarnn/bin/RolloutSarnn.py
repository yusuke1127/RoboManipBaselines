import os
import argparse
import numpy as np
import matplotlib.pylab as plt
import cv2
import torch
from eipl.model import SARNN
from eipl.utils import restore_args, tensor2numpy, deprocess_img, normalization, resize_img
from multimodal_robot_model.demos.DemoUtils import MotionManager, RecordStatus, RecordManager
from multimodal_robot_model.common import RolloutBase

class RolloutSarnn(RolloutBase):
    def setupArgs(self, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser()

        parser.add_argument("--checkpoint", type=str, default=None, help="SARNN policy checkpoint file (*.pth)")

        super().setupArgs(parser)

        if self.args.skip is None:
            self.args.skip = 6
        if self.args.skip_draw is None:
            self.args.skip_draw = self.args.skip

    def setupPolicy(self):
        # Restore parameters
        checkpoint_dir = os.path.split(self.args.checkpoint)[0]
        self.params = restore_args(os.path.join(checkpoint_dir, "args.json"))

        # Define model
        self.im_size = 64
        self.joint_dim = 7
        self.policy = SARNN(
            rec_dim=self.params["rec_dim"],
            joint_dim=self.joint_dim,
            k_dim=self.params["k_dim"],
            heatmap_size=self.params["heatmap_size"],
            temperature=self.params["temperature"],
            im_size=[self.im_size, self.im_size],
        )
        if self.params["compile"]:
            self.policy = torch.compile(self.policy)

        # Load weight
        ckpt = torch.load(self.args.checkpoint, map_location=torch.device("cpu"))
        self.policy.load_state_dict(ckpt["model_state_dict"])
        self.policy.eval()

        # Set variables
        self.v_min_max = [self.params["vmin"], self.params["vmax"]]
        self.joint_bounds = np.load(os.path.join(checkpoint_dir, "action_bounds.npy"))
        self.joint_scales = [1.0] * 6 + [0.01]
        self.pred_joint_list = np.empty((0, self.joint_dim))
        self.rnn_state = None

    def setupPlot(self):
        fig_ax = plt.subplots(1, 3, figsize=(13.5, 5.0), dpi=60, squeeze=False)
        super().setupPlot(fig_ax=fig_ax)

    def inferPolicy(self):
        if self.auto_time_idx % self.args.skip == 0:
            # Load data and normalization
            self.front_image = self.info["rgb_images"]["front"]
            cropped_img_size = 280
            [fro_lef, fro_top] = [(self.front_image.shape[ax] - cropped_img_size) // 2 for ax in [0, 1]]
            [fro_rig, fro_bot] = [(p + cropped_img_size) for p in [fro_lef, fro_top]]
            self.front_image = self.front_image[fro_lef:fro_rig, fro_top:fro_bot, :]
            self.front_image = resize_img(np.expand_dims(self.front_image, 0), (self.im_size, self.im_size))[0]
            front_image_t = self.front_image.transpose(2, 0, 1)
            front_image_t = normalization(front_image_t, (0, 255), self.v_min_max)
            front_image_t = torch.Tensor(np.expand_dims(front_image_t, 0))
            joint = self.motion_manager.getAction()
            joint_t = normalization(joint, self.joint_bounds, self.v_min_max)
            joint_t = torch.Tensor(np.expand_dims(joint_t, 0))

            # Infer
            y_front_image, y_joint, y_enc_front_pts, y_dec_front_pts, self.rnn_state = self.policy(front_image_t, joint_t, self.rnn_state)

            # denormalization
            self.pred_front_image = tensor2numpy(y_front_image[0])
            self.pred_front_image = deprocess_img(self.pred_front_image, self.params["vmin"], self.params["vmax"])
            self.pred_front_image = self.pred_front_image.transpose(1, 2, 0)
            self.pred_joint = tensor2numpy(y_joint[0])
            self.pred_joint = normalization(self.pred_joint, self.v_min_max, self.joint_bounds)
            self.pred_joint_list = np.concatenate([self.pred_joint_list, np.expand_dims(self.pred_joint, 0)])
            self.enc_front_pts = tensor2numpy(y_enc_front_pts[0]).reshape(self.params["k_dim"], 2) * self.im_size
            self.dec_front_pts = tensor2numpy(y_dec_front_pts[0]).reshape(self.params["k_dim"], 2) * self.im_size

    def drawPlot(self):
        if self.auto_time_idx % self.args.skip_draw == 0:
            for _ax in np.ravel(self.ax):
                _ax.cla()
                _ax.axis("off")

            # Draw camera front_image
            self.ax[0, 0].imshow(self.front_image)
            for j in range(self.params["k_dim"]):
                self.ax[0, 0].plot(self.enc_front_pts[j, 0], self.enc_front_pts[j, 1], "co", markersize=12)  # encoder
                self.ax[0, 0].plot(
                    self.dec_front_pts[j, 0], self.dec_front_pts[j, 1], "rx", markersize=12, markeredgewidth=2
                )  # decoder
            self.ax[0, 0].axis("off")
            self.ax[0, 0].set_title("Input front_image", fontsize=20)

            # Draw predicted front_image
            self.ax[0, 1].imshow(self.pred_front_image)
            self.ax[0, 1].axis("off")
            self.ax[0, 1].set_title("Predicted front_image", fontsize=20)

            # Plot joint
            T = 100
            self.ax[0, 2].set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
            self.ax[0, 2].set_xlim(0, T)
            for joint_idx in range(self.pred_joint_list.shape[1]):
                self.ax[0, 2].plot(np.arange(self.pred_joint_list.shape[0]), self.pred_joint_list[:, joint_idx] * self.joint_scales[joint_idx])
            self.ax[0, 2].set_xlabel("Step", fontsize=20)
            self.ax[0, 2].set_title("Joint", fontsize=20)
            self.ax[0, 2].tick_params(axis="x", labelsize=16)
            self.ax[0, 2].tick_params(axis="y", labelsize=16)
            self.ax[0, 2].axis("on")

            self.fig.tight_layout()
            self.canvas.draw()
            policy_image = np.asarray(self.canvas.buffer_rgba())
            cv2.imshow("Policy image", cv2.cvtColor(policy_image, cv2.COLOR_RGB2BGR))
