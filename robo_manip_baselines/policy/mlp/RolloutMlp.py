import cv2
import matplotlib.pylab as plt
import numpy as np
import torch

from robo_manip_baselines.common import RolloutBase, denormalize_data, normalize_data

from .MlpPolicy import MlpPolicy


class RolloutMlp(RolloutBase):
    def setup_policy(self):
        # Print policy information
        self.print_policy_info()

        # Construct policy
        self.policy = MlpPolicy(
            self.state_dim,
            self.action_dim,
            len(self.camera_names),
            **self.model_meta_info["policy"]["args"],
        )

        # Load checkpoint
        self.load_ckpt()

    def setup_plot(self):
        fig_ax = plt.subplots(
            2,
            len(self.camera_names),
            figsize=(13.5, 6.0),
            dpi=60,
            squeeze=False,
            constrained_layout=True,
        )
        super().setup_plot(fig_ax)

    def setup_variables(self):
        super().setup_variables()

        self.state_buf = None
        self.images_buf = None

    def get_buffered_state(self):
        # Get latest value
        state = np.concatenate(
            [
                self.motion_manager.get_data(state_key, self.obs)
                for state_key in self.state_keys
            ]
        )
        state = normalize_data(state, self.model_meta_info["state"])
        state = torch.tensor(state, dtype=torch.float32)

        # Store and return
        if self.state_buf is None:
            self.state_buf = [
                state for _ in range(self.model_meta_info["data"]["n_obs_steps"])
            ]
        else:
            self.state_buf.pop(0)
            self.state_buf.append(state)

        state = torch.stack(self.state_buf, dim=0)[torch.newaxis].to(self.device)

        return state

    def get_buffered_images(self):
        # Get latest value
        images = self.get_images()
        # Store and return
        if self.images_buf is None:
            self.images_buf = [
                [image for _ in range(self.model_meta_info["data"]["n_obs_steps"])]
                for image in images
            ]
        else:
            for single_images_buf, image in zip(self.images_buf, images):
                single_images_buf.pop(0)
                single_images_buf.append(image)

        images = [
            torch.stack(single_images_buf, dim=0)[torch.newaxis].to(self.device)
            for single_images_buf in self.images_buf
        ]

        return images[0]

    def infer_policy(self):
        if self.model_meta_info["data"]["n_obs_steps"] > 1:
            state = self.get_buffered_state()
            images = self.get_buffered_images()
        else:
            state = self.get_state()
            images = self.get_images()
        action = (
            self.policy(state, images)[0][0]
            if self.policy.horizon > 1
            else self.policy(state, images)[0]
        )
        action = action.cpu().detach().numpy().astype(np.float64)
        self.policy_action = denormalize_data(action, self.model_meta_info["action"])
        self.policy_action_list = np.concatenate(
            [self.policy_action_list, self.policy_action[np.newaxis]]
        )

    def draw_plot(self):
        # Clear plot
        for _ax in np.ravel(self.ax):
            _ax.cla()
            _ax.axis("off")

        # Plot images
        self.plot_images(self.ax[0, 0 : len(self.camera_names)])

        # Plot action
        self.plot_action(self.ax[1, 0])

        # Finalize plot
        self.canvas.draw()
        cv2.imshow(
            self.policy_name,
            cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGB2BGR),
        )
