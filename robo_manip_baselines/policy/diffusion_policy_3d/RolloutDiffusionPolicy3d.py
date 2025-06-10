import cv2
import matplotlib.pylab as plt
import numpy as np
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy_3d.policy.dp3 import DP3
from robo_manip_baselines.common import (
    RolloutBase,
    convert_depth_image_to_point_cloud,
    denormalize_data,
    normalize_data,
)


class RolloutDiffusionPolicy3d(RolloutBase):
    def setup_policy(self):
        # Print policy information
        self.print_policy_info()
        print(f"  - use ema: {self.model_meta_info['policy']['use_ema']}")
        print(
            f"  - horizon: {self.model_meta_info['data']['horizon']}, obs steps: {self.model_meta_info['data']['n_obs_steps']}, action steps: {self.model_meta_info['data']['n_action_steps']}"
        )
        print(
            f"  - image size: {self.model_meta_info['data']['image_size']}, image crop size: {self.model_meta_info['data']['image_crop_size']}"
        )

        # Construct policy
        noise_scheduler = DDPMScheduler(
            **self.model_meta_info["policy"]["noise_scheduler_args"]
        )
        self.policy = DP3(
            noise_scheduler=noise_scheduler,
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

        self.pointcloud_shape = self.model_meta_info["policy"]["args"]["shape_meta"][
            "obs"
        ]["point_cloud"]["shape"]
        self.state_buf = None
        self.pointclouds_buf = None
        self.policy_action_buf = None
        self.min_bound = self.model_meta_info["data"]["min_bound"]
        self.max_bound = self.model_meta_info["data"]["max_bound"]
        self.num_points = self.model_meta_info["data"]["num_points"]

    def infer_policy(self):
        # Infer
        if self.policy_action_buf is None or len(self.policy_action_buf) == 0:
            input_data = {}
            if len(self.state_keys) > 0:
                input_data["state"] = self.get_state()
            if len(self.camera_names) > 0:
                input_data["point_cloud"] = self.get_pointclouds()[0]
            action = self.policy.predict_action(input_data)["action"][0]
            self.policy_action_buf = list(
                action.cpu().detach().numpy().astype(np.float64)
            )

        # Store action
        self.policy_action = denormalize_data(
            self.policy_action_buf.pop(0), self.model_meta_info["action"]
        )
        self.policy_action_list = np.concatenate(
            [self.policy_action_list, self.policy_action[np.newaxis]]
        )

    def get_state(self):
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

    def get_pointclouds(self):
        # Get latest value
        pointclouds = []
        for camera_name in self.camera_names:
            image = self.info["rgb_images"][camera_name]
            depth = self.info["depth_images"][camera_name]
            fovy = self.info["fovy"][camera_name]

            image = cv2.resize(image, self.model_meta_info["data"]["image_size"])
            depth = cv2.resize(depth, self.model_meta_info["data"]["image_size"])

            image = np.moveaxis(image, -1, -3)
            image = torch.tensor(image, dtype=torch.uint8)
            image = self.image_transforms(image)
            # Adjust to a range from -1 to 1 to match the original implementation
            image = image * 2.0 - 1.0

            depth = torch.tensor(depth, dtype=torch.uint16)
            depth = self.image_transforms(depth)

            # TODO: Convert to pointcloud
            # dummy pointcloud
            pointcloud = self.farthest_point_sampling(
                self.crop_points(
                    convert_depth_image_to_point_cloud(
                        depth, fovy, image, far_clip=3.0
                    ),
                    self.min_bound,
                    self.max_bound,
                ),
                num_points=self.num_points,
            )
            pointclouds.append(torch.tensor(pointcloud, dtype=torch.float32))

        # Store and return
        if self.pointclouds_buf is None:
            self.pointclouds_buf = [
                [pointcloud for _ in range(self.model_meta_info["data"]["n_obs_steps"])]
                for pointcloud in pointclouds
            ]
        else:
            for single_pointclouds_buf, pointcloud in zip(
                self.pointclouds_buf, pointclouds
            ):
                single_pointclouds_buf.pop(0)
                single_pointclouds_buf.append(pointcloud)

        pointclouds = [
            torch.stack(single_pointclouds_buf, dim=0)[torch.newaxis].to(self.device)
            for single_pointclouds_buf in self.pointclouds_buf
        ]

        return pointclouds

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

    def crop_points(self, pointcloud: np.ndarray, min_bound=None, max_bound=None):
        # Crop pointcloud before downsampling.
        if min_bound is not None:
            mask = np.all(pointcloud[:, :3] > min_bound, axis=1)
            pointcloud = pointcloud[mask]
        if max_bound is not None:
            mask = np.all(pointcloud[:, :3] < max_bound, axis=1)
            pointcloud = pointcloud[mask]
        return pointcloud

    def farthest_point_sampling(self, pointcloud: np.ndarray, num_points: int):
        # DownSampling pointclouds.
        N, D = pointcloud.shape[:2]
        xyz = pointcloud[:, :, :3]
        centroids = np.zeros((num_points,))
        distance = np.ones((N,)) * 1e10
        farthest = np.random.randint(0, N)
        for i in range(num_points):
            centroids[i] = farthest
            centroid = xyz[farthest, :]
            dist = np.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = np.argmax(distance, -1)
        pc = pointcloud[centroids.astype(np.int32)]
        return pc
