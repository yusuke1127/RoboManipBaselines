import argparse

import cv2
import numpy as np
import pytorch3d.ops as torch3d_ops
import torch
from tqdm import tqdm

from robo_manip_baselines.common import (
    DataKey,
    RmbData,
    convert_depth_image_to_point_cloud,
    find_rmb_files,
)


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "path",
        type=str,
        help="path to data (*.hdf5 or *.rmb) or directory containing them",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=[84, 84],
        help="Image size (width, height) to be resized before crop",
    )
    parser.add_argument(
        "--min_bound",
        type=float,
        nargs=3,
        default=[-0.4, -0.4, -0.4],
        help="min bounds of the bounding box before downsampling",
    )
    parser.add_argument(
        "--max_bound",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        help="max bounds of the bounding box before downsampling",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=512,
        help="number of points in a point cloud",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="whether to overwrite existing value if it exists",
    )

    return parser.parse_args()


def crop_pointcloud(pointcloud: np.ndarray, min_bound=None, max_bound=None):
    if min_bound is not None:
        mask = np.all(pointcloud[:, :3] > min_bound, axis=1)
        pointcloud = pointcloud[mask]
    if max_bound is not None:
        mask = np.all(pointcloud[:, :3] < max_bound, axis=1)
        pointcloud = pointcloud[mask]
    return pointcloud


def farthest_point_sampling(pointcloud: np.ndarray, num_points: int = 512):
    # Downsample point cloud with farthest point sampling (FPS)
    pointcloud_tensor = torch.from_numpy(pointcloud).unsqueeze(0)
    num_points_tensor = torch.tensor([num_points])
    _, sampled_indices = torch3d_ops.sample_farthest_points(
        pointcloud_tensor, K=num_points_tensor
    )
    pointcloud = pointcloud[sampled_indices.squeeze(0).numpy()]
    return pointcloud


class AddPointCloudToRmbData:
    def __init__(
        self,
        path,
        image_size,
        min_bound=None,
        max_bound=None,
        num_points=512,
        overwrite=False,
    ):
        self.path = path
        self.image_size = image_size
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.num_points = num_points
        self.overwrite = overwrite

    def run(self):
        print(
            f"[{self.__class__.__name__}] Store pointcloud generated from RGB and depth images."
        )
        rmb_path_list = find_rmb_files(self.path)
        for rmb_path in tqdm(rmb_path_list):
            tqdm.write(f"[{self.__class__.__name__}] Open {rmb_path}")
            with RmbData(rmb_path, mode="r+") as rmb_data:
                pc_dict = self.get_pc_dict(rmb_data)
                for pc_key in pc_dict.keys():
                    if pc_key in rmb_data.keys():
                        if self.overwrite:
                            del rmb_data.h5file[pc_key]
                        else:
                            tqdm.write(
                                f"[{self.__class__.__name__}] Pointcloud already exists: {rmb_path} (use --overwrite to replace)"
                            )
                            return
                    rmb_data.h5file[pc_key] = pc_dict[pc_key]

    def get_pc_dict(self, rmb_data):
        # Load images
        camera_names = rmb_data.attrs["camera_names"]
        rgb_image_list = np.stack(
            [
                rmb_data[DataKey.get_rgb_image_key(camera_name)][:]
                for camera_name in camera_names
            ],
            axis=0,
        )
        depth_image_list = np.stack(
            [
                rmb_data[DataKey.get_depth_image_key(camera_name)][:]
                for camera_name in camera_names
            ],
            axis=0,
        )
        fovy_list = [
            rmb_data.attrs[DataKey.get_depth_image_key(camera_name) + "_fovy"]
            for camera_name in camera_names
        ]

        # Resize images
        K, T, H, W, C = rgb_image_list.shape
        rgb_image_list = np.array(
            [
                cv2.resize(image, self.image_size)
                for image in rgb_image_list.reshape(-1, H, W, C)
            ]
        ).reshape(K, T, *self.image_size[::-1], C)
        K, T, H, W = depth_image_list.shape
        depth_image_list = np.array(
            [
                cv2.resize(image, self.image_size)
                for image in depth_image_list.reshape(-1, H, W)
            ]
        ).reshape(K, T, *self.image_size[::-1])

        # Generate pointclouds
        pc_dict = {}
        for rgb_image, depth_image, fovy, camera_name in zip(
            rgb_image_list, depth_image_list, fovy_list, camera_names
        ):
            pointcloud = []
            for single_rgb_image, single_depth_image in zip(rgb_image, depth_image):
                single_pointcloud = np.concat(
                    convert_depth_image_to_point_cloud(
                        single_depth_image, fovy, single_rgb_image
                    ),
                    axis=1,
                )
                single_pointcloud = crop_pointcloud(
                    single_pointcloud, self.min_bound, self.max_bound
                )
                single_pointcloud = farthest_point_sampling(
                    single_pointcloud, self.num_points
                )
                pointcloud.append(single_pointcloud.tolist())
            pc_dict[camera_name + "_pointcloud"] = pointcloud

        return pc_dict


if __name__ == "__main__":
    add_point_cloud = AddPointCloudToRmbData(**vars(parse_argument()))
    add_point_cloud.run()
