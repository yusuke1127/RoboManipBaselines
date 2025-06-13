import argparse
import glob
import os

import cv2
import h5py
import numpy as np
import pytorch3d.ops as torch3d_ops
import torch
from tqdm import tqdm

from robo_manip_baselines.common import (
    DataKey,
    RmbData,
    convert_depth_image_to_point_cloud,
)


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("path", type=str, help="path to data (*.hdf5 or *.rmb)")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="whether to overwrite existing value if it exists",
    )
    parser.add_argument(
        "--min_bound",
        type=float,
        nargs=3,
        default=[-3, -3, -3],
        help="Min bounding box for cropping pointcloud before downsampling.",
    )
    parser.add_argument(
        "--max_bound",
        type=float,
        nargs=3,
        default=[3, 3, 3],
        help="Max bounding box for cropping pointcloud before downsampling.",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=512,
        help="number of points in one pointcloud.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=[320, 240],
        help="Image size (width, height) to be resized before crop. In the case of multiple image inputs, it is assumed that all images share the same size.",
    )

    return parser.parse_args()


def farthest_point_sampling(pointcloud: np.ndarray, num_points: int = 512):
    # Downsample point cloud with FPS.
    points = torch.from_numpy(pointcloud).unsqueeze(0)
    n_points = torch.tensor([num_points])
    _, sampled_indices = torch3d_ops.sample_farthest_points(points, K=n_points)
    pointcloud = pointcloud[sampled_indices.squeeze(0).numpy()]
    return pointcloud


def crop_pointcloud(pointcloud: np.ndarray, min_bound=None, max_bound=None):
    pc_xyz = pointcloud[:, :3]
    if min_bound is not None:
        mask = np.all(pc_xyz > min_bound, axis=1)
        pointcloud = pointcloud[mask]
        pc_xyz = pc_xyz[mask]
    if max_bound is not None:
        mask = np.all(pc_xyz < max_bound, axis=1)
        pointcloud = pointcloud[mask]
    return pointcloud


class AddPointCloudtoRmbData:
    def __init__(
        self,
        path,
        image_size,
        overwrite=False,
        min_bound=None,
        max_bound=None,
        num_points=512,
    ):
        self.path = path.rstrip("/")
        self.overwrite = overwrite
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.n_points = num_points
        self.image_size = image_size
        self.hdf5_paths, self.rmb_paths = self.resolve_hdf5_and_rmb_path(self.path)

    def get_pointcloud(
        self, rmb_file_path: str, min_bound=None, max_bound=None, num_points: int = 512
    ):
        pointclouds = {}
        with RmbData(rmb_file_path) as rmb:
            camera_names = rmb.attrs["camera_names"]
            # Load image
            images = np.stack(
                [
                    rmb[DataKey.get_rgb_image_key(camera_name)][:]
                    for camera_name in camera_names
                ],
                axis=0,
            )
            # Load depth
            depthes = np.stack(
                [
                    rmb[DataKey.get_depth_image_key(camera_name)][:]
                    for camera_name in camera_names
                ],
                axis=0,
            )
            # Load fovy
            fovies = [
                rmb.attrs[DataKey.get_depth_image_key(camera_name) + "_fovy"]
                for camera_name in camera_names
            ]

        # Resize images
        K, T, H, W, C = images.shape
        image_size = self.image_size
        images = np.array(
            [cv2.resize(img, image_size) for img in images.reshape(-1, H, W, C)]
        ).reshape(K, T, *image_size[::-1], C)

        # Resize depthes
        K, T, H, W = depthes.shape
        image_size = self.image_size
        depthes = np.array(
            [cv2.resize(dpth, image_size) for dpth in depthes.reshape(-1, H, W)]
        ).reshape(K, T, *image_size[::-1])

        for image, depth, fovy, camera_name in zip(
            images, depthes, fovies, camera_names
        ):
            # Generate pointclouds
            pointcloud = []
            for i_t, d_t in zip(image, depth):
                pc_t = np.concat(
                    convert_depth_image_to_point_cloud(
                        d_t[::10, ::10], fovy, i_t[::10, ::10]
                    ),
                    axis=1,
                )
                pc_t = crop_pointcloud(pc_t, min_bound, max_bound)
                pc_t = farthest_point_sampling(pc_t, num_points)
                pointcloud.append(pc_t.tolist())
            # Add pointclouds to RMB
            pointclouds[camera_name + "_pointcloud"] = pointcloud
        return pointclouds

    def resolve_hdf5_and_rmb_path(self, path):
        hdf5_list = []
        rmb_list = []
        if path.endswith(".rmb"):
            hdf5_list.append(os.path.join(path, "main.rmb.hdf5"))
            rmb_list.append(path)
        elif path.endswith(".hdf5"):
            hdf5_list.append(path)
            rmb_list.append(str(path).rstrip("/main.rmb.hdf5"))
        elif os.path.isdir(path):
            rmb_dirs = glob.glob(os.path.join(path, "**", "*.rmb"), recursive=True)
            if not rmb_dirs:
                raise ValueError(
                    f"[{self.__class__.__name__}] No '*.rmb' directories found under the given "
                    f"path: {path}"
                )
            for rmb in rmb_dirs:
                hdf5_path = os.path.join(rmb, "main.rmb.hdf5")
                if not os.path.exists(hdf5_path):
                    raise FileNotFoundError(
                        f"[{self.__class__.__name__}] HDF5 file not found: {hdf5_path}"
                    )
                hdf5_list.append(hdf5_path)
                rmb_list.append(rmb)
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Unsupported file extension: {path}"
            )

        return hdf5_list, rmb_list

    def run(self):
        print(
            f"[{self.__class__.__name__}] Generate pointcloud from RGB and depth images and add it."
        )
        for hdf5_path, rmb_path in tqdm(zip(self.hdf5_paths, self.rmb_paths)):
            tqdm.write(f"[{self.__class__.__name__}] Open {hdf5_path}")
            with h5py.File(hdf5_path, "r+") as f:
                if any(["pointcloud" in k for k in f.keys()]) and not self.overwrite:
                    print(
                        f"[{self.__class__.__name__}] pointcloud already exists (use --overwrite to replace)"
                    )
                    continue

                pcs = self.get_pointcloud(
                    rmb_path, self.min_bound, self.max_bound, self.n_points
                )
                tqdm.write(f"[{self.__class__.__name__}] Add pointcloud to {rmb_path}")
                for k in pcs.keys():
                    if k in f.keys() and self.overwrite:
                        del f[k]
                    elif k in f.keys():
                        tqdm.write(
                            f"[{self.__class__.__name__}] pointcloud already exists in {hdf5_path}. skipping."
                        )
                        continue
                    f[k] = pcs[k]


if __name__ == "__main__":
    refine = AddPointCloudtoRmbData(**vars(parse_argument()))
    refine.run()
