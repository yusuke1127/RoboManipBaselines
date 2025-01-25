import cv2
import numpy as np


def convert_depth_image_to_color_image(image):
    """Convert depth image (float type) to color image (uint8 type)."""
    eps = 1e-6
    image_copied = image.copy()
    image_copied[np.logical_not(np.isfinite(image_copied))] = image_copied[
        np.isfinite(image_copied)
    ].max()
    image_copied = (
        255
        * (
            (image_copied - image_copied.min())
            / (image_copied.max() - image_copied.min() + eps)
        )
    ).astype(np.uint8)
    return cv2.merge((image_copied,) * 3)


def convert_depth_image_to_point_cloud(
    depth_image, fovy, rgb_image=None, near_clip=0.0, far_clip=np.inf
):
    """Convert depth image (float type) to point cloud (array of 3D point)."""
    focal_scaling = (1.0 / np.tan(np.deg2rad(fovy) / 2.0)) * depth_image.shape[0] / 2.0
    xyz_array = np.array(
        [
            (i, j)
            for i in range(depth_image.shape[0])
            for j in range(depth_image.shape[1])
        ],
        dtype=np.float32,
    )
    xyz_array = (
        xyz_array - 0.5 * np.array(depth_image.shape[:2], dtype=np.float32)
    ) / focal_scaling
    xyz_array *= depth_image.flatten()[:, np.newaxis]
    xyz_array = np.hstack((xyz_array[:, [1, 0]], depth_image.flatten()[:, np.newaxis]))

    if rgb_image is not None:
        rgb_array = rgb_image.reshape(-1, 3).astype(np.float32) / 255.0

    clip_indices = np.argwhere(
        (near_clip < depth_image.flatten()) & (depth_image.flatten() < far_clip)
    )[:, 0]
    xyz_array = xyz_array[clip_indices]
    if rgb_image is not None:
        rgb_array = rgb_array[clip_indices]

    if rgb_image is None:
        return xyz_array
    else:
        return xyz_array, rgb_array
