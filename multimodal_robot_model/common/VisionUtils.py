import numpy as np
import cv2

def convertDepthImageToColorImage(image):
    """Convert depth image (float type) to color image (uint8 type)."""
    eps = 1e-6
    image = (255 * ((image - image.min()) / (image.max() - image.min() + eps))).astype(np.uint8)
    return cv2.merge((image,) * 3)

def convertDepthImageToPointCloud(depth_image, fovy, rgb_image=None, dist_thre=None):
    """Convert depth image (float type) to point cloud (array of 3D position)."""
    focal_scaling = (1.0 / np.tan(np.deg2rad(fovy) / 2.0)) * depth_image.shape[0] / 2.0
    xyz_array = np.array([(i, j) for i in range(depth_image.shape[0]) for j in range(depth_image.shape[1])], dtype=np.float32)
    xyz_array = (xyz_array - 0.5 * np.array(depth_image.shape[:2], dtype=np.float32)) / focal_scaling
    xyz_array *= depth_image.flatten()[:, np.newaxis]
    xyz_array = np.hstack((xyz_array[:, [1, 0]], depth_image.flatten()[:, np.newaxis]))
    if dist_thre:
        dist_thre_indices = np.argwhere(depth_image.flatten() < dist_thre)[:, 0]
        xyz_array = xyz_array[dist_thre_indices]
        if rgb_image is not None:
            rgb_array = rgb_image.reshape(-1, 3)[dist_thre_indices]
    if rgb_image is None:
        return xyz_array
    else:
        rgb_array = rgb_array.astype(np.float32) / 255.0
        return xyz_array, rgb_array
