import torch
from eipl.data import MultimodalDataset


class RmbSarnnDatasetWithMask(MultimodalDataset):
    """
    This class is used to train models that deal with multimodal data (e.g., images, joints), such as CNNRNN/SARNN.

    Args:
        images (numpy array): Set of images in the dataset, expected to be a 5D array [data_num, seq_num, channel, height, width].
        joints (numpy array): Set of joints in the dataset, expected to be a 3D array [data_num, seq_num, joint_dim].
        masks (numpy array): Set of masks in the dataset, expected to be a 2D array [data_num, seq_num].
        stdev (float, optional): Set the standard deviation for normal distribution to generate noise.
    """

    def __init__(self, images, joints, masks, device="cpu", stdev=None):
        """
        The constructor of Multimodal Dataset class. Initializes the images, joints, and transformation.

        Args:
            images (numpy array): The images data, expected to be a 5D array [data_num, seq_num, channel, height, width].
            joints (numpy array): The joints data, expected to be a 3D array [data_num, seq_num, joint_dim].
            masks (numpy array): Set of masks in the dataset, expected to be a 2D array [data_num, seq_num].
            stdev (float, optional): The standard deviation for the normal distribution to generate noise. Defaults to 0.02.
        """
        super().__init__(images, joints, device, stdev)
        self.masks = torch.Tensor(masks).to(self.device)

    def __getitem__(self, idx):
        """
        Extraction and preprocessing of images and joints at the specified indexes.

        Args:
            idx (int): The index of the element.

        Returns:
            dataset (list): A list containing lists of transformed and noise added image and joint (x_img, x_joint) and the original image and joint (y_img, y_joint).
        """
        ret = super().__getitem__(idx)
        ret.append(self.masks[idx])
        return ret
