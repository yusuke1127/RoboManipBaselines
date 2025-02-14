import torch
from torchvision.transforms import v2

from robo_manip_baselines.common import normalize_data


class DatasetBase(torch.utils.data.Dataset):
    def __init__(
        self,
        filenames,
        model_meta_info,
    ):
        self.filenames = filenames
        self.model_meta_info = model_meta_info

        self.setup_image_transforms()

    def setup_image_transforms(self):
        """
        Setup image transforms.

        Image transforms should also be responsible for converting the data type from uint8 to float32 with scale (255 -> 1.0).
        """
        self.image_transforms = v2.Compose(
            [
                v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05),
                v2.RandomAffine(degrees=4.0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
                v2.ToDtype(torch.float32, scale=True),
                v2.GaussianNoise(sigma=0.1),
            ]
        )

    def pre_convert_data(self, state, action, images):
        """Pre-convert data. Arguments must be numpy arrays (not torch tensors)."""
        state = normalize_data(state, self.model_meta_info["state"])
        action = normalize_data(action, self.model_meta_info["action"])
        images = np.einsum("k h w c -> k c h w", images)

        return state, action, images

    def augment_data(self, state, action, images):
        """Augment data. Arguments must be torch tensors (not numpy arrays)."""
        if "aug_std" in self.model_meta_info["state"]:
            state += self.model_meta_info["state"]["aug_std"] * torch.randn_like(state)
        if "aug_std" in self.model_meta_info["action"]:
            action += self.model_meta_info["action"]["aug_std"] * torch.randn_like(
                action
            )
        images = self.image_transforms(images)

        return state, action, images
