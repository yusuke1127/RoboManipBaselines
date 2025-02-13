import h5py
import numpy as np
import torch
from torchvision.transforms import v2

from robo_manip_baselines.common import DataKey, get_skipped_single_data, normalize_data


class MlpDataset(torch.utils.data.Dataset):
    """Dataset to train MLP policy."""

    def __init__(
        self,
        filenames,
        model_meta_info,
    ):
        super().__init__()

        self.filenames = filenames
        self.model_meta_info = model_meta_info

        self.skip = self.model_meta_info["data"]["skip"]

        self.accum_step_idxes = []
        for filename in self.filenames:
            with h5py.File(filename, "r") as h5file:
                episode_len = h5file[DataKey.TIME][:: self.skip].shape[0]
                if len(self.accum_step_idxes) == 0:
                    self.accum_step_idxes.append(episode_len)
                else:
                    self.accum_step_idxes.append(
                        self.accum_step_idxes[-1] + episode_len
                    )

        self.image_transforms = v2.Compose(
            [
                v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05),
                v2.RandomAffine(degrees=4.0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
                v2.ToDtype(torch.float32, scale=True),
                v2.GaussianNoise(sigma=0.1),
            ]
        )

    def __len__(self):
        return self.accum_step_idxes[-1]

    def __getitem__(self, step_idx_in_whole):
        episode_idx = (self.accum_step_idxes > step_idx_in_whole).nonzero()[0][0]
        step_idx_in_episode = step_idx_in_whole
        if episode_idx > 0:
            step_idx_in_episode -= self.accum_step_idxes[episode_idx - 1]

        with h5py.File(self.filenames[episode_idx], "r") as h5file:
            episode_len = h5file[DataKey.TIME][:: self.skip].shape[0]
            assert step_idx_in_episode < episode_len

            # Load state
            if len(self.model_meta_info["state"]["keys"]) == 0:
                state = np.zeros(0, dtype=np.float32)
            else:
                state = np.concatenate(
                    [
                        get_skipped_single_data(
                            h5file[key], step_idx_in_episode * self.skip, key, self.skip
                        )
                        for key in self.model_meta_info["state"]["keys"]
                    ]
                )

            # Load action
            action = np.concatenate(
                [
                    get_skipped_single_data(
                        h5file[key], step_idx_in_episode * self.skip, key, self.skip
                    )
                    for key in self.model_meta_info["action"]["keys"]
                ]
            )

            # Load images
            images = np.stack(
                [
                    h5file[DataKey.get_rgb_image_key(camera_name)][
                        step_idx_in_episode * self.skip
                    ]
                    for camera_name in self.model_meta_info["image"]["camera_names"]
                ],
                axis=0,
            )

        # Pre-convert data
        state = normalize_data(state, self.model_meta_info["state"])
        action = normalize_data(action, self.model_meta_info["action"])
        images = np.einsum("k h w c -> k c h w", images)

        # Convert to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_tensor = torch.tensor(action, dtype=torch.float32)
        images_tensor = torch.tensor(images, dtype=torch.uint8)

        # Augment data
        if "aug_std" in self.model_meta_info["state"]:
            state_tensor += self.model_meta_info["state"]["aug_std"] * torch.randn_like(
                state_tensor
            )
        if "aug_std" in self.model_meta_info["action"]:
            action_tensor += self.model_meta_info["action"][
                "aug_std"
            ] * torch.randn_like(action_tensor)
        images_tensor = self.image_transforms(images_tensor)

        return state_tensor, action_tensor, images_tensor
