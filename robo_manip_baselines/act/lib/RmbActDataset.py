import h5py
import numpy as np
import torch

from robo_manip_baselines.common import DataKey, get_skipped_data_seq


class RmbActDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        filenames,
        state_keys,
        action_keys,
        camera_names,
        dataset_stats,
        skip,
        chunk_size,
    ):
        super().__init__()

        self.filenames = filenames
        self.action_keys = action_keys
        self.state_keys = state_keys
        self.camera_names = camera_names
        self.dataset_stats = dataset_stats
        self.skip = skip
        self.chunk_size = chunk_size

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, episode_idx):
        with h5py.File(self.filenames[episode_idx], "r") as h5file:
            episode_len = h5file[DataKey.TIME][:: self.skip].shape[0]
            start_time_idx = np.random.choice(episode_len)

            # Load state
            if len(self.state_keys) == 0:
                state = np.zeros(0, dtype=np.float32)
            else:
                state = np.concatenate(
                    [
                        get_skipped_data_seq(
                            h5file[state_key][
                                start_time_idx * self.skip : (start_time_idx + 1)
                                * self.skip
                            ],
                            state_key,
                            self.skip,
                        )[0]
                        for state_key in self.state_keys
                    ]
                )

            # Load action
            action = np.concatenate(
                [
                    get_skipped_data_seq(
                        h5file[action_key][start_time_idx * self.skip :],
                        action_key,
                        self.skip,
                    )
                    for action_key in self.action_keys
                ],
                axis=1,
            )

            # Load images
            images = np.stack(
                [
                    h5file[DataKey.get_rgb_image_key(camera_name)][
                        start_time_idx * self.skip
                    ]
                    for camera_name in self.camera_names
                ],
                axis=0,
            )

        # Set chunked action
        action_len = action.shape[0]
        action_chunked = np.zeros((self.chunk_size, action.shape[1]), dtype=np.float32)
        action_chunked[:action_len] = action[: self.chunk_size]
        is_pad = np.zeros(self.chunk_size, dtype=bool)
        is_pad[action_len:] = True

        # Pre-convert data
        state = (state - self.dataset_stats["state_mean"]) / self.dataset_stats[
            "state_std"
        ]
        action_chunked = (
            action_chunked - self.dataset_stats["action_mean"]
        ) / self.dataset_stats["action_std"]
        images = np.einsum("k h w c -> k c h w", images)
        images = images / 255.0

        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(action_chunked, dtype=torch.float32),
            torch.tensor(images, dtype=torch.float32),
            torch.tensor(is_pad, dtype=torch.bool),
        )
