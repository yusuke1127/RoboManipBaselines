import numpy as np
import torch

from robo_manip_baselines.common import (
    DataKey,
    DatasetBase,
    RmbData,
    get_skipped_data_seq,
    get_skipped_single_data,
)


class MlpDataset(DatasetBase):
    """Dataset to train MLP policy."""

    def setup_variables(self):
        skip = self.model_meta_info["data"]["skip"]
        pad_before = self.model_meta_info["data"]["n_obs_steps"] - 1
        pad_after = self.model_meta_info["data"]["n_action_steps"] - 1
        self.accum_step_idxes = None

        # Switch single load mode and sequential load mode.
        # Single load
        if pad_before <= 0 and pad_after <= 0:
            self.accum_step_idxes = []
            for filename in self.filenames:
                with RmbData(filename) as rmb_data:
                    episode_len = rmb_data[DataKey.TIME][::skip].shape[0]
                    if len(self.accum_step_idxes) == 0:
                        self.accum_step_idxes.append(episode_len)
                    else:
                        self.accum_step_idxes.append(
                            self.accum_step_idxes[-1] + episode_len
                        )
            self.accum_step_idxes = np.array(self.accum_step_idxes)

        # Sequential load
        else:
            horizon = self.model_meta_info["data"]["horizon"]
            self.chunk_into_list = []
            for episode_idx, filename in enumerate(self.filenames):
                with RmbData(filename) as rmb_data:
                    episode_len = rmb_data[DataKey.TIME][::skip].shape[0]
                    for start_time_idx in range(
                        -1 * pad_before, episode_len - (horizon - 1) + pad_after
                    ):
                        self.chunk_info_list.append((episode_idx, start_time_idx))

    def __len__(self):
        return (
            self.accum_step_idxes[-1]
            if self.accum_step_idxes is not None
            else len(self.chunk_into_list)
        )

    def __getitem__(self, data_idx):
        if self.accum_step_idxes is not None:
            return self._get_single_data(data_idx)
        else:
            return self._get_sequential_data(data_idx)

    def _get_single_data(self, step_idx_in_whole):
        # Set episode_idx and step_idx_in_episode
        skip = self.model_meta_info["data"]["skip"]
        episode_idx = (self.accum_step_idxes > step_idx_in_whole).nonzero()[0][0]
        step_idx_in_episode = step_idx_in_whole
        if episode_idx > 0:
            step_idx_in_episode -= self.accum_step_idxes[episode_idx - 1]

        with RmbData(self.filenames[episode_idx], self.enable_rmb_cache) as rmb_data:
            episode_len = rmb_data[DataKey.TIME][::skip].shape[0]
            assert 0 <= step_idx_in_episode < episode_len

            # Load state
            if len(self.model_meta_info["state"]["keys"]) == 0:
                state = np.zeros(0, dtype=np.float32)
            else:
                state = np.concatenate(
                    [
                        get_skipped_single_data(
                            rmb_data[key], step_idx_in_episode * skip, key, skip
                        )
                        for key in self.model_meta_info["state"]["keys"]
                    ]
                )

            # Load action
            action = np.concatenate(
                [
                    get_skipped_single_data(
                        rmb_data[key], step_idx_in_episode * skip, key, skip
                    )
                    for key in self.model_meta_info["action"]["keys"]
                ]
            )

            # Load images
            image_keys = [
                DataKey.get_rgb_image_key(camera_name)
                for camera_name in self.model_meta_info["image"]["camera_names"]
            ]
            images = np.stack(
                [
                    # This allows for a common hash of cache
                    rmb_data[key][::skip][step_idx_in_episode]
                    if self.enable_rmb_cache
                    # This allows for minimal loading when reading from HDF5
                    else rmb_data[key][step_idx_in_episode * skip]
                    for key in image_keys
                ],
                axis=0,
            )

        # Pre-convert data
        state, action, images = self.pre_convert_data(state, action, images)

        # Convert to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_tensor = torch.tensor(action, dtype=torch.float32)
        images_tensor = torch.tensor(images, dtype=torch.uint8)

        # Augment data
        state_tensor, action_tensor, images_tensor = self.augment_data(
            state_tensor, action_tensor, images_tensor
        )

        # Sort in the order of policy inputs and outputs
        return state_tensor, images_tensor, action_tensor

    def _get_sequential_data(self, chunk_idx):
        skip = self.model_meta_info["data"]["skip"]
        horizon = self.model_meta_info["data"]["horizon"]
        episode_idx, start_time_idx = self.chunk_info_list[chunk_idx]

        with RmbData(self.filenames[episode_idx], self.enable_rmb_cache) as rmb_data:
            episode_len = rmb_data[DataKey.TIME][::skip].shape[0]
            time_idxes = np.clip(
                np.arange(start_time_idx, start_time_idx + horizon), 0, episode_len - 1
            )

            # Load state
            if len(self.model_meta_info["state"]["keys"]) == 0:
                state = np.zeros(0, dtype=np.float64)
            else:
                state = np.concatenate(
                    [
                        get_skipped_data_seq(rmb_data[key][:], key, skip)[time_idxes]
                        for key in self.model_meta_info["state"]["keys"]
                    ],
                    axis=1,
                )

            # Load action
            action = np.concatenate(
                [
                    get_skipped_data_seq(rmb_data[key][:], key, skip)[time_idxes]
                    for key in self.model_meta_info["action"]["keys"]
                ],
                axis=1,
            )

            # Load images
            images = np.stack(
                [
                    rmb_data[DataKey.get_rgb_image_key(camera_name)][::skip][time_idxes]
                    for camera_name in self.model_meta_info["image"]["camera_names"]
                ],
                axis=0,
            )

        # Pre-convert data
        state, action, images = self.pre_convert_data(state, action, images)

        # Convert to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_tensor = torch.tensor(action, dtype=torch.float32)
        images_tensor = torch.tensor(images, dtype=torch.uint8)

        # Augment data
        state_tensor, action_tensor, images_tensor = self.augment_data(
            state_tensor, action_tensor, images_tensor
        )

        # Sort in the order of policy inputs and outputs
        return state_tensor, images_tensor, action_tensor
