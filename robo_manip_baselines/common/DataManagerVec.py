import os
import numpy as np
import h5py
from .DataManager import MotionStatus, DataKey, DataManager


class DataManagerVec(DataManager):
    """Data manager with vectorization."""

    def reset(self):
        """Reset."""
        self.status = MotionStatus(0)

        self.all_data_seq_list = [{} for env_idx in range(self.env.unwrapped.num_envs)]

    def append_single_data(self, key, data_list):
        """Append a single data to the data sequence."""
        key = DataKey.replace_deprecated_key(key)  # For backward compatibility
        for all_data_seq, data in zip(self.all_data_seq_list, data_list):
            if key not in all_data_seq:
                all_data_seq[key] = []
            all_data_seq[key].append(data)

    def get_single_data(self, key, time_idx):
        """Get a single data from the data sequence."""
        key = DataKey.replace_deprecated_key(key)  # For backward compatibility
        data_list = []
        for all_data_seq in self.all_data_seq_list:
            data = all_data_seq[key][time_idx]
            data_list.append(data)
        return data_list

    def get_data(self, key):
        """Get a data sequence."""
        key = DataKey.replace_deprecated_key(key)  # For backward compatibility
        data_seq_list = []
        for all_data_seq in self.all_data_seq_list:
            data_seq = all_data_seq[key]
            data_seq_list.append(data_seq)
        return data_seq_list

    def save_data(self, filename_list):
        """Save data."""
        # For backward compatibility
        for orig_key in self.all_data_seq_list[0].keys():
            new_key = DataKey.replace_deprecated_key(orig_key)
            if orig_key != new_key:
                for all_data_seq in self.all_data_seq_list:
                    all_data_seq[new_key] = all_data_seq.pop(orig_key)

        for all_data_seq, filename in zip(self.all_data_seq_list, filename_list):
            if filename is None:
                continue

            all_data_seq.update(self.general_info)
            all_data_seq.update(self.world_info)
            all_data_seq.update(self.camera_info)

            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with h5py.File(filename, mode="w") as f:
                for key in all_data_seq.keys():
                    if isinstance(all_data_seq[key], list):
                        f.create_dataset(key, data=np.array(all_data_seq[key]))
                    elif isinstance(all_data_seq[key], np.ndarray):
                        f.create_dataset(key, data=all_data_seq[key])
                    else:
                        f.attrs[key] = all_data_seq[key]

        self.data_idx += 1

    def load_data(self, filename_list):
        """Load data."""
        self.all_data_seq_list = [{} for file_idx in range(len(filename_list))]
        for all_data_seq, filename in zip(self.all_data_seq_list, filename_list):
            with h5py.File(filename, mode="r") as f:
                for orig_key in f.keys():
                    new_key = DataKey.replace_deprecated_key(
                        orig_key
                    )  # For backward compatibility
                    all_data_seq[new_key] = f[orig_key][()]
                for orig_key in f.attrs.keys():
                    new_key = DataKey.replace_deprecated_key(
                        orig_key
                    )  # For backward compatibility
                    all_data_seq[new_key] = f.attrs[orig_key]
