import os
import numpy as np
import cv2
from .DataManager import MotionStatus, DataKey, DataManager

class DataManagerVec(DataManager):
    """Data manager with vectorization."""

    def reset(self):
        """Reset."""
        self.status = MotionStatus(0)

        self.all_data_seq_list = [{} for env_idx in range(self.env.unwrapped.num_envs)]

    def appendSingleData(self, key, data_list):
        """Append a single data to the data sequence."""
        key = DataKey.replaceDeprecatedKey(key) # For backward compatibility
        for all_data_seq, data in zip(self.all_data_seq_list, data_list):
            if key not in all_data_seq:
                all_data_seq[key] = []
            all_data_seq[key].append(data)

    def getSingleData(self, key, time_idx):
        """Get a single data from the data sequence."""
        key = DataKey.replaceDeprecatedKey(key) # For backward compatibility
        data_list = []
        for all_data_seq in self.all_data_seq_list:
            data = all_data_seq[key][time_idx]
            if "rgb_image" in key:
                if data.ndim == 1:
                    data = cv2.imdecode(data, flags=cv2.IMREAD_COLOR)
            elif ("depth_image" in key) and ("fov" not in key):
                if data.ndim == 1:
                    data = cv2.imdecode(data, flags=cv2.IMREAD_UNCHANGED)
            data_list.append(data)
        return data_list

    def getData(self, key):
        """Get a data sequence."""
        key = DataKey.replaceDeprecatedKey(key) # For backward compatibility
        data_seq_list = []
        for all_data_seq in self.all_data_seq_list:
            data_seq = all_data_seq[key]
            if "rgb_image" in key:
                if data_seq[0].ndim == 1:
                    data_seq = np.array([cv2.imdecode(data, flags=cv2.IMREAD_COLOR) for data in data_seq])
            elif ("depth_image" in key) and ("fov" not in key):
                if data_seq[0].ndim == 1:
                    data_seq = np.array([cv2.imdecode(data, flags=cv2.IMREAD_UNCHANGED) for data in data_seq])
            data_seq_list.append(data_seq)
        return data_seq_list

    def compressData(self, key, compress_flag, filter_list=None):
        """Compress data."""
        key = DataKey.replaceDeprecatedKey(key) # For backward compatibility
        for data_idx, all_data_seq in enumerate(self.all_data_seq_list):
            if (filter_list is not None) and (not filter_list[data_idx]):
                continue
            for time_idx, data in enumerate(all_data_seq[key]):
                if compress_flag == "jpg":
                    all_data_seq[key][time_idx] = cv2.imencode(".jpg", data, (cv2.IMWRITE_JPEG_QUALITY, 95))[1]
                elif compress_flag == "exr":
                    all_data_seq[key][time_idx] = cv2.imencode(".exr", data)[1]

    def saveData(self, filename_list):
        """Save data."""
        # For backward compatibility
        for orig_key in self.all_data_seq_list[0].keys():
            new_key = DataKey.replaceDeprecatedKey(orig_key)
            if orig_key != new_key:
                for all_data_seq in self.all_data_seq_list:
                    all_data_seq[new_key] = all_data_seq.pop(orig_key)

        # If each element has a different shape, save it as an object array
        for all_data_seq in self.all_data_seq_list:
            for key in all_data_seq.keys():
                if isinstance(all_data_seq[key], list) and \
                   len({data.shape if isinstance(data, np.ndarray) else None for data in all_data_seq[key]}) > 1:
                    all_data_seq[key] = np.array(all_data_seq[key], dtype=object)

        for all_data_seq, filename in zip(self.all_data_seq_list, filename_list):
            if filename is None:
                continue
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            np.savez(filename, **all_data_seq, **self.general_info, **self.world_info, **self.camera_info)

        self.data_idx += 1

    def loadData(self, filename_list):
        """Load data."""
        self.all_data_seq_list = [{} for file_idx in range(len(filename_list))]
        for all_data_seq, filename in zip(self.all_data_seq_list, filename_list):
            npz_data = np.load(filename, allow_pickle=True)
            for orig_key in npz_data.keys():
                new_key = DataKey.replaceDeprecatedKey(orig_key) # For backward compatibility
                all_data_seq[new_key] = np.copy(npz_data[orig_key])
