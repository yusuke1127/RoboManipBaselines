import os
import warnings
import numpy as np
import cv2
from enum import Enum
from multimodal_robot_model import __version__

# https://github.com/opencv/opencv/issues/21326
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

class MotionStatus(Enum):
    """Motion status."""
    INITIAL = 0
    PRE_REACH = 1
    REACH = 2
    GRASP = 3
    TELEOP = 4
    END = 5

class DataKey(object):
    """Data key."""
    TIME = "time"

    MEASURED_JOINT_POS = "measured_joint_pos"
    COMMAND_JOINT_POS = "command_joint_pos"

    MEASURED_JOINT_VEL = "measured_joint_vel"
    COMMAND_JOINT_VEL = "command_joint_vel"

    MEASURED_JOINT_TORQUE = "measured_joint_torque"
    COMMAND_JOINT_TORQUE = "command_joint_torque"

    MEASURED_EEF_POSE = "measured_eef_pose"
    COMMAND_EEF_POSE = "command_eef_pose"

    MEASURED_EEF_VEL = "measured_eef_vel"
    COMMAND_EEF_VEL = "command_eef_vel"

    MEASURED_WRENCH = "measured_wrench"
    COMMAND_WRENCH = "command_wrench"

    @classmethod
    def getRgbImageKey(cls, camera_name):
        """Get the rgb image key from the camera name."""
        return camera_name.lower() + "_rgb_image"

    @classmethod
    def getDepthImageKey(cls, camera_name):
        """Get the depth image key from the camera name."""
        return camera_name.lower() + "_depth_image"

    @classmethod
    def replaceDeprecatedKey(cls, orig_key):
        """Replace a deprecated key with a new key for backward compatibility."""
        if orig_key == "joint_pos":
            new_key = DataKey.MEASURED_JOINT_POS
        elif orig_key == "joint_vel":
            new_key = DataKey.MEASURED_JOINT_VEL
        elif orig_key == "wrench":
            new_key = DataKey.MEASURED_WRENCH
        elif orig_key == "measured_eef":
            new_key = DataKey.MEASURED_EEF_POSE
        elif orig_key == "command_eef":
            new_key = DataKey.COMMAND_EEF_POSE
        elif orig_key == "action":
            new_key = DataKey.COMMAND_JOINT_POS
        else:
            new_key = orig_key.lower()
        if orig_key != new_key:
            warnings.warn(f"[DataKey] \"{orig_key}\" is deprecated, use \"{new_key}\" instead.")
        return new_key

class DataManager(object):
    """Data manager."""

    def __init__(self, env):
        self.env = env

        self.general_info = {"version": __version__}

        self.data_idx = 0
        self.world_idx = 0
        self.world_info = {}

        self.camera_info = {}

        self.reset()

    def reset(self):
        """Reset."""
        self.status = MotionStatus(0)

        self.all_data_seq = {}

    def appendSingleData(self, key, data):
        """Append a single data to the data sequence."""
        key = DataKey.replaceDeprecatedKey(key) # For backward compatibility
        if key not in self.all_data_seq:
            self.all_data_seq[key] = []
        self.all_data_seq[key].append(data)

    def getSingleData(self, key, time_idx):
        """Get a single data from the data sequence."""
        key = DataKey.replaceDeprecatedKey(key) # For backward compatibility
        data = self.all_data_seq[key][time_idx]
        if "rgb_image" in key:
            if data.ndim == 1:
                data = cv2.imdecode(data, flags=cv2.IMREAD_COLOR)
        elif ("depth_image" in key) and ("fov" not in key):
            if data.ndim == 1:
                data = cv2.imdecode(data, flags=cv2.IMREAD_UNCHANGED)
        return data

    def getData(self, key):
        """Get a data sequence."""
        key = DataKey.replaceDeprecatedKey(key) # For backward compatibility
        data_seq = self.all_data_seq[key]
        if "rgb_image" in key:
            if data_seq[0].ndim == 1:
                data_seq = np.array([cv2.imdecode(data, flags=cv2.IMREAD_COLOR) for data in data_seq])
        elif ("depth_image" in key) and ("fov" not in key):
            if data_seq[0].ndim == 1:
                data_seq = np.array([cv2.imdecode(data, flags=cv2.IMREAD_UNCHANGED) for data in data_seq])
        return data_seq

    def compressData(self, key, compress_flag):
        """Compress data."""
        key = DataKey.replaceDeprecatedKey(key) # For backward compatibility
        for time_idx, data in enumerate(self.all_data_seq[key]):
            if compress_flag == "jpg":
                self.all_data_seq[key][time_idx] = cv2.imencode(".jpg", data, (cv2.IMWRITE_JPEG_QUALITY, 95))[1]
            elif compress_flag == "exr":
                self.all_data_seq[key][time_idx] = cv2.imencode(".exr", data)[1]

    def saveData(self, filename):
        """Save data."""
        # For backward compatibility
        for orig_key in self.all_data_seq.keys():
            new_key = DataKey.replaceDeprecatedKey(orig_key)
            if orig_key != new_key:
                self.all_data_seq[new_key] = self.all_data_seq.pop(orig_key)

        # If each element has a different shape, save it as an object array
        for key in self.all_data_seq.keys():
            if isinstance(self.all_data_seq[key], list) and \
               len({data.shape if isinstance(data, np.ndarray) else None for data in self.all_data_seq[key]}) > 1:
                self.all_data_seq[key] = np.array(self.all_data_seq[key], dtype=object)

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.savez(filename, **self.all_data_seq, **self.general_info, **self.world_info, **self.camera_info)
        self.data_idx += 1

    def loadData(self, filename):
        """Load data."""
        self.all_data_seq = {}
        npz_data = np.load(filename, allow_pickle=True)
        for orig_key in npz_data.keys():
            new_key = DataKey.replaceDeprecatedKey(orig_key) # For backward compatibility
            self.all_data_seq[new_key] = np.copy(npz_data[orig_key])

    def goToNextStatus(self):
        """Go to the next status."""
        if self.status == MotionStatus(len(MotionStatus) - 1):
            raise ValueError("Cannot go from the last status to the next.")
        self.status = MotionStatus(self.status.value + 1)

    def getStatusImage(self):
        """Get the image corresponding to the current status."""
        status_image = np.zeros((50, 160, 3), dtype=np.uint8)
        if self.status == MotionStatus.INITIAL:
            status_image[:, :] = np.array([200, 255, 200])
        elif self.status in (MotionStatus.PRE_REACH, MotionStatus.REACH, MotionStatus.GRASP):
            status_image[:, :] = np.array([255, 255, 200])
        elif self.status == MotionStatus.TELEOP:
            status_image[:, :] = np.array([255, 200, 200])
        elif self.status == MotionStatus.END:
            status_image[:, :] = np.array([200, 200, 255])
        else:
            raise ValueError("Unknown status: {}".format(self.status))
        cv2.putText(status_image, self.status.name, (5, 35), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 2)
        return status_image

    def setupSimWorld(self, world_idx=None):
        """Setup the simulation world."""
        if world_idx is None:
            kwargs = {"cumulative_idx": self.data_idx}
        else:
            kwargs = {"world_idx": world_idx}
        self.world_idx = self.env.unwrapped.modify_world(**kwargs)
        self.world_info = {"world_idx": self.world_idx}

    def setupCameraInfo(self):
        """Set camera info."""
        for camera_name in self.env.unwrapped.camera_names:
            depth_key = DataKey.getDepthImageKey(camera_name)
            self.camera_info[depth_key + "_fovy"] = self.env.unwrapped.get_camera_fovy(camera_name)

    @property
    def status(self):
        """Get the status."""
        return self._status

    @status.setter
    def status(self, new_status):
        """Set the status."""
        self._status = new_status
        if self.env is None:
            self.status_start_time = 0.0
        else:
            self.status_start_time = self.env.unwrapped.get_sim_time()

    @property
    def status_elapsed_duration(self):
        """Get the elapsed duration of the current status."""
        return self.env.unwrapped.get_sim_time() - self.status_start_time
