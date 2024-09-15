import os
import numpy as np
import cv2
from enum import Enum

# https://github.com/opencv/opencv/issues/21326
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

class RecordStatus(Enum):
    """Status for recording."""
    INITIAL = 0
    PRE_REACH = 1
    REACH = 2
    GRASP = 3
    TELEOP = 4
    END = 5

class RecordKey(Enum):
    """Data key for recording."""
    TIME = 0
    JOINT_POS = 1
    JOINT_VEL = 2
    FRONT_RGB_IMAGE = 3
    SIDE_RGB_IMAGE = 4
    HAND_RGB_IMAGE = 5
    FRONT_DEPTH_IMAGE = 6
    SIDE_DEPTH_IMAGE = 7
    HAND_DEPTH_IMAGE = 8
    WRENCH = 9
    MEASURED_EEF = 10
    COMMAND_EEF = 11
    ACTION = 12

    def key(self):
        """Get the key of the dictionary."""
        return self.name.lower()

class RecordManager(object):
    """Recording manager for demonstrations by teleoperation."""

    def __init__(self, env):
        self.env = env

        self.data_idx = 0
        self.world_idx = 0
        self.world_info = {}

        self.camera_info = {}

        self.reset()

    def reset(self):
        """Reset recording."""
        self.status = RecordStatus(0)

        self.data_seq = {}
        for record_key in RecordKey:
            self.data_seq[record_key.key()] = []

    def appendSingleData(self, record_key, data):
        """Append a single data to the data sequence."""
        self.data_seq[record_key.key()].append(data)

    def getSingleData(self, record_key, time_idx):
        """Get single data from the data sequence."""
        key = record_key.key()
        data = self.data_seq[key][time_idx]
        if "rgb" in key:
            if data.ndim == 1:
                data = cv2.imdecode(data, flags=cv2.IMREAD_COLOR)
        elif "depth" in key:
            if data.ndim == 1:
                data = cv2.imdecode(data, flags=cv2.IMREAD_UNCHANGED)
        return data

    def getData(self, record_key):
        """Get data."""
        key = record_key.key()
        data_seq = self.data_seq[key]
        if "rgb" in key:
            if data_seq[0].ndim == 1:
                data_seq = np.array([cv2.imdecode(data, flags=cv2.IMREAD_COLOR) for data in data_seq])
        elif "depth" in key:
            if data_seq[0].ndim == 1:
                data_seq = np.array([cv2.imdecode(data, flags=cv2.IMREAD_UNCHANGED) for data in data_seq])
        return data_seq

    def compressData(self, record_key, compress_flag):
        """Compress data."""
        key = record_key.key()
        for time_idx, data in enumerate(self.data_seq[key]):
            if compress_flag == "jpg":
                self.data_seq[key][time_idx] = cv2.imencode(".jpg", data, (cv2.IMWRITE_JPEG_QUALITY, 95))[1]
            elif compress_flag == "exr":
                self.data_seq[key][time_idx] = cv2.imencode(".exr", data)[1]

    def saveData(self, filename):
        """Save data."""
        # If each element has a different shape, save it as an object array
        for key in self.data_seq.keys():
            if isinstance(self.data_seq[key], list) and \
               len({data.shape if isinstance(data, np.ndarray) else None for data in self.data_seq[key]}) > 1:
                self.data_seq[key] = np.array(self.data_seq[key], dtype=object)

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.savez(filename, **self.data_seq, **self.world_info, **self.camera_info)
        self.data_idx += 1

    def loadData(self, filename):
        """Load data."""
        npz_data = np.load(filename, allow_pickle=True)
        self.data_seq = dict()
        for key in npz_data.keys():
            self.data_seq[key] = np.copy(npz_data[key])

    def goToNextStatus(self):
        """Go to the next status."""
        self.status = RecordStatus((self.status.value + 1) % len(RecordStatus))

    def getStatusImage(self):
        """Get the image corresponding to the current status."""
        status_image = np.zeros((50, 160, 3), dtype=np.uint8)
        if self.status == RecordStatus.INITIAL:
            status_image[:, :] = np.array([200, 255, 200])
        elif self.status in {RecordStatus.PRE_REACH, RecordStatus.REACH, RecordStatus.GRASP}:
            status_image[:, :] = np.array([255, 255, 200])
        elif self.status == RecordStatus.TELEOP:
            status_image[:, :] = np.array([255, 200, 200])
        elif self.status == RecordStatus.END:
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

    def setupCameraInfo(self, camera_names, depth_keys):
        """Set camera info."""
        for camera_name, depth_key in zip(camera_names, depth_keys):
            self.camera_info[depth_key.key() + "_fovy"] = self.env.unwrapped.model.cam(camera_name).fovy[0]

    @property
    def status_elapsed_duration(self):
        """Get the elapsed duration of the current status."""
        return self.env.unwrapped.data.time - self.status_start_time

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
            self.status_start_time = self.env.unwrapped.data.time
