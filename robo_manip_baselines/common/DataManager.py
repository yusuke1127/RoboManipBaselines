import os
import numpy as np
import h5py
import cv2
import pinocchio as pin
from enum import Enum
from robo_manip_baselines import __version__
from .DataKey import DataKey


class MotionStatus(Enum):
    """Motion status."""

    INITIAL = 0
    PRE_REACH = 1
    REACH = 2
    GRASP = 3
    TELEOP = 4
    END = 5


class DataManager(object):
    """Data manager."""

    def __init__(self, env, demo_name=""):
        self.env = env

        self.general_info = {
            "format": "RoboManipBaselines-TeleopData-HDF5",
            "demo": demo_name,
            "version": __version__,
        }
        if self.env is not None:
            self.general_info["env"] = self.env.spec.name

        self.episode_idx = 0
        self.world_idx = 0
        self.world_info = {}

        self.camera_info = {}

        self.reset()

    def reset(self):
        """Reset."""
        self.status = MotionStatus(0)

        self.all_data_seq = {}

    def append_single_data(self, key, data):
        """Append a single data to the data sequence."""
        key = DataKey.replace_deprecated_key(key)  # For backward compatibility
        if key not in self.all_data_seq:
            self.all_data_seq[key] = []
        self.all_data_seq[key].append(data)

    def get_single_data(self, key, time_idx):
        """Get a single data from the data sequence."""
        key = DataKey.replace_deprecated_key(key)  # For backward compatibility
        data = self.all_data_seq[key][time_idx]
        return data

    def get_data(self, key):
        """Get a data sequence."""
        key = DataKey.replace_deprecated_key(key)  # For backward compatibility
        data_seq = self.all_data_seq[key]
        return data_seq

    def calc_relative_data(self, key, all_data_seq=None):
        """Calculate relative data."""
        if all_data_seq is None:
            all_data_seq = self.all_data_seq

        abs_key = DataKey.get_abs_key(key)

        if key in (DataKey.MEASURED_JOINT_POS_REL, DataKey.COMMAND_JOINT_POS_REL):
            if len(all_data_seq[abs_key]) < 2:
                return np.zeros_like(all_data_seq[abs_key][0])
            else:
                current_pos = all_data_seq[abs_key][-1]
                prev_pos = all_data_seq[abs_key][-2]
                return current_pos - prev_pos
        elif key in (DataKey.MEASURED_EEF_POSE_REL, DataKey.COMMAND_EEF_POSE_REL):
            if len(all_data_seq[abs_key]) < 2:
                return np.zeros(6)
            else:
                current_pose = all_data_seq[abs_key][-1]
                prev_pose = all_data_seq[abs_key][-2]
                rel_pos = current_pose[0:3] - prev_pose[0:3]
                rel_rpy = pin.rpy.matrixToRpy(
                    (
                        pin.Quaternion(*prev_pose[3:7]).inverse()
                        * pin.Quaternion(*current_pose[3:7])
                    ).toRotationMatrix()
                )
                return np.concatenate([rel_pos, rel_rpy])

    def save_data(self, filename, all_data_seq=None, increment_episode_idx=True):
        """Save data."""
        if all_data_seq is None:
            all_data_seq = self.all_data_seq

        # For backward compatibility
        for orig_key in all_data_seq.keys():
            new_key = DataKey.replace_deprecated_key(orig_key)
            if orig_key != new_key:
                all_data_seq[new_key] = all_data_seq.pop(orig_key)

        # Set meta data
        all_data_seq.update(self.general_info)
        all_data_seq.update(self.world_info)
        all_data_seq.update(self.camera_info)

        # Dump to a file
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with h5py.File(filename, "w") as h5file:
            for key in all_data_seq.keys():
                if isinstance(all_data_seq[key], list):
                    h5file.create_dataset(key, data=np.array(all_data_seq[key]))
                elif isinstance(all_data_seq[key], np.ndarray):
                    h5file.create_dataset(key, data=all_data_seq[key])
                else:
                    h5file.attrs[key] = all_data_seq[key]

        if increment_episode_idx:
            self.episode_idx += 1

    def load_data(self, filename):
        """Load data."""
        self.all_data_seq = {}
        with h5py.File(filename, "r") as h5file:
            for orig_key in h5file.keys():
                new_key = DataKey.replace_deprecated_key(
                    orig_key
                )  # For backward compatibility
                self.all_data_seq[new_key] = h5file[orig_key][()]
            for orig_key in h5file.attrs.keys():
                new_key = DataKey.replace_deprecated_key(
                    orig_key
                )  # For backward compatibility
                self.all_data_seq[new_key] = h5file.attrs[orig_key]

    def go_to_next_status(self):
        """Go to the next status."""
        if self.status == MotionStatus(len(MotionStatus) - 1):
            raise ValueError("Cannot go from the last status to the next.")
        self.status = MotionStatus(self.status.value + 1)

    def get_status_image(self):
        """Get the image corresponding to the current status."""
        status_image = np.zeros((50, 320, 3), dtype=np.uint8)
        if self.status == MotionStatus.INITIAL:
            status_image[:, :] = np.array([200, 255, 200])
        elif self.status in (
            MotionStatus.PRE_REACH,
            MotionStatus.REACH,
            MotionStatus.GRASP,
        ):
            status_image[:, :] = np.array([255, 255, 200])
        elif self.status == MotionStatus.TELEOP:
            status_image[:, :] = np.array([255, 200, 200])
        elif self.status == MotionStatus.END:
            status_image[:, :] = np.array([200, 200, 255])
        else:
            raise ValueError("Unknown status: {}".format(self.status))
        cv2.putText(
            status_image,
            self.status.name,
            (5, 35),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (0, 0, 0),
            2,
        )
        return status_image

    def setup_sim_world(self, world_idx=None):
        """Setup the simulation world."""
        if world_idx is None:
            kwargs = {"cumulative_idx": self.episode_idx}
        else:
            kwargs = {"world_idx": world_idx}
        self.world_idx = self.env.unwrapped.modify_world(**kwargs)
        self.world_info = {"world_idx": self.world_idx}

    def setup_camera_info(self):
        """Set camera info."""
        for camera_name in self.env.unwrapped.camera_names:
            depth_key = DataKey.get_depth_image_key(camera_name)
            self.camera_info[depth_key + "_fovy"] = self.env.unwrapped.get_camera_fovy(
                camera_name
            )

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
            self.status_start_time = self.env.unwrapped.get_time()

    @property
    def status_elapsed_duration(self):
        """Get the elapsed duration of the current status."""
        return self.env.unwrapped.get_time() - self.status_start_time
