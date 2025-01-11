import os
import warnings
import numpy as np
import h5py
import cv2
import pinocchio as pin
from enum import Enum
from robo_manip_baselines import __version__

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

    MEASURED_JOINT_POS_REL = "measured_joint_pos_rel"
    COMMAND_JOINT_POS_REL = "command_joint_pos_rel"

    MEASURED_JOINT_VEL = "measured_joint_vel"
    COMMAND_JOINT_VEL = "command_joint_vel"

    MEASURED_JOINT_TORQUE = "measured_joint_torque"
    COMMAND_JOINT_TORQUE = "command_joint_torque"

    MEASURED_EEF_POSE = "measured_eef_pose"
    COMMAND_EEF_POSE = "command_eef_pose"

    MEASURED_EEF_POSE_REL = "measured_eef_pose_rel"
    COMMAND_EEF_POSE_REL = "command_eef_pose_rel"

    MEASURED_EEF_VEL = "measured_eef_vel"
    COMMAND_EEF_VEL = "command_eef_vel"

    MEASURED_EEF_WRENCH = "measured_eef_wrench"
    COMMAND_EEF_WRENCH = "command_eef_wrench"

    @classmethod
    def get_rgb_image_key(cls, camera_name):
        """Get the rgb image key from the camera name."""
        return camera_name.lower() + "_rgb_image"

    @classmethod
    def get_depth_image_key(cls, camera_name):
        """Get the depth image key from the camera name."""
        return camera_name.lower() + "_depth_image"

    @classmethod
    def replace_deprecated_key(cls, orig_key):
        """Replace a deprecated key with a new key for backward compatibility."""
        if orig_key == "joint_pos":
            new_key = DataKey.MEASURED_JOINT_POS
        elif orig_key == "joint_vel":
            new_key = DataKey.MEASURED_JOINT_VEL
        elif orig_key == "wrench":
            new_key = DataKey.MEASURED_EEF_WRENCH
        elif orig_key == "measured_eef":
            new_key = DataKey.MEASURED_EEF_POSE
        elif orig_key == "command_eef":
            new_key = DataKey.COMMAND_EEF_POSE
        elif orig_key == "measured_wrench":
            new_key = DataKey.MEASURED_EEF_WRENCH
        elif orig_key == "command_wrench":
            new_key = DataKey.COMMAND_EEF_WRENCH
        elif orig_key == "action":
            new_key = DataKey.COMMAND_JOINT_POS
        else:
            new_key = orig_key.lower()
        if orig_key != new_key:
            warnings.warn(
                f'[DataKey] "{orig_key}" is deprecated, use "{new_key}" instead.'
            )
        return new_key


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

    def finalize_data(self, all_data_seq=None):
        """Finalize data."""
        if all_data_seq is None:
            all_data_seq = self.all_data_seq

        # Set relative joint position
        for joint_pos_key, joint_pos_rel_key in [
            (DataKey.MEASURED_JOINT_POS, DataKey.MEASURED_JOINT_POS_REL),
            (DataKey.COMMAND_JOINT_POS, DataKey.COMMAND_JOINT_POS_REL),
        ]:
            all_data_seq[joint_pos_rel_key] = np.concatenate(
                [
                    np.zeros((1, len(all_data_seq[joint_pos_key][0]))),
                    all_data_seq[joint_pos_key][1:] - all_data_seq[joint_pos_key][:-1],
                ]
            )

        # Set relative end-effector pose
        for eef_pose_key, eef_pose_rel_key in [
            (DataKey.MEASURED_EEF_POSE, DataKey.MEASURED_EEF_POSE_REL),
            (DataKey.COMMAND_EEF_POSE, DataKey.COMMAND_EEF_POSE_REL),
        ]:
            all_data_seq[eef_pose_rel_key] = []
            for time_idx in range(len(all_data_seq[DataKey.TIME])):
                if time_idx == 0:
                    rel_pose = np.zeros(6)
                else:
                    current_pose = all_data_seq[eef_pose_key][time_idx]
                    prev_pose = all_data_seq[eef_pose_key][time_idx - 1]
                    rel_pose = np.concatenate(
                        [
                            current_pose[0:3] - prev_pose[0:3],
                            (
                                pin.Quaternion(*prev_pose[3:7]).inverse()
                                * pin.Quaternion(*current_pose[3:7])
                            ).coeffs()[[3, 0, 1, 2]],
                        ]
                    )
                all_data_seq[eef_pose_rel_key].append(rel_pose)

        # Convert list data to numpy array
        for key in all_data_seq.keys():
            if isinstance(all_data_seq[key], list):
                all_data_seq[key] = np.array(all_data_seq[key])

    def save_data(self, filename, all_data_seq=None):
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
        with h5py.File(filename, "w") as f:
            for key in all_data_seq.keys():
                if isinstance(all_data_seq[key], list):
                    raise RuntimeError(
                        "[DataManager] List data is not assumed. finalize_data() should be called first."
                    )
                elif isinstance(all_data_seq[key], np.ndarray):
                    f.create_dataset(key, data=all_data_seq[key])
                else:
                    f.attrs[key] = all_data_seq[key]

        if all_data_seq is None:
            self.episode_idx += 1

    def load_data(self, filename):
        """Load data."""
        self.all_data_seq = {}
        with h5py.File(filename, "r") as f:
            for orig_key in f.keys():
                new_key = DataKey.replace_deprecated_key(
                    orig_key
                )  # For backward compatibility
                self.all_data_seq[new_key] = f[orig_key][()]
            for orig_key in f.attrs.keys():
                new_key = DataKey.replace_deprecated_key(
                    orig_key
                )  # For backward compatibility
                self.all_data_seq[new_key] = f.attrs[orig_key]

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
