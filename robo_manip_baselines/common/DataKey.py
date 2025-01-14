import warnings


class DataKey(object):
    """Data key."""

    # Time [s]
    TIME = "time"

    # Measured joint position (including both arm and gripper)
    MEASURED_JOINT_POS = "measured_joint_pos"
    # Command joint position (including both arm and gripper)
    COMMAND_JOINT_POS = "command_joint_pos"

    # Measured joint position relative to previous step (including both arm and gripper)
    MEASURED_JOINT_POS_REL = "measured_joint_pos_rel"
    # Command joint position relative to previous step (including both arm and gripper)
    COMMAND_JOINT_POS_REL = "command_joint_pos_rel"

    # Measured joint velocity
    MEASURED_JOINT_VEL = "measured_joint_vel"
    # Command joint velocity
    COMMAND_JOINT_VEL = "command_joint_vel"

    # Measured joint torque
    MEASURED_JOINT_TORQUE = "measured_joint_torque"
    # Command joint torque
    COMMAND_JOINT_TORQUE = "command_joint_torque"

    # Measured gripper joint position
    MEASURED_GRIPPER_JOINT_POS = "measured_gripper_joint_pos"
    # Command gripper joint position
    COMMAND_GRIPPER_JOINT_POS = "command_gripper_joint_pos"

    # Measured end-effector pose (tx, ty, tz, qw, qx, qy, qz)
    # Note: This is the end-effector pose corresponding to the measured joint position.
    MEASURED_EEF_POSE = "measured_eef_pose"
    # Command end-effector pose (tx, ty, tz, qw, qx, qy, qz)
    # Note: This is the target end-effector pose for IK, not the end-effector pose corresponding to the command joint position.
    COMMAND_EEF_POSE = "command_eef_pose"

    # Measured end-effector pose relative to previous step in the previous pose frame (tx, ty, tz, roll, pitch, yaw)
    MEASURED_EEF_POSE_REL = "measured_eef_pose_rel"
    # Command end-effector pose relative to previous step in the previous pose frame (tx, ty, tz, roll, pitch, yaw)
    COMMAND_EEF_POSE_REL = "command_eef_pose_rel"

    # Measured end-effector velocity (vx, vy, vz, wx, wy, wz)
    MEASURED_EEF_VEL = "measured_eef_vel"
    # Command end-effector velocity (vx, vy, vz, wx, wy, wz)
    COMMAND_EEF_VEL = "command_eef_vel"

    # Measured end-effector wrench (fx, fy, fz, nx, ny, nz)
    MEASURED_EEF_WRENCH = "measured_eef_wrench"
    # Command end-effector wrench (fx, fy, fz, nx, ny, nz)
    COMMAND_EEF_WRENCH = "command_eef_wrench"

    @classmethod
    def get_rel_key(cls, key):
        """Get the relative key corresponding to the absolute key."""
        if key == DataKey.MEASURED_JOINT_POS:
            return DataKey.MEASURED_JOINT_POS_REL
        elif key == DataKey.COMMAND_JOINT_POS:
            return DataKey.COMMAND_JOINT_POS_REL
        elif key == DataKey.MEASURED_EEF_POSE:
            return DataKey.MEASURED_EEF_POSE_REL
        elif key == DataKey.COMMAND_EEF_POSE:
            return DataKey.COMMAND_EEF_POSE_REL
        else:
            raise RuntimeError(f"[DataKey] Relative data key not found: {key}")

    @classmethod
    def get_abs_key(cls, key):
        """Get the absolute key corresponding to the relative key."""
        if key == DataKey.MEASURED_JOINT_POS_REL:
            return DataKey.MEASURED_JOINT_POS
        elif key == DataKey.COMMAND_JOINT_POS_REL:
            return DataKey.COMMAND_JOINT_POS
        elif key == DataKey.MEASURED_EEF_POSE_REL:
            return DataKey.MEASURED_EEF_POSE
        elif key == DataKey.COMMAND_EEF_POSE_REL:
            return DataKey.COMMAND_EEF_POSE
        else:
            raise RuntimeError(f"[DataKey] Absolute data key not found: {key}")

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
