import numpy as np
import pinocchio as pin


def get_pose_from_se3(se3):
    """Get pose (tx, ty, tz, qw, qx, qy, qz) from pinocchio SE3."""
    return np.concatenate(
        [se3.translation, pin.Quaternion(se3.rotation).coeffs()[[3, 0, 1, 2]]]
    )


def get_se3_from_pose(pose):
    """Get pinocchio SE3 from pose (tx, ty, tz, qw, qx, qy, qz)."""
    return pin.SE3(pin.Quaternion(*pose[3:7]), pose[0:3])


def get_rel_pose_from_se3(se3):
    """Get relative pose (tx, ty, tz, roll, pitch, yaw) from pinocchio SE3."""
    return np.concatenate([se3.translation, pin.rpy.matrixToRpy(se3.rotation)])


def get_se3_from_rel_pose(rel_pose):
    """Get pinocchio SE3 from relative pose (tx, ty, tz, roll, pitch, yaw)."""
    return pin.SE3(pin.rpy.rpyToMatrix(rel_pose[3:6]), rel_pose[0:3])
