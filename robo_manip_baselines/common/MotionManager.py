import numpy as np
import pinocchio as pin

from .DataManager import DataKey


class MotionManager(object):
    """
    Motion manager for robot arm and gripper.

    The manager computes forward and inverse kinematics using Pinocchio, a robot modeling library.
    """

    def __init__(self, env):
        self.env = env

        # Setup pinocchio model and data
        self.pin_model = pin.buildModelFromUrdf(self.env.unwrapped.arm_urdf_path)
        if self.env.unwrapped.arm_root_pose is not None:
            root_se3 = pin.SE3(
                pin.Quaternion(self.env.unwrapped.arm_root_pose[[4, 5, 6, 3]]),
                self.env.unwrapped.arm_root_pose[0:3],
            )
            self.pin_model.jointPlacements[1] = root_se3.act(
                self.pin_model.jointPlacements[1]
            )
        self.pin_data = self.pin_model.createData()
        self.pin_data_obs = self.pin_model.createData()

        # Setup arm
        self.joint_pos = self.env.unwrapped.init_qpos[
            self.env.unwrapped.ik_arm_joint_ids
        ].copy()
        self.forward_kinematics()
        self._original_target_se3 = self.pin_data.oMi[
            self.env.unwrapped.ik_eef_joint_id
        ].copy()
        self.target_se3 = self._original_target_se3.copy()

        # Setup gripper
        self._gripper_pos = self.env.unwrapped.init_qpos[
            self.env.unwrapped.gripper_action_idx
        ]

    def reset(self):
        """Reset states of arm and gripper."""
        self.joint_pos = self.env.unwrapped.init_qpos[
            self.env.unwrapped.ik_arm_joint_ids
        ].copy()
        self.target_se3 = self._original_target_se3.copy()
        self.gripper_pos = self.env.unwrapped.init_qpos[
            self.env.unwrapped.gripper_action_idx
        ]

    def forward_kinematics(self):
        pin.forwardKinematics(self.pin_model, self.pin_data, self.joint_pos)

    def inverse_kinematics(self):
        """Solve inverse kinematics."""
        # https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/md_doc_b-examples_d-inverse-kinematics.html
        error_se3 = self.current_se3.actInv(self.target_se3)
        error_vec = pin.log(error_se3).vector  # in joint frame
        J = pin.computeJointJacobian(
            self.pin_model,
            self.pin_data,
            self.joint_pos,
            self.env.unwrapped.ik_eef_joint_id,
        )  # in joint frame
        J = -1 * np.dot(pin.Jlog6(error_se3.inverse()), J)
        damping_scale = 1e-6
        delta_joint_pos = -1 * J.T.dot(
            np.linalg.solve(
                J.dot(J.T)
                + (np.dot(error_vec, error_vec) + damping_scale) * np.identity(6),
                error_vec,
            )
        )
        self.joint_pos = pin.integrate(self.pin_model, self.joint_pos, delta_joint_pos)
        self.forward_kinematics()

    def set_relative_target_se3(
        self, delta_pos=None, delta_rpy=None, is_delta_rpy_in_world_frame=True
    ):
        """Set the target pose of the end-effector relatively."""
        if delta_pos is not None:
            self.target_se3.translation += delta_pos
        if delta_rpy is not None:
            if is_delta_rpy_in_world_frame:
                self.target_se3.rotation = (
                    pin.rpy.rpyToMatrix(*delta_rpy) @ self.target_se3.rotation
                )
            else:
                self.target_se3.rotation = (
                    self.target_se3.rotation @ pin.rpy.rpyToMatrix(*delta_rpy)
                )

    def draw_markers(self):
        """Draw markers of the current and target poses of the end-effector to viewer."""
        self.env.unwrapped.draw_box_marker(
            pos=self.target_se3.translation,
            mat=self.target_se3.rotation,
            size=(0.02, 0.02, 0.03),
            rgba=(0, 1, 0, 0.5),
        )
        self.env.unwrapped.draw_box_marker(
            pos=self.current_se3.translation,
            mat=self.current_se3.rotation,
            size=(0.02, 0.02, 0.03),
            rgba=(1, 0, 0, 0.5),
        )

    def get_action(self):
        """Get action for Gym."""
        return np.concatenate([self.joint_pos, [self.gripper_pos]])

    def get_measured_data(self, key, obs):
        """Get measured data from observation by specifying a key."""
        if key == DataKey.MEASURED_JOINT_POS:
            return self.get_joint_pos(obs)
        elif key == DataKey.MEASURED_JOINT_VEL:
            return self.get_joint_vel(obs)
        elif key == DataKey.MEASURED_EEF_POSE:
            return self.get_measured_eef_pose(obs)
        elif key == DataKey.MEASURED_EEF_WRENCH:
            return self.get_eef_wrench(obs)
        else:
            raise RuntimeError(f"[MotionManager] Invalid data key: {key}")

    def get_joint_pos(self, obs):
        """Get joint position from observation."""
        return self.env.unwrapped.get_joint_pos_from_obs(obs, exclude_gripper=False)

    def get_joint_vel(self, obs):
        """Get joint velocity from observation."""
        return self.env.unwrapped.get_joint_vel_from_obs(obs, exclude_gripper=False)

    def get_eef_wrench(self, obs):
        """Get end-effector wrench from observation."""
        return self.env.unwrapped.get_eef_wrench_from_obs(obs)

    def get_measured_eef_pose(self, obs):
        """Get measured end-effector pose (tx, ty, tz, qw, qx, qy, qz) from observation.

        This is the end-effector pose corresponding to the measured joint position.
        """
        measured_joint_pos = self.env.unwrapped.get_joint_pos_from_obs(
            obs, exclude_gripper=True
        )
        pin.forwardKinematics(self.pin_model, self.pin_data_obs, measured_joint_pos)
        measured_se3 = self.pin_data_obs.oMi[self.env.unwrapped.ik_eef_joint_id]
        return np.concatenate(
            [
                measured_se3.translation,
                pin.Quaternion(measured_se3.rotation).coeffs()[[3, 0, 1, 2]],
            ]
        )

    def get_command_eef_pose(self):
        """Get command end-effector pose (tx, ty, tz, qw, qx, qy, qz).

        Note that this is the target end-effector pose for IK, not the end-effector pose corresponding to the command joint position.
        """
        return np.concatenate(
            [
                self.target_se3.translation,
                pin.Quaternion(self.target_se3.rotation).coeffs()[[3, 0, 1, 2]],
            ]
        )

    @property
    def current_se3(self):
        """Get the current pose of the end-effector."""
        return self.pin_data.oMi[self.env.unwrapped.ik_eef_joint_id]

    @property
    def gripper_pos(self):
        """Get the target gripper position."""
        return self._gripper_pos

    @gripper_pos.setter
    def gripper_pos(self, new_gripper_pos):
        """Set the target gripper position."""
        self._gripper_pos = np.clip(
            new_gripper_pos,
            self.env.action_space.low[self.env.unwrapped.gripper_action_idx],
            self.env.action_space.high[self.env.unwrapped.gripper_action_idx],
        )
