import numpy as np
import pinocchio as pin
import mujoco

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
            root_se3 = pin.SE3(pin.Quaternion(self.env.unwrapped.arm_root_pose[[4, 5, 6, 3]]),
                               self.env.unwrapped.arm_root_pose[0:3])
            self.pin_model.jointPlacements[1] = root_se3.act(self.pin_model.jointPlacements[1])
        self.pin_data = self.pin_model.createData()
        self.pin_data_obs = self.pin_model.createData()

        # Setup arm
        self.joint_pos = self.env.unwrapped.init_qpos[:6].copy()
        self.eef_joint_id = 6
        pin.forwardKinematics(self.pin_model, self.pin_data, self.joint_pos)
        self._original_target_se3 = self.pin_data.oMi[self.eef_joint_id].copy()
        self.target_se3 = self._original_target_se3.copy()

        # Setup gripper
        self._gripper_pos = 0.0
        self.gripper_action_idx = 6

    def reset(self):
        """Reset states of arm and gripper."""
        self.joint_pos = self.env.unwrapped.init_qpos[:6].copy()
        self.target_se3 = self._original_target_se3.copy()
        self.gripper_pos = 0.0

    def inverseKinematics(self):
        """Solve inverse kinematics."""
        # https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/md_doc_b-examples_d-inverse-kinematics.html
        error_se3 = self.current_se3.actInv(self.target_se3)
        error_vec = pin.log(error_se3).vector # in joint frame
        J = pin.computeJointJacobian(self.pin_model, self.pin_data, self.joint_pos, self.eef_joint_id) # in joint frame
        J = -1 * np.dot(pin.Jlog6(error_se3.inverse()), J)
        damping_scale = 1e-6
        delta_joint_pos = -1 * J.T.dot(np.linalg.solve(
            J.dot(J.T) + (np.dot(error_vec, error_vec) + damping_scale) * np.identity(6), error_vec))
        self.joint_pos = pin.integrate(self.pin_model, self.joint_pos, delta_joint_pos)
        pin.forwardKinematics(self.pin_model, self.pin_data, self.joint_pos)

    def setRelativeTargetSE3(self, delta_pos=None, delta_rpy=None):
        """Set the target pose of the end-effector relatively."""
        if delta_pos is not None:
            self.target_se3.translation += delta_pos
        if delta_rpy is not None:
            self.target_se3.rotation = np.matmul(pin.rpy.rpyToMatrix(*delta_rpy), self.target_se3.rotation)

    def drawMarkers(self):
        """Draw markers of the current and target poses of the end-effector to viewer."""
        self.env.unwrapped.draw_box_marker(
            pos=self.target_se3.translation,
            mat=self.target_se3.rotation,
            size=(0.02, 0.02, 0.03),
            rgba=(0, 1, 0, 0.5))
        self.env.unwrapped.draw_box_marker(
            pos=self.current_se3.translation,
            mat=self.current_se3.rotation,
            size=(0.02, 0.02, 0.03),
            rgba=(1, 0, 0, 0.5))

    def getAction(self):
        """Get action for Gym."""
        return np.concatenate([self.joint_pos, [self.gripper_pos]])

    def getJointPos(self, obs):
        """Get joint position from observation."""
        arm_qpos = self.env.unwrapped.get_arm_qpos_from_obs(obs)
        gripper_pos = self.env.unwrapped.get_gripper_pos_from_obs(obs)
        return np.concatenate((arm_qpos, gripper_pos))

    def getJointVel(self, obs):
        """Get joint velocity from observation."""
        arm_qvel = self.env.unwrapped.get_arm_qvel_from_obs(obs)
        # Set zero as a dummy because the joint velocity of gripper cannot be obtained
        gripper_vel = np.zeros(1)
        return np.concatenate((arm_qvel, gripper_vel))

    def getEefWrench(self, obs):
        """Get end-effector wrench from observation."""
        return self.env.unwrapped.get_eef_wrench_from_obs(obs)

    def getMeasuredEef(self, obs):
        """Get measured end-effector pose (tx, ty, tz, rx, ry, rz, rw) from observation."""
        arm_qpos = self.env.unwrapped.get_arm_qpos_from_obs(obs)
        pin.forwardKinematics(self.pin_model, self.pin_data_obs, arm_qpos)
        measured_se3 = self.pin_data_obs.oMi[self.eef_joint_id]
        return np.concatenate([measured_se3.translation, pin.Quaternion(measured_se3.rotation).coeffs()])

    def getCommandEef(self):
        """Get command end-effector pose (tx, ty, tz, rx, ry, rz, rw)."""
        return np.concatenate([self.target_se3.translation, pin.Quaternion(self.target_se3.rotation).coeffs()])

    @property
    def current_se3(self):
        """Get the current pose of the end-effector."""
        return self.pin_data.oMi[self.eef_joint_id]

    @property
    def gripper_pos(self):
        """Get the target gripper position."""
        return self._gripper_pos

    @gripper_pos.setter
    def gripper_pos(self, new_gripper_pos):
        """Set the target gripper position."""
        self._gripper_pos = np.clip(new_gripper_pos,
                                    self.env.action_space.low[self.gripper_action_idx],
                                    self.env.action_space.high[self.gripper_action_idx])
