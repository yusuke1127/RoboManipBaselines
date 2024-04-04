import numpy as np
import gymnasium as gym
import multimodal_robot_model
import pinocchio as pin
import pyspacemouse
import mujoco

# Setup gym
env = gym.make("multimodal_robot_model/UR5eCableEnv-v0", render_mode="human")
obs, info = env.reset(seed=42)
action = np.zeros(env.action_space.shape)

# Setup pinocchio
root_se3 = pin.SE3(np.matmul(pin.rpy.rpyToMatrix(2.35619, 0.0, 0.0), pin.rpy.rpyToMatrix(0.0, 0.0, -1.5708)), np.array([-0.27, -0.18, 1.32]))
model = pin.buildModelFromUrdf(env.unwrapped.urdf_path)
model.jointPlacements[1] = root_se3.act(model.jointPlacements[1])
data = model.createData()
q = env.unwrapped.init_qpos[:6].copy()
pin.forwardKinematics(model, data, q)
eef_joint_id = 6
target_se3 = data.oMi[eef_joint_id].copy()

# Setup spacemouse
pyspacemouse.open()

for _ in range(10000):
    # Solve FK
    pin.forwardKinematics(model, data, q)
    current_se3 = data.oMi[eef_joint_id]

    # Read spacemouse
    spacemouse_state = pyspacemouse.read()

    # Set arm command
    pos_scale = 1e-2
    target_se3.translation += pos_scale * np.array([spacemouse_state.y, -1.0 * spacemouse_state.x, spacemouse_state.z])
    ori_scale = 1e-2
    rpy = ori_scale * np.array([spacemouse_state.roll, spacemouse_state.pitch, -1.0 * spacemouse_state.yaw])
    target_se3.rotation = np.matmul(pin.rpy.rpyToMatrix(*rpy), target_se3.rotation)

    # Set gripper command
    gripper_scale = 5.0
    if spacemouse_state.buttons[0] > 0 and spacemouse_state.buttons[1] <= 0:
        action[6] = np.clip(action[6] + gripper_scale, env.action_space.low[6], env.action_space.high[6])
    elif spacemouse_state.buttons[1] > 0 and spacemouse_state.buttons[0] <= 0:
        action[6] = np.clip(action[6] - gripper_scale, env.action_space.low[6], env.action_space.high[6])

    # Draw markers
    env.unwrapped.mujoco_renderer.viewer.add_marker(
        pos=target_se3.translation,
        mat=target_se3.rotation,
        label="",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.02, 0.02, 0.03),
        rgba=(0, 1, 0, 0.5))
    env.unwrapped.mujoco_renderer.viewer.add_marker(
        pos=current_se3.translation,
        mat=current_se3.rotation,
        label="",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.02, 0.02, 0.03),
        rgba=(1, 0, 0, 0.5))

    # Solve IK
    # https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/md_doc_b-examples_d-inverse-kinematics.html
    error_se3 = current_se3.actInv(target_se3)
    error_vec = pin.log(error_se3).vector # in joint frame
    J = pin.computeJointJacobian(model, data, q, eef_joint_id) # in joint frame
    J = -1 * np.dot(pin.Jlog6(error_se3.inverse()), J)
    damping_scale = 1e-6
    delta_q = -1 * J.T.dot(np.linalg.solve(J.dot(J.T) + (np.dot(error_vec, error_vec) + damping_scale) * np.identity(6), error_vec))
    q = pin.integrate(model, q, delta_q)

    # Step environment
    action[:6] = q
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
        print("Reset environment. terminated: {}, truncated: {}".format(terminated, truncated))

env.close()
