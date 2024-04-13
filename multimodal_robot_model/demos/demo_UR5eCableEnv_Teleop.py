import os
import numpy as np
import cv2
import gymnasium as gym
import multimodal_robot_model
import pinocchio as pin
import pyspacemouse
import mujoco

def reset_recording(pole_pos_idx=None):
    global status, data_seq

    # Reset variables for recording
    status = "initial"
    data_seq = {
        "time": [],
        "joint": [],
        "front_image": [],
        "side_image": [],
        "wrench": [],
    }

    # Set position of poles
    pole_pos_offsets = np.array([
        [-0.03, 0, 0.0],
        [0.0, 0, 0.0],
        [0.03, 0, 0.0],
        [0.06, 0, 0.0],
        [0.09, 0, 0.0],
        [0.12, 0, 0.0],
    ])
    if pole_pos_idx is None:
        pole_pos_idx = data_idx % len(pole_pos_offsets)
    env.unwrapped.model.body("poles").pos = original_pole_pos + pole_pos_offsets[pole_pos_idx]

    # Set environment index (currently only the position of poles is a dependent parameter)
    env_idx = pole_pos_idx
    print("Press space key to start teleoperation. (env_idx: {})".format(env_idx))
    return env_idx

def get_status_image(status):
    status_image = np.zeros((50, 224, 3), dtype=np.uint8)
    if status == "initial":
        status_image[:, :] = np.array([200, 255, 200])
    elif status == "record":
        status_image[:, :] = np.array([255, 200, 200])
    elif status == "end":
        status_image[:, :] = np.array([200, 200, 255])
    else:
        raise ValueError("Unknown status: {}".format(status))
    cv2.putText(status_image, status, (5, 35), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 2)
    return status_image

# Setup gym
env = gym.make(
  "multimodal_robot_model/UR5eCableEnv-v0",
  render_mode="human",
  extra_camera_configs=[{"name": "front", "size": (224, 224)}, {"name": "side", "size": (224, 224)}]
)
obs, info = env.reset(seed=42)
action = np.zeros(env.action_space.shape)

# Setup pinocchio
root_se3 = pin.SE3(np.identity(3), np.array([-0.605, 0.0, 0.8])) # env_ur5e_cable_verticalup.xml
# root_se3 = pin.SE3(np.matmul(pin.rpy.rpyToMatrix(2.35619, 0.0, 0.0), pin.rpy.rpyToMatrix(0.0, 0.0, -1.5708)), np.array([-0.27, -0.18, 1.32])) # env_ur5e_cable_diagonaldown.xml
model = pin.buildModelFromUrdf(env.unwrapped.urdf_path)
model.jointPlacements[1] = root_se3.act(model.jointPlacements[1])
data = model.createData()
joint_pos = env.unwrapped.init_qpos[:6].copy()
eef_joint_id = 6
pin.forwardKinematics(model, data, joint_pos)
original_target_se3 = data.oMi[eef_joint_id].copy()
target_se3 = original_target_se3.copy()
gripper_pos = 0

# Setup spacemouse
pyspacemouse.open()

# Setup data recording
data_idx = 0
status_list = ["initial", "record", "end"]
original_pole_pos = env.unwrapped.model.body("poles").pos.copy()
env_idx = reset_recording()

while True:
    # Solve FK
    pin.forwardKinematics(model, data, joint_pos)
    current_se3 = data.oMi[eef_joint_id]

    # Read spacemouse
    spacemouse_state = pyspacemouse.read()

    # Set arm command
    pos_scale = 1e-2
    target_se3.translation += pos_scale * np.array([-1.0 * spacemouse_state.y, spacemouse_state.x, spacemouse_state.z])
    ori_scale = 5e-3
    rpy = ori_scale * np.array([-1.0 * spacemouse_state.roll, -1.0 * spacemouse_state.pitch, -2.0 * spacemouse_state.yaw])
    target_se3.rotation = np.matmul(pin.rpy.rpyToMatrix(*rpy), target_se3.rotation)

    # Set gripper command
    gripper_scale = 5.0
    if spacemouse_state.buttons[0] > 0 and spacemouse_state.buttons[1] <= 0:
        gripper_pos = np.clip(gripper_pos + gripper_scale, env.action_space.low[6], env.action_space.high[6])
    elif spacemouse_state.buttons[1] > 0 and spacemouse_state.buttons[0] <= 0:
        gripper_pos = np.clip(gripper_pos - gripper_scale, env.action_space.low[6], env.action_space.high[6])

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
    J = pin.computeJointJacobian(model, data, joint_pos, eef_joint_id) # in joint frame
    J = -1 * np.dot(pin.Jlog6(error_se3.inverse()), J)
    damping_scale = 1e-6
    delta_joint_pos = -1 * J.T.dot(np.linalg.solve(J.dot(J.T) + (np.dot(error_vec, error_vec) + damping_scale) * np.identity(6), error_vec))
    joint_pos = pin.integrate(model, joint_pos, delta_joint_pos)

    # Step environment
    action[:6] = joint_pos
    action[6] = gripper_pos
    obs, _, _, _, info = env.step(action)

    # Record data
    if status == "record":
        data_seq["time"].append(env.unwrapped.data.time - start_time)
        data_seq["joint"].append(action)
        data_seq["front_image"].append(info["images"]["front"])
        data_seq["side_image"].append(info["images"]["side"])
        data_seq["wrench"].append(obs[16:])

    # Draw images
    window_image = np.concatenate([info["images"]["front"], info["images"]["side"], get_status_image(status)])
    cv2.imshow("image", cv2.cvtColor(window_image, cv2.COLOR_RGB2BGR))
    key = cv2.waitKey(1)
    reset = False
    if status == "initial":
        if key == 32: # space key
            status = "record"
            start_time = env.unwrapped.data.time
            print("Press space key to finish teleoperation.")
    elif status == "record":
        if key == 32: # space key
            status = "end"
            print("Press the 's' key if the teleoperation succeeded, or the 'f' key if it failed. (duration: {:.1f} [s])".format(data_seq["time"][-1]))
    elif status == "end":
        if key == ord("s"):
            # Save data
            dirname = "teleop_data/env{:0>1}".format(env_idx)
            os.makedirs(dirname, exist_ok=True)
            filename = "{}/UR5eCableEnv_env{:0>1}_{:0>3}.npz".format(dirname, env_idx, data_idx)
            print("Teleoperation succeeded: Save the data as {}".format(filename))
            np.savez(filename,
                     times=data_seq["time"],
                     joints=data_seq["joint"],
                     front_image=data_seq["front_image"],
                     side_image=data_seq["side_image"],
                     wrench=data_seq["wrench"])
            data_idx += 1
            reset = True
        elif key == ord("f"):
            print("Teleoperation failed: Reset without saving")
            reset = True
    if key == 27: # escape key
        break

    # Check end conditions
    if reset:
        env_idx = reset_recording()
        obs, info = env.reset()
        joint_pos = env.unwrapped.init_qpos[:6].copy()
        target_se3 = original_target_se3.copy()
        gripper_pos = 0

# env.close()
