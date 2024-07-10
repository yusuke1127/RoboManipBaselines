import os
import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pylab as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import cv2
import gymnasium as gym
import pinocchio as pin
import torch
from eipl.model import SARNN
from eipl.utils import restore_args, tensor2numpy, deprocess_img, normalization, resize_img
import multimodal_robot_model
from multimodal_robot_model.demos.Utils_UR5eCableEnv import MotionManager, RecordStatus, RecordKey, RecordManager

parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, default=None, help=".pth file that PyTorch loads as checkpoint for model")
parser.add_argument("--pole-pos-idx", type=int, default=0, help="index of the position of poles (0-5)")
parser.add_argument('--win_xy_policy', type=int, nargs=2, help='window xy policy', required=False)
parser.add_argument('--win_xy_simulation', type=int, nargs=2, help='window xy simulation', required=False)
args = parser.parse_args()
win_xy_policy = args.win_xy_policy
win_xy_simulation = args.win_xy_simulation

# Setup model
## Restore parameters
dir_name = os.path.split(args.filename)[0]
params = restore_args(os.path.join(dir_name, "args.json"))

## Load dataset
minmax = [params["vmin"], params["vmax"]]
joint_bounds = np.load(os.path.join(dir_name, "joint_bounds.npy"))
joint_scales = [1.0] * 6 + [0.01]

## Define model
im_size = 64
joint_dim = 7
model = SARNN(
    rec_dim=params["rec_dim"],
    joint_dim=joint_dim,
    k_dim=params["k_dim"],
    heatmap_size=params["heatmap_size"],
    temperature=params["temperature"],
    im_size=[im_size, im_size],
)

if params["compile"]:
    model = torch.compile(model)

## Load weight
ckpt = torch.load(args.filename, map_location=torch.device("cpu"))
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Setup gym
env = gym.make(
  "multimodal_robot_model/UR5eCableEnv-v0",
  render_mode="human",
  extra_camera_configs=[{"name": "front", "size": (224, 224)}, {"name": "side", "size": (224, 224)}]
)
obs, info = env.reset(seed=42)

# Setup motion manager
motion_manager = MotionManager(env)
RecordStatus.TELEOP._name_ = "AUTO"

# Setup record manager
record_manager = RecordManager(env)
record_manager.setupSimWorld(pole_pos_idx=args.pole_pos_idx)
original_pole_pos_x = env.unwrapped.model.body("poles").pos[0]
pole_swing_phase_offset = 2.0 * np.pi * np.random.rand()

# Setup window for policy image
matplotlib.use("agg")
fig, ax = plt.subplots(1, 3, figsize=(13.4, 5.0), dpi=60)
ax = ax.reshape(-1, 3)
for _ax in np.ravel(ax):
    _ax.cla()
    _ax.axis("off")
canvas = FigureCanvasAgg(fig)
pred_joint_list = np.empty((0, joint_dim))
canvas.draw()
buf = canvas.buffer_rgba()
policy_image = np.asarray(buf)
cv2.imshow("Policy image", cv2.cvtColor(policy_image, cv2.COLOR_RGB2BGR))
if win_xy_policy is not None:
    cv2.moveWindow("Policy image", *win_xy_policy)

print("- Press space key to start automatic grasping.")

rnn_state = None
while True:
    # Set arm command
    if record_manager.status == RecordStatus.PRE_REACH:
        target_pos = env.unwrapped.model.body("cable_end").pos.copy()
        target_pos[2] = 1.02 # [m]
        motion_manager.target_se3 = pin.SE3(np.diag([-1.0, 1.0, -1.0]), target_pos)
    elif record_manager.status == RecordStatus.REACH:
        target_pos = env.unwrapped.model.body("cable_end").pos.copy()
        target_pos[2] = 0.995 # [m]
        motion_manager.target_se3 = pin.SE3(np.diag([-1.0, 1.0, -1.0]), target_pos)

    enable_pole_swing = False
    if record_manager.status == RecordStatus.TELEOP and enable_pole_swing:
        pole_swing_scale = 0.05
        pole_swing_freq = 0.1
        env.unwrapped.model.body("poles").pos[0] = original_pole_pos_x + \
            pole_swing_scale * np.sin(2.0 * np.pi * pole_swing_freq * record_manager.status_elapsed_duration + pole_swing_phase_offset)

    skip = 10
    if record_manager.status == RecordStatus.TELEOP and time_idx % skip == 0:
        # Load data and normalization
        front_image = info["images"]["front"]
        cropped_img_size = 128
        [fro_lef, fro_top] = [(front_image.shape[ax] - cropped_img_size) // 2 for ax in [0, 1]]
        [fro_rig, fro_bot] = [(p + cropped_img_size) for p in [fro_lef, fro_top]]
        front_image = front_image[fro_lef:fro_rig, fro_top:fro_bot, :]
        front_image = resize_img(np.expand_dims(front_image, 0), (im_size, im_size))[0]
        front_image_t = front_image.transpose(2, 0, 1)
        front_image_t = normalization(front_image_t, (0, 255), minmax)
        front_image_t = torch.Tensor(np.expand_dims(front_image_t, 0))
        joint = motion_manager.getAction()
        joint_t = normalization(joint, joint_bounds, minmax)
        joint_t = torch.Tensor(np.expand_dims(joint_t, 0))

        # Infer
        y_front_image, y_joint, y_enc_front_pts, y_dec_front_pts, rnn_state = model(front_image_t, joint_t, rnn_state)

        # denormalization
        pred_front_image = tensor2numpy(y_front_image[0])
        pred_front_image = deprocess_img(pred_front_image, params["vmin"], params["vmax"])
        pred_front_image = pred_front_image.transpose(1, 2, 0)
        pred_joint = tensor2numpy(y_joint[0])
        pred_joint = normalization(pred_joint, minmax, joint_bounds)
        pred_joint_list = np.concatenate([pred_joint_list, np.expand_dims(pred_joint, 0)])
        enc_front_pts = tensor2numpy(y_enc_front_pts[0]).reshape(params["k_dim"], 2) * im_size
        dec_front_pts = tensor2numpy(y_dec_front_pts[0]).reshape(params["k_dim"], 2) * im_size

    # Set gripper command
    if record_manager.status == RecordStatus.GRASP:
        motion_manager.gripper_pos = env.action_space.high[6]
    elif record_manager.status == RecordStatus.TELEOP:
        motion_manager.gripper_pos = pred_joint[6]

    # Set joint command
    if record_manager.status == RecordStatus.PRE_REACH or record_manager.status == RecordStatus.REACH:
        motion_manager.inverseKinematics()
    elif record_manager.status == RecordStatus.TELEOP:
        motion_manager.joint_pos = pred_joint[:6]

    # Step environment
    action = motion_manager.getAction()
    _, _, _, _, info = env.step(action)

    # Draw simulation images
    enable_simulation_image = False
    if enable_simulation_image:
        status_image = record_manager.getStatusImage()
        window_image = cv2.vconcat([info["images"]["front"], info["images"]["side"], status_image])
        cv2.imshow("Simulation image", cv2.cvtColor(window_image, cv2.COLOR_RGB2BGR))
        if win_xy_simulation is not None:
            cv2.moveWindow("Simulation image", *win_xy_simulation)

    # Draw policy images
    if record_manager.status == RecordStatus.TELEOP and time_idx % skip == 0:
        for _ax in np.ravel(ax):
            _ax.cla()
            _ax.axis("off")

        # Draw camera front_image
        ax[0, 0].imshow(front_image)
        for j in range(params["k_dim"]):
            ax[0, 0].plot(enc_front_pts[j, 0], enc_front_pts[j, 1], "co", markersize=12)  # encoder
            ax[0, 0].plot(
                dec_front_pts[j, 0], dec_front_pts[j, 1], "rx", markersize=12, markeredgewidth=2
            )  # decoder
        ax[0, 0].axis("off")
        ax[0, 0].set_title("Input front_image", fontsize=20)

        # Draw predicted front_image
        ax[0, 1].imshow(pred_front_image)
        ax[0, 1].axis("off")
        ax[0, 1].set_title("Predicted front_image", fontsize=20)

        # Plot joint
        T = 100
        ax[0, 2].set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax[0, 2].set_xlim(0, T)
        for joint_idx in range(pred_joint_list.shape[1]):
            ax[0, 2].plot(np.arange(pred_joint_list.shape[0]), pred_joint_list[:, joint_idx] * joint_scales[joint_idx])
        ax[0, 2].set_xlabel("Step", fontsize=20)
        ax[0, 2].set_title("Joint", fontsize=20)
        ax[0, 2].tick_params(axis="x", labelsize=16)
        ax[0, 2].tick_params(axis="y", labelsize=16)
        ax[0, 2].axis("on")

        canvas.draw()
        buf = canvas.buffer_rgba()
        policy_image = np.asarray(buf)
        cv2.imshow("Policy image", cv2.cvtColor(policy_image, cv2.COLOR_RGB2BGR))
        # plt.draw()
        # plt.pause(0.001)

    key = cv2.waitKey(1)

    # Manage status
    if record_manager.status == RecordStatus.INITIAL:
        initial_duration = 1.0 # [s]
        if record_manager.status_elapsed_duration > initial_duration:
            record_manager.goToNextStatus()
    elif record_manager.status == RecordStatus.PRE_REACH:
        pre_reach_duration = 0.7 # [s]
        if record_manager.status_elapsed_duration > pre_reach_duration:
            record_manager.goToNextStatus()
    elif record_manager.status == RecordStatus.REACH:
        reach_duration = 0.3 # [s]
        if record_manager.status_elapsed_duration > reach_duration:
            record_manager.goToNextStatus()
    elif record_manager.status == RecordStatus.GRASP:
        grasp_duration = 0.5 # [s]
        if record_manager.status_elapsed_duration > grasp_duration:
            time_idx = 0
            record_manager.goToNextStatus()
            print("- Press space key to finish policy rollout.")
    elif record_manager.status == RecordStatus.TELEOP:
        time_idx += 1
        if key == 32: # space key
            record_manager.goToNextStatus()
            print("- Press space key to exit.")
    elif record_manager.status == RecordStatus.END:
        if key == 32: # space key
            break
    if key == 27: # escape key
        break

# env.close()
