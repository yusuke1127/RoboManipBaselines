import os
import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pylab as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import cv2
import gymnasium as gym
import multimodal_robot_model
import pinocchio as pin
from tqdm import tqdm
import torch
sys.path.append("../../third_party/eipl/")
from eipl.utils import restore_args, tensor2numpy, deprocess_img, normalization, resize_img
sys.path.append("../../multimodal_robot_model/demos/")
from Utils_UR5eCableEnv import MotionManager, RecordStatus, RecordKey, RecordManager

parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, default=None, help=".pth file that PyTorch loads as checkpoint for model")
parser.add_argument("--dirname", type=str, default="../simulator/data", help="directory that stores test data, that has been generated, and will be loaded")
parser.add_argument("--pole-pos-idx", type=int, default=0, help="index of the position of poles (0-5)")
args = parser.parse_args()

# restore parameters
dir_name = os.path.split(args.filename)[0]
params = restore_args(os.path.join(dir_name, "args.json"))

# load dataset
minmax = [params["vmin"], params["vmax"]]
joint_bounds = np.load(os.path.join(args.dirname, "joint_bounds.npy"))
joint_scales = [1.0] * 6 + [0.01]

# define model
from eipl.model import SARNN
im_size = 64
model = SARNN(
    rec_dim=params["rec_dim"],
    joint_dim=7,
    k_dim=params["k_dim"],
    heatmap_size=params["heatmap_size"],
    temperature=params["temperature"],
    im_size=[im_size, im_size],
)

if params["compile"]:
    model = torch.compile(model)

# load weight
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

# Setup record manager
record_manager = RecordManager(env)
record_manager.setupSimWorld(pole_pos_idx=args.pole_pos_idx)

print("- Press space key to start automatic grasping.")

state = None
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

    skip = 10
    if record_manager.status == RecordStatus.TELEOP and time_idx % skip == 0:
        # load data and normalization
        front_img = info["images"]["front"]
        cropped_img_size = 112
        [fro_lef, fro_top] = [(front_img.shape[ax] - cropped_img_size) // 2 for ax in [0, 1]]
        [fro_rig, fro_bot] = [(p + cropped_img_size) for p in [fro_lef, fro_top]]
        front_img = front_img[fro_lef:fro_rig, fro_top:fro_bot, :]
        front_img = resize_img(np.expand_dims(front_img, 0), (64, 64))[0]
        front_img_t = front_img.transpose(2, 0, 1)
        front_img_t = normalization(front_img_t, (0, 255), minmax)
        front_img_t = torch.Tensor(np.expand_dims(front_img_t, 0))
        joint = motion_manager.getAction()
        joint_t = normalization(joint, joint_bounds, minmax)
        joint_t = torch.Tensor(np.expand_dims(joint_t, 0))
        y_front_image, y_joint, y_enc_front_pts, y_dec_front_pts, state = model(front_img_t, joint_t, state)

        # denormalization
        pred_front_image = tensor2numpy(y_front_image[0])
        pred_front_image = deprocess_img(pred_front_image, params["vmin"], params["vmax"])
        pred_front_image = pred_front_image.transpose(1, 2, 0)
        pred_joint = tensor2numpy(y_joint[0])
        pred_joint = normalization(pred_joint, minmax, joint_bounds)
        pred_joint_list.append(pred_joint)
        enc_front_pts = tensor2numpy(y_enc_front_pts[0]).reshape(params["k_dim"], 2) * im_size
        dec_front_pts = tensor2numpy(y_dec_front_pts[0]).reshape(params["k_dim"], 2) * im_size

    # Set gripper command
    if record_manager.status == RecordStatus.GRASP:
        motion_manager.gripper_pos = env.action_space.high[6]
    elif record_manager.status == RecordStatus.TELEOP:
        motion_manager.gripper_pos = pred_joint[6]

    # Solve IK
    if record_manager.status == RecordStatus.PRE_REACH or record_manager.status == RecordStatus.REACH:
        motion_manager.inverseKinematics()
    elif record_manager.status == RecordStatus.TELEOP:
        motion_manager.joint_pos = pred_joint[:6]

    # Step environment
    action = motion_manager.getAction()
    _, _, _, _, info = env.step(action)

    # Draw images
    status_image = record_manager.getStatusImage()
    online_image = cv2.vconcat([info["images"]["front"], info["images"]["side"], status_image])
    # front_img, pred_front_image
    window_image = online_image
    cv2.imshow("image", cv2.cvtColor(window_image, cv2.COLOR_RGB2BGR))

    if record_manager.status == RecordStatus.TELEOP and time_idx % skip == 0:
        for j in range(ax.shape[0]):
            for k in range(ax.shape[1]):
                ax[j, k].cla()

        # plot camera front_image
        ax[0, 0].imshow(front_img)
        for j in range(params["k_dim"]):
            ax[0, 0].plot(enc_front_pts[j, 0], enc_front_pts[j, 1], "co", markersize=12)  # encoder
            ax[0, 0].plot(
                dec_front_pts[j, 0], dec_front_pts[j, 1], "rx", markersize=12, markeredgewidth=2
            )  # decoder
        ax[0, 0].axis("off")
        ax[0, 0].set_title("Input front_image", fontsize=20)

        # plot predicted front_image
        ax[0, 1].imshow(pred_front_image)
        ax[0, 1].axis("off")
        ax[0, 1].set_title("Predicted front_image", fontsize=20)

        # plot joint
        ax[0, 2].set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        T = 100
        ax[0, 2].set_xlim(0, T)
        # ax[0, 2].plot(joints[1:] * joint_scales, linestyle="dashed", c="k")
        pred_joint_arr = np.array(pred_joint_list)
        for joint_idx in range(len(pred_joint_list[0])):
            ax[0, 2].plot(np.arange(len(pred_joint_list)), pred_joint_arr[:, joint_idx] * joint_scales[joint_idx])
        ax[0, 2].set_xlabel("Step", fontsize=20)
        ax[0, 2].set_title("Joint", fontsize=20)
        ax[0, 2].tick_params(axis="x", labelsize=16)
        ax[0, 2].tick_params(axis="y", labelsize=16)

        canvas.draw()
        buf = canvas.buffer_rgba()
        model_image = np.asarray(buf)
        cv2.imshow("image2", cv2.cvtColor(model_image, cv2.COLOR_RGB2BGR))
        # plt.draw()
        # plt.pause(0.001)

    key = cv2.waitKey(1)

    # Manage status
    if record_manager.status == RecordStatus.INITIAL:
        if key == 32: # space key
            record_manager.goToNextStatus()
    elif record_manager.status == RecordStatus.PRE_REACH:
        pre_reach_duration = 0.7 # [s]
        if record_manager.status_elapsed_duration > pre_reach_duration:
            record_manager.goToNextStatus()
    elif record_manager.status == RecordStatus.REACH:
        reach_duration = 0.3 # [s]
        if record_manager.status_elapsed_duration > reach_duration:
            record_manager.goToNextStatus()
            print("- Press space key to start playback after robot grasps the cable.")
    elif record_manager.status == RecordStatus.GRASP:
        if key == 32: # space key
            time_idx = 0
            matplotlib.use("agg")
            fig, ax = plt.subplots(1, 3, figsize=(14, 6), dpi=60)
            canvas = FigureCanvasAgg(fig)
            ax = ax.reshape(-1, 3)
            pred_joint_list = []
            record_manager.goToNextStatus()
    elif record_manager.status == RecordStatus.TELEOP:
        time_idx += 1
        if key == 32: # space key
            record_manager.goToNextStatus()
    elif record_manager.status == RecordStatus.END:
        if key == 32: # space key
            print("- Press space key to exit.")
            break
    if key == 27: # escape key
        break

# env.close()
