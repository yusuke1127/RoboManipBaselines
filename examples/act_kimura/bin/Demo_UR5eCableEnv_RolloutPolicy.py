import os
import sys
import argparse
import pickle
import numpy as np
import matplotlib
import matplotlib.pylab as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import cv2
import gymnasium as gym
import pinocchio as pin
from tqdm import tqdm
from einops import rearrange
import torch
sys.path.append("../../third_party/act")
from policy import ACTPolicy
from eipl.utils import restore_args, tensor2numpy, deprocess_img, normalization, resize_img
import multimodal_robot_model
from multimodal_robot_model.demos.Utils_UR5eCableEnv import MotionManager, RecordStatus, RecordKey, RecordManager

# command line parameters
parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, default=None, help=".pth file that PyTorch loads as checkpoint for policy")
parser.add_argument("--dirname", type=str, default="./data", help="directory that stores test data, that has been generated, and will be loaded")
parser.add_argument("--pole-pos-idx", type=int, default=0, help="index of the position of poles (0-5)")
parser.add_argument("--vmin", type=float, default=0.0)
parser.add_argument("--vmax", type=float, default=1.0)
parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
parser.add_argument('--task_name', choices=['sim_ur5ecable'], action='store', type=str, help='task_name', required=True)
parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
# for ACT
parser.add_argument('--kl_weight', default=10, type=int, help='KL Weight', required=False)
parser.add_argument('--chunk_size', default=100, type=int, help='chunk_size', required=False)
parser.add_argument('--hidden_dim', default=512, type=int, help='hidden_dim', required=False)
parser.add_argument('--dim_feedforward', default=3200, type=int, help='dim_feedforward', required=False)
parser.add_argument('--temporal_agg', action='store_true')
# repeat args in imitate_episodes just to avoid error. Will not be used
parser.add_argument('--policy_class', choices=['ACT'], action='store', type=str, help='policy_class, capitalize', required=True)
parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
args = parser.parse_args()

## Load dataset
minmax = [args.vmin, args.vmax]
joint_bounds = np.load(os.path.join(args.dirname, "joint_bounds.npy"))
joint_scales = [1.0] * 6 + [0.01]

# command line parameters
ckpt_dir = args.ckpt_dir
policy_class = args.policy_class
task_name = args.task_name
seed = args.seed

# get task parameters
is_sim = task_name[:4] == 'sim_'
if is_sim:
    from constants import SIM_TASK_CONFIGS
    task_config = SIM_TASK_CONFIGS[task_name]
else:
    assert False, f"{task_name=}"
dataset_dir = task_config['dataset_dir']
camera_names = task_config['camera_names']

# fixed parameters
state_dim = 14
lr_backbone = 1e-5
backbone = 'resnet18'
if policy_class == 'ACT':
    enc_layers = 4
    dec_layers = 7
    nheads = 8
    policy_config = {'num_queries': args.chunk_size,
                     'kl_weight': args.kl_weight,
                     'hidden_dim': args.hidden_dim,
                     'dim_feedforward': args.dim_feedforward,
                     'lr_backbone': lr_backbone,
                     'backbone': backbone,
                     'enc_layers': enc_layers,
                     'dec_layers': dec_layers,
                     'nheads': nheads,
                     'camera_names': camera_names,
                     }
else:
    assert False, f"{policy_class=}"

ckpt_name = f'policy_best.ckpt'

## Define policy
im_size = 64
joint_dim = 7
policy = ACTPolicy(policy_config)

## Load weight
ckpt_path = os.path.join(ckpt_dir, ckpt_name)
try:
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
except RuntimeError as e:
    if "size mismatch" in str(e.args):
        sys.stderr.write(f"\n{sys.stderr.name} {args.chunk_size=}\n\n")  # may be helpful 
    raise
print(loading_status)
policy.cuda()
policy.eval()

print(f'Loaded: {ckpt_path}')
stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
with open(stats_path, 'rb') as f:
    stats = pickle.load(f)

pre_process = lambda _joint: (_joint - stats['joint_mean']) / stats['joint_std']
post_process = lambda _action: _action * stats['action_std'] + stats['action_mean']

# Setup gym
env = gym.make(
  "multimodal_robot_model/UR5eCableEnv-v0",
  render_mode="human",
  extra_camera_configs=[{"name": "front", "size": (224, 224)}, {"name": "side", "size": (224, 224)}]
)
obs, info = env.reset(seed=seed)

# Setup motion manager
motion_manager = MotionManager(env)

# Setup record manager
record_manager = RecordManager(env)
record_manager.setupSimWorld(pole_pos_idx=args.pole_pos_idx)

# Setup window for policy image
matplotlib.use("agg")
fig, ax = plt.subplots(1, 3, figsize=(14, 6), dpi=60)
ax = ax.reshape(-1, 3)
canvas = FigureCanvasAgg(fig)
pred_joint_list = np.empty((0, joint_dim))
canvas.draw()
buf = canvas.buffer_rgba()
policy_image = np.asarray(buf)
cv2.imshow("Policy image", cv2.cvtColor(policy_image, cv2.COLOR_RGB2BGR))

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
        # Load data and normalization
        front_image = info["images"]["front"]
        cropped_img_size = 128
        [fro_lef, fro_top] = [(front_image.shape[ax] - cropped_img_size) // 2 for ax in [0, 1]]
        [fro_rig, fro_bot] = [(p + cropped_img_size) for p in [fro_lef, fro_top]]
        front_image = front_image[fro_lef:fro_rig, fro_top:fro_bot, :]
        front_image = resize_img(np.expand_dims(front_image, 0), (im_size, im_size))[0]
        front_image_t = front_image.transpose(2, 0, 1)
        front_image_t = front_image_t.astype(np.float32) / 255.0
        front_image_t = torch.Tensor(np.expand_dims(front_image_t, 0)).cuda().unsqueeze(0)
        joint = motion_manager.getAction()
        joint_t = pre_process(joint)
        joint_t = torch.Tensor(np.expand_dims(joint_t, 0)).cuda()

        # Infer
        all_actions = policy(joint_t, front_image_t)

        # denormalization
        pred_joint = tensor2numpy(all_actions[0, 0])
        pred_joint = post_process(pred_joint)
        pred_joint_list = np.concatenate([pred_joint_list, np.expand_dims(pred_joint, 0)])

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
    status_image = record_manager.getStatusImage()
    window_image = cv2.vconcat([info["images"]["front"], info["images"]["side"], status_image])
    cv2.imshow("Simulation image", cv2.cvtColor(window_image, cv2.COLOR_RGB2BGR))

    # Draw policy images
    if record_manager.status == RecordStatus.TELEOP and time_idx % skip == 0:
        for j in range(ax.shape[0]):
            for k in range(ax.shape[1]):
                ax[j, k].cla()

        # Draw camera front_image
        ax[0, 0].imshow(front_image)
        for j in range(args.k_dim):
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

        canvas.draw()
        buf = canvas.buffer_rgba()
        policy_image = np.asarray(buf)
        cv2.imshow("Policy image", cv2.cvtColor(policy_image, cv2.COLOR_RGB2BGR))
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
