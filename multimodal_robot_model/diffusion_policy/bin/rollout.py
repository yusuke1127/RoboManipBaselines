import os
import sys
import argparse
import hydra
import dill
import numpy as np
import matplotlib
import matplotlib.pylab as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import cv2
import gymnasium as gym
import pinocchio as pin
import torch
from diffusion_policy.common.pytorch_util import dict_apply
import multimodal_robot_model
from multimodal_robot_model.demos.Utils_UR5eCableEnv import MotionManager, RecordStatus, RecordKey, RecordManager
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# command line parameters
parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, default=None, help=".ckpt file that PyTorch loads as checkpoint for policy")
parser.add_argument("--pole-pos-idx", type=int, default=0, help="index of the position of poles (0-5)")
parser.add_argument('--skip', default=4, type=int, help='skip', required=False)
parser.add_argument('--win_xy_policy', type=int, nargs=2, help='window xy policy', required=False)
parser.add_argument('--win_xy_simulation', type=int, nargs=2, help='window xy simulation', required=False)
args = parser.parse_args()
win_xy_policy = args.win_xy_policy
win_xy_simulation = args.win_xy_simulation

# Setup model
ckpt_data = torch.load(args.filename)
cfg = ckpt_data["cfg"]
policy = hydra.utils.instantiate(cfg.policy)
policy.load_state_dict(ckpt_data["state_dicts"]["ema_model"])
dataset = hydra.utils.instantiate(cfg.task.dataset)
normalizer = dataset.get_normalizer()
policy.set_normalizer(normalizer)
joint_dim = cfg.shape_meta.action.shape[0]
joint_scales = [1.0] * 6 + [0.01]
n_obs_steps = cfg.n_obs_steps

policy.cuda()
policy.eval()

print(f'Loaded: {args.filename}')

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
fig, ax = plt.subplots(1, 2, figsize=(13.4, 5.0), dpi=60)
ax = ax.reshape(-1, 2)
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
cv2.waitKey(1)

action_seq = []
front_image_seq = None
joint_seq = None
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

    skip = args.skip
    if record_manager.status == RecordStatus.TELEOP and time_idx % skip == 0:
        front_image = info["images"]["front"]
        if front_image_seq is None:
            front_image_seq = []
            for _ in range(n_obs_steps):
                front_image_seq.append(np.copy(front_image))
        else:
            front_image_seq.pop(0)
            front_image_seq.append(front_image)
        joint = motion_manager.getAction()
        if joint_seq is None:
            joint_seq = []
            for _ in range(n_obs_steps):
                joint_seq.append(np.copy(joint))
        else:
            joint_seq.pop(0)
            joint_seq.append(joint)
        if len(action_seq) == 0:
            # Load data and normalization
            np_front_image_seq = np.moveaxis(np.array(front_image_seq).astype(np.float32) / 255, -1, 1)
            np_joint_seq = np.array(joint_seq).astype(np.float32)
            np_obs_dict = {
                "image": np.expand_dims(np_front_image_seq, 0),
                "joint": np.expand_dims(np_joint_seq, 0)
            }
            obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=policy.device))
            # import ipdb; ipdb.set_trace()

            # Infer
            action_dict = policy.predict_action(obs_dict)
            np_action_dict = dict_apply(action_dict, lambda x: x.detach().to("cpu").numpy())
            action_seq = list(np_action_dict["action"][0])
        pred_joint = action_seq.pop(0)
        if time_idx % skip_draw == 0:
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
    enable_simulation_image = False
    if enable_simulation_image:
        status_image = record_manager.getStatusImage()
        window_image = cv2.vconcat([info["images"]["front"], info["images"]["side"], status_image])
        cv2.imshow("Simulation image", cv2.cvtColor(window_image, cv2.COLOR_RGB2BGR))
        if win_xy_simulation is not None:
            cv2.moveWindow("Simulation image", *win_xy_simulation)

    # Draw policy images
    skip_draw = 10
    if record_manager.status == RecordStatus.TELEOP and time_idx % skip_draw == 0:
        for _ax in np.ravel(ax):
            _ax.cla()
            _ax.axis("off")

        # Draw camera front_image
        ax[0, 0].imshow(front_image)
        ax[0, 0].set_title("Input front_image", fontsize=20)

        # Plot joint
        T = 100
        ax[0, 1].set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax[0, 1].set_xlim(0, T)
        for joint_idx in range(pred_joint_list.shape[1]):
            ax[0, 1].plot(np.arange(pred_joint_list.shape[0]), pred_joint_list[:, joint_idx] * joint_scales[joint_idx])
        ax[0, 1].set_xlabel("Step", fontsize=20)
        ax[0, 1].set_title("Joint", fontsize=20)
        ax[0, 1].tick_params(axis="x", labelsize=16)
        ax[0, 1].tick_params(axis="y", labelsize=16)
        ax[0, 1].axis("on")

        fig.tight_layout()
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
