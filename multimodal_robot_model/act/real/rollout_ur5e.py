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
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../third_party/act"))
from policy import ACTPolicy
from eipl.utils import restore_args, tensor2numpy, deprocess_img, normalization, resize_img
import rospy
from sensor_msgs.msg import CompressedImage
import rtde_control
import rtde_receive
import time

# command line parameters
parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, default=None, help=".pth file that PyTorch loads as checkpoint for policy")
parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
parser.add_argument('--ckpt_name', default='policy_best.ckpt', type=str, help='ckpt_name')
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

# Command line parameters
ckpt_dir = args.ckpt_dir
policy_class = args.policy_class
task_name = args.task_name
seed = args.seed

# Get task parameters
from multimodal_robot_model.act import SIM_TASK_CONFIGS
task_config = SIM_TASK_CONFIGS[task_name]
camera_names = task_config['camera_names']

# Set fixed parameters
apply_crop = False
apply_resize = False
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

ckpt_name = args.ckpt_name

## Define policy
im_size = (640, 480)
joint_dim = 6
policy = ACTPolicy(policy_config)
def forward_fook(self, _input, _output):
    # Output of MultiheadAttention is a tuple (attn_output, attn_output_weights)
    # https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
    self.correlation_mat = _output[1][0].detach().cpu().numpy()
for layer in policy.model.transformer.encoder.layers:
    layer.self_attn.correlation_mat = None
    layer.self_attn.register_forward_hook(forward_fook)

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

# Setup window for policy image
matplotlib.use("agg")
fig, ax = plt.subplots(2, max(2, enc_layers), figsize=(13.4, 6.0), dpi=60)
for _ax in np.ravel(ax):
    _ax.cla()
    _ax.axis("off")
canvas = FigureCanvasAgg(fig)
pred_joint_list = np.empty((0, joint_dim))
canvas.draw()
buf = canvas.buffer_rgba()
policy_image = np.asarray(buf)
cv2.imshow("Policy image", cv2.cvtColor(policy_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(1)

# Setup ROS
rospy.init_node("rollout_real")

raw_front_image = None
def callback(msg):
    global raw_front_image
    _raw_front_image = np.frombuffer(msg.data, np.uint8)
    _raw_front_image = cv2.imdecode(_raw_front_image, cv2.IMREAD_COLOR)
    raw_front_image = cv2.cvtColor(_raw_front_image, cv2.COLOR_BGR2RGB).astype(np.uint8)

rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, callback, queue_size=1)
rate = rospy.Rate(10)

# Setup robot
robot_ip = "192.168.11.4"
rtde_c = rtde_control.RTDEControlInterface(robot_ip)
rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
rtde_c.endFreedriveMode()
joint_order = [2,1,0,3,4,5]

## Go to initial posture
print("- Start initial posture")
init_q = np.array([1.5929644, -1.9200722, 1.1177821, -1.243697, -1.5707978, -0.45310742])
velocity = 0.5
acceleration = 0.5
dt = 10.0 # [s]
lookahead_time = 0.2
gain = 100
t_start = rtde_c.initPeriod()
rtde_c.servoJ(init_q[joint_order], velocity, acceleration, dt, lookahead_time, gain)
rtde_c.waitPeriod(t_start)
time.sleep(dt)
print("- Finish initial posture")

state = None
all_actions_history = []
while True:
    # Wait first image
    while raw_front_image is None:
        rate.sleep()

    # Load data and normalization
    front_image = resize_img(np.expand_dims(np.copy(raw_front_image), 0), im_size)[0]
    front_image_t = front_image.transpose(2, 0, 1)
    front_image_t = front_image_t.astype(np.float32) / 255.0
    front_image_t = torch.Tensor(np.expand_dims(front_image_t, 0)).cuda().unsqueeze(0)
    joint = np.array(rtde_r.getActualQ())[joint_order]
    joint_t = pre_process(joint)
    joint_t = torch.Tensor(np.expand_dims(joint_t, 0)).cuda()

    # Infer
    all_actions = policy(joint_t, front_image_t)[0]
    all_actions_history.append(tensor2numpy(all_actions))
    if len(all_actions_history) > args.chunk_size:
        all_actions_history.pop(0)

    # Apply temporal ensembling
    k = 0.01
    exp_weights = np.exp(-k * np.arange(len(all_actions_history)))
    exp_weights = exp_weights / exp_weights.sum()
    action = np.zeros(joint_dim)
    for action_idx, _all_actions in enumerate(reversed(all_actions_history)):
        action += exp_weights[::-1][action_idx] * _all_actions[action_idx]
    pred_joint = post_process(action)
    pred_joint_list = np.concatenate([pred_joint_list, np.expand_dims(pred_joint, 0)])

    # Send joint command
    velocity = 0.5
    acceleration = 0.5
    dt = 0.4 # 1.0 # [s]
    lookahead_time = 0.2
    gain = 100
    t_start = rtde_c.initPeriod()
    rtde_c.servoJ(pred_joint[joint_order], velocity, acceleration, dt, lookahead_time, gain)
    rtde_c.waitPeriod(t_start)
    time.sleep(dt)

    # Draw policy images
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
        ax[0, 1].plot(np.arange(pred_joint_list.shape[0]), pred_joint_list[:, joint_idx])
    ax[0, 1].set_xlabel("Step", fontsize=20)
    ax[0, 1].set_title("Joint", fontsize=20)
    ax[0, 1].tick_params(axis="x", labelsize=16)
    ax[0, 1].tick_params(axis="y", labelsize=16)
    ax[0, 1].axis("on")

    # Draw predicted front_image
    for layer_idx, layer in enumerate(policy.model.transformer.encoder.layers):
        if layer.self_attn.correlation_mat is None:
            continue
        ax[1, layer_idx].imshow(layer.self_attn.correlation_mat[2:, 1].reshape((15, 20)))
        ax[1, layer_idx].set_title(f"Attention ({layer_idx})", fontsize=20)

    fig.tight_layout()
    canvas.draw()
    buf = canvas.buffer_rgba()
    policy_image = np.asarray(buf)
    cv2.imshow("Policy image", cv2.cvtColor(policy_image, cv2.COLOR_RGB2BGR))
    # plt.draw()
    # plt.pause(0.001)

    key = cv2.waitKey(1)

    # Manage status
    if key == 32: # space key
        cv2.waitKey(0)
    elif key == 27: # escape key
        break

    rate.sleep()
