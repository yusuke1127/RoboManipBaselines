import os
import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pylab as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import cv2
import torch
from eipl.model import SARNN
from eipl.utils import restore_args, tensor2numpy, deprocess_img, normalization, resize_img
import rospy
from sensor_msgs.msg import CompressedImage
import rtde_control
import rtde_receive
import time

parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, default=None, help=".pth file that PyTorch loads as checkpoint for model")
args = parser.parse_args()

# Setup model
## Restore parameters
dir_name = os.path.split(args.filename)[0]
params = restore_args(os.path.join(dir_name, "args.json"))

## Load dataset
minmax = [params["vmin"], params["vmax"]]
joint_bounds = np.load(os.path.join(dir_name, "joint_bounds.npy"))

## Define model
im_size = 64
joint_dim = 6
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

rnn_state = None
while True:
    # Wait first image
    while raw_front_image is None:
        rate.sleep()

    # Load data and normalization
    front_image = resize_img(np.expand_dims(np.copy(raw_front_image), 0), (im_size, im_size))[0]
    front_image_t = front_image.transpose(2, 0, 1)
    front_image_t = normalization(front_image_t, (0, 255), minmax)
    front_image_t = torch.Tensor(np.expand_dims(front_image_t, 0))
    joint = np.array(rtde_r.getActualQ())[joint_order]
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

    # Send joint command
    velocity = 0.5
    acceleration = 0.5
    dt = 0.5 # 1.0 # [s]
    lookahead_time = 0.2
    gain = 100
    t_start = rtde_c.initPeriod()
    rtde_c.servoJ(pred_joint[joint_order], velocity, acceleration, dt, lookahead_time, gain)
    rtde_c.waitPeriod(t_start)
    time.sleep(dt)

    # Draw policy images
    for j in range(ax.shape[0]):
        for k in range(ax.shape[1]):
            ax[j, k].cla()

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
        ax[0, 2].plot(np.arange(pred_joint_list.shape[0]), pred_joint_list[:, joint_idx])
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
    if key == 32: # space key
        cv2.waitKey(0)
    elif key == 27: # escape key
        break

    rate.sleep()
