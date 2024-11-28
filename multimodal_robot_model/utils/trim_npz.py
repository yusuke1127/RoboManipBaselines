import numpy as np
import glob
import matplotlib
import matplotlib.pylab as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import cv2
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("in_dir", type=str)
parser.add_argument("--out_dir", type=str)
args = parser.parse_args()

matplotlib.use("agg")
fig, ax = plt.subplots(1, 2, figsize=(14, 6), dpi=60)
ax = ax.reshape(-1, 2)
canvas = FigureCanvasAgg(fig)

in_npz_path_list = glob.glob(os.path.join(args.in_dir, "**/*.npz"), recursive=True)
in_npz_path_list.sort()

print("[trim_npz] Usage:")
print("    <right arrow> : Advance time one step")
print("    <left arrow> : Turn back time one step")
print("    <up arrow> : Advance time five step")
print("    <down arrow> : Turn back time five step")
print("    <space> : Finish setting start/end indexes and save trimmed npz")
print("    's' : Set start index")
print("    'e' : Set end index")
print("    'r' : Reset start/end indexes")

for in_npz_path in in_npz_path_list:
    print(f"[trim_npz] Load a npz file: {in_npz_path}")
    npz_data = np.load(in_npz_path)
    joints = npz_data["joint_pos"]
    images = npz_data["front_rgb_image"]
    seq_len = len(joints)

    start_idx, end_idx = None, None

    time_idx = 0
    while True:
        # Clear plot
        for j in range(ax.shape[0]):
            for k in range(ax.shape[1]):
                ax[j, k].cla()

        # Draw image
        ax[0, 0].imshow(images[time_idx])
        ax[0, 0].axis("off")
        ax[0, 0].set_title("image", fontsize=20)

        # Plot joint
        ax[0, 1].set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax[0, 1].set_xlim(0, seq_len)
        for joint_idx in range(joints.shape[1]):
            ax[0, 1].plot(np.arange(seq_len), joints[:, joint_idx])
        ax[0, 1].set_xlabel("Step", fontsize=20)
        ax[0, 1].set_title("Joint", fontsize=20)
        ax[0, 1].tick_params(axis="x", labelsize=16)
        ax[0, 1].tick_params(axis="y", labelsize=16)
        ax[0, 1].vlines(
            time_idx, -np.pi, np.pi, color="black", linestyles="dotted", linewidth=4
        )
        if start_idx is not None:
            ax[0, 1].vlines(start_idx, -np.pi, np.pi, color="blue", linewidth=2)
        if end_idx is not None:
            ax[0, 1].vlines(end_idx, -np.pi, np.pi, color="red", linewidth=2)

        canvas.draw()
        buf = canvas.buffer_rgba()
        draw_image = np.asarray(buf)
        cv2.imshow("image", cv2.cvtColor(draw_image, cv2.COLOR_RGB2BGR))

        key = cv2.waitKey(1)
        if key == ord("s"):
            start_idx = time_idx
        elif key == ord("e"):
            end_idx = time_idx
        elif key == ord("r"):
            start_idx, end_idx = None, None
        elif key == 81:  # left key
            time_idx = max(time_idx - 1, 0)
        elif key == 82:  # up key
            time_idx = min(time_idx + 5, seq_len - 1)
        elif key == 83:  # right key
            time_idx = min(time_idx + 1, seq_len - 1)
        elif key == 84:  # down key
            time_idx = max(time_idx - 5, 0)
        elif key == 32:  # space key
            if (
                (start_idx is not None)
                and (end_idx is not None)
                and (start_idx >= end_idx)
            ):
                print("[trim_npz] Error: start index is larger than end index.")
            else:
                break

    print(f"[trim_npz] start index: {start_idx}, end index: {end_idx}")
    cv2.imshow("image", cv2.cvtColor(np.full_like(draw_image, 255), cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)

    if (start_idx is not None) or (end_idx is not None):
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = seq_len

        npz_data = dict(npz_data)
        for key in npz_data:
            npz_data[key] = npz_data[key][start_idx:end_idx]

        if args.out_dir is None:
            out_npz_path = in_npz_path
        else:
            out_npz_path = os.path.join(args.out_dir, os.path.basename(in_npz_path))
        print(f"[trim_npz] Save a npz file: {out_npz_path}")
        np.savez(
            out_npz_path,
            time=npz_data["time"],
            joint=npz_data["joint_pos"],
            wrench=npz_data["wrench"],
            front_image=npz_data["front_rgb_image"],
            side_image=npz_data["side_rgb_image"],
        )
