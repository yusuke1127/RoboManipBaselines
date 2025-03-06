import argparse
import io
import os

import imageio.v2 as imageio
import matplotlib.pylab as plt
import numpy as np
from tqdm import tqdm

from robo_manip_baselines.common import (
    DataKey,
    DataManager,
    convert_depth_image_to_point_cloud,
)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("teleop_filename", type=str)
parser.add_argument("--skip", default=10, type=int, help="skip", required=False)
parser.add_argument(
    "--save_gif_filename",
    type=str,
    default="./output_" + os.path.splitext(os.path.basename(__file__))[0] + ".gif",
)
args = parser.parse_args()

plt.rcParams["keymap.quit"] = ["q", "escape"]

data_manager = DataManager(env=None)
data_manager.load_data(args.teleop_filename)

camera_names = data_manager.get_meta_data("camera_names").tolist()
tactile_names = data_manager.get_meta_data("tactile_names").tolist()
sensor_names = camera_names + tactile_names

frames = []
fig, ax = plt.subplots(len(sensor_names) + 1, 4)
for ax_idx in range(1, len(sensor_names) + 1):
    ax[ax_idx, 2].remove()
    ax[ax_idx, 3].remove()
    ax[ax_idx, 2] = fig.add_subplot(
        len(sensor_names) + 1, 4, 4 * (ax_idx + 1) - 1, projection="3d"
    )
fig.tight_layout(pad=0.1)

time_range = (
    data_manager.get_data_seq(DataKey.TIME)[0],
    data_manager.get_data_seq(DataKey.TIME)[-1],
)
ax[0, 0].set_xlim(*time_range)
ax[0, 1].set_xlim(*time_range)
ax[0, 2].set_xlim(*time_range)
action_data = data_manager.get_data_seq(DataKey.COMMAND_JOINT_POS)
joint_pos_data = data_manager.get_data_seq(DataKey.MEASURED_JOINT_POS)
q_data = np.concatenate([action_data, joint_pos_data])
ax[0, 0].set_ylim(q_data[:, :-1].min(), q_data[:, :-1].max())
ax00_twin = ax[0, 0].twinx()
ax00_twin.set_ylim(q_data[:, -1].min(), q_data[:, -1].max())
joint_vel_data = data_manager.get_data_seq(DataKey.MEASURED_JOINT_VEL)
ax[0, 1].set_ylim(joint_vel_data.min(), joint_vel_data.max())
wrench_data = data_manager.get_data_seq(DataKey.MEASURED_EEF_WRENCH)
ax[0, 2].set_ylim(wrench_data.min(), wrench_data.max())
measured_eef_data = data_manager.get_data_seq(DataKey.MEASURED_EEF_POSE)
command_eef_data = data_manager.get_data_seq(DataKey.COMMAND_EEF_POSE)
eef_data = np.concatenate([measured_eef_data, command_eef_data])
ax[0, 3].set_ylim(eef_data[:, 0:3].min(), eef_data[:, 0:3].max())
ax03_twin = ax[0, 3].twinx()
ax03_twin.set_ylim(-1.0, 1.0)

scatter_list = [None] * 3
time_list = []
action_list = []
joint_pos_list = []
joint_vel_list = []
wrench_list = []
command_eef_list = []
measured_eef_list = []

break_flag = False


def key_event(event):
    if event.key == "q" or event.key == "escape":
        global break_flag
        break_flag = True


def handle_rgb_image(data_manager, ax, ax_idx, time_idx, rgb_key):
    ax[ax_idx, 0].axis("off")
    rgb_image = data_manager.get_single_data(rgb_key, time_idx)
    rgb_image_skip = 4
    ax[ax_idx, 0].imshow(rgb_image[::rgb_image_skip, ::rgb_image_skip])
    return rgb_image


def handle_depth_image(data_manager, fig, ax, ax_idx, time_idx, sensor_name, depth_key):
    depth_names_count = len(
        [
            k
            for k in data_manager.all_data_seq.keys()
            if k.startswith(f"{sensor_name}_") and ("_depth" in k)
        ]
    )
    assert depth_names_count in (0, 1)
    if not depth_names_count:
        if ax[ax_idx, 1] in fig.axes:
            ax[ax_idx, 1].remove()
        return None
    ax[ax_idx, 1].axis("off")
    depth_image = data_manager.get_single_data(depth_key, time_idx)
    depth_image_skip = 4
    ax[ax_idx, 1].imshow(depth_image[::depth_image_skip, ::depth_image_skip])
    return depth_image


def handle_point_cloud(
    data_manager,
    fig,
    ax,
    ax_idx,
    scatter_list,
    far_clip_list,
    depth_key,
    rgb_image,
    depth_image,
):
    if f"{depth_key}_fovy" not in data_manager.meta_data.keys():
        if ax[ax_idx, 2] in fig.axes:
            ax[ax_idx, 2].remove()
        return
    point_cloud_skip = 10
    small_depth_image = depth_image[::point_cloud_skip, ::point_cloud_skip]
    small_rgb_image = rgb_image[::point_cloud_skip, ::point_cloud_skip]
    fovy = data_manager.get_meta_data(f"{depth_key}_fovy")
    xyz_array, rgb_array = convert_depth_image_to_point_cloud(
        small_depth_image,
        fovy=fovy,
        rgb_image=small_rgb_image,
        far_clip=far_clip_list[ax_idx - 1],
    )
    if not xyz_array.size:
        return
    if scatter_list[ax_idx - 1] is None:

        def get_min_max(v_min, v_max):
            return (
                0.75 * v_min + 0.25 * v_max,
                0.25 * v_min + 0.75 * v_max,
            )

        ax[ax_idx, 2].view_init(elev=-90, azim=-90)
        ax[ax_idx, 2].set_xlim(
            *get_min_max(xyz_array[:, 0].min(), xyz_array[:, 0].max())
        )
        ax[ax_idx, 2].set_ylim(
            *get_min_max(xyz_array[:, 1].min(), xyz_array[:, 1].max())
        )
        ax[ax_idx, 2].set_zlim(
            *get_min_max(xyz_array[:, 2].min(), xyz_array[:, 2].max())
        )
    else:
        scatter_list[ax_idx - 1].remove()
    ax[ax_idx, 2].axis("off")
    ax[ax_idx, 2].set_box_aspect(np.ptp(xyz_array, axis=0))
    scatter_list[ax_idx - 1] = ax[ax_idx, 2].scatter(
        xyz_array[:, 0], xyz_array[:, 1], xyz_array[:, 2], c=rgb_array
    )


for time_idx in tqdm(
    range(0, len(data_manager.get_data_seq(DataKey.TIME)), args.skip),
    desc=ax[0, 0].plot.__name__,
):
    if break_flag:
        break

    time_list.append(data_manager.get_single_data(DataKey.TIME, time_idx))
    action_list.append(
        data_manager.get_single_data(DataKey.COMMAND_JOINT_POS, time_idx)
    )
    joint_pos_list.append(
        data_manager.get_single_data(DataKey.MEASURED_JOINT_POS, time_idx)
    )
    joint_vel_list.append(
        data_manager.get_single_data(DataKey.MEASURED_JOINT_VEL, time_idx)
    )
    wrench_list.append(
        data_manager.get_single_data(DataKey.MEASURED_EEF_WRENCH, time_idx)
    )
    command_eef_list.append(
        data_manager.get_single_data(DataKey.COMMAND_EEF_POSE, time_idx)
    )
    measured_eef_list.append(
        data_manager.get_single_data(DataKey.MEASURED_EEF_POSE, time_idx)
    )

    ax[0, 0].cla()
    ax00_twin.cla()
    ax[0, 0].plot(time_list, np.array(action_list)[:, :-1], linestyle="--", linewidth=3)
    ax[0, 0].set_prop_cycle(None)
    ax[0, 0].plot(time_list, np.array(joint_pos_list)[:, :-1])
    ax00_twin.plot(
        time_list, np.array(action_list)[:, [-1]], linestyle="--", linewidth=3
    )
    ax00_twin.set_prop_cycle(None)
    ax00_twin.plot(time_list, np.array(joint_pos_list)[:, [-1]])
    ax[0, 1].cla()
    ax[0, 1].plot(time_list, np.array(joint_vel_list)[:, :-1])
    ax[0, 2].cla()
    ax[0, 2].plot(time_list, wrench_list)
    ax[0, 3].cla()
    ax03_twin.cla()
    ax[0, 3].plot(
        time_list, np.array(command_eef_list)[:, :3], linestyle="--", linewidth=3
    )
    ax[0, 3].set_prop_cycle(None)
    ax[0, 3].plot(time_list, np.array(measured_eef_list)[:, :3])
    ax03_twin.plot(
        time_list, np.array(command_eef_list)[:, 3:], linestyle="--", linewidth=3
    )
    ax03_twin.set_prop_cycle(None)
    ax03_twin.plot(time_list, np.array(measured_eef_list)[:, 3:])

    far_clip_list = (3.0, 3.0, 0.8)  # [m]
    for ax_idx, sensor_name in enumerate(sensor_names, start=1):
        rgb_key = DataKey.get_rgb_image_key(sensor_name)
        depth_key = DataKey.get_depth_image_key(sensor_name)

        rgb_image = handle_rgb_image(data_manager, ax, ax_idx, time_idx, rgb_key)

        depth_image = handle_depth_image(
            data_manager, fig, ax, ax_idx, time_idx, sensor_name, depth_key
        )

        handle_point_cloud(
            data_manager,
            fig,
            ax,
            ax_idx,
            scatter_list,
            far_clip_list,
            depth_key,
            rgb_image,
            depth_image,
        )

    plt.draw()
    plt.pause(0.001)

    buf = io.BytesIO()
    fig.savefig(buf, format="jpg")
    buf.seek(0)
    img = imageio.imread(buf)
    frames.append(img)
    buf.close()

    fig.canvas.mpl_connect("key_press_event", key_event)

os.makedirs(os.path.dirname(args.save_gif_filename), exist_ok=True)
imageio.mimsave(args.save_gif_filename, frames, fps=10)
print(f"File '{args.save_gif_filename}' has been successfully saved.")
print("Press 'Q' or 'Esc' to quit.")

plt.show()
