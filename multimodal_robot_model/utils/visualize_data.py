import argparse
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2
from multimodal_robot_model.common import DataKey, RecordManager, convertDepthImageToPointCloud

parser = argparse.ArgumentParser()
parser.add_argument("teleop_filename", type=str)
parser.add_argument('--skip', default=10, type=int, help='skip', required=False)
args = parser.parse_args()

plt.rcParams["keymap.quit"] = ["q", "escape"]
fig, ax = plt.subplots(4, 4)
for ax_idx in range(1, 4):
    ax[ax_idx, 2].remove()
    ax[ax_idx, 3].remove()
    ax[ax_idx, 2] = fig.add_subplot(4, 4, 4 * (ax_idx + 1) - 1, projection="3d")
fig.tight_layout(pad=0.1)

record_manager = RecordManager(env=None)
record_manager.loadData(args.teleop_filename)

time_range = (record_manager.data_seq["time"][0], record_manager.data_seq["time"][-1])
ax[0, 0].set_xlim(*time_range)
ax[0, 1].set_xlim(*time_range)
ax[0, 2].set_xlim(*time_range)
action_data = record_manager.data_seq[DataKey.ACTION.key()]
joint_pos_data = record_manager.data_seq[DataKey.JOINT_POS.key()]
q_data = np.concatenate([action_data, joint_pos_data])
ax[0, 0].set_ylim(q_data[:, :-1].min(), q_data[:, :-1].max())
ax00_twin = ax[0, 0].twinx()
ax00_twin.set_ylim(q_data[:, -1].min(), q_data[:, -1].max())
joint_vel_data = record_manager.data_seq[DataKey.JOINT_VEL.key()]
ax[0, 1].set_ylim(joint_vel_data.min(), joint_vel_data.max())
wrench_data = record_manager.data_seq[DataKey.WRENCH.key()]
ax[0, 2].set_ylim(wrench_data.min(), wrench_data.max())
measured_eef_data = record_manager.data_seq[DataKey.MEASURED_EEF.key()]
command_eef_data = record_manager.data_seq[DataKey.COMMAND_EEF.key()]
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

for time_idx in range(0, len(record_manager.data_seq["time"]), args.skip):
    if break_flag:
        break

    time_list.append(record_manager.data_seq["time"][time_idx])
    action_list.append(record_manager.getSingleData(DataKey.ACTION, time_idx))
    joint_pos_list.append(record_manager.getSingleData(DataKey.JOINT_POS, time_idx))
    joint_vel_list.append(record_manager.getSingleData(DataKey.JOINT_VEL, time_idx))
    wrench_list.append(record_manager.getSingleData(DataKey.WRENCH, time_idx))
    command_eef_list.append(record_manager.getSingleData(DataKey.COMMAND_EEF, time_idx))
    measured_eef_list.append(record_manager.getSingleData(DataKey.MEASURED_EEF, time_idx))

    ax[0, 0].cla()
    ax00_twin.cla()
    ax[0, 0].plot(time_list, np.array(action_list)[:, :-1], linestyle="--", linewidth=3)
    ax[0, 0].set_prop_cycle(None)
    ax[0, 0].plot(time_list, np.array(joint_pos_list)[:, :-1])
    ax00_twin.plot(time_list, np.array(action_list)[:, [-1]], linestyle="--", linewidth=3)
    ax00_twin.set_prop_cycle(None)
    ax00_twin.plot(time_list, np.array(joint_pos_list)[:, [-1]])
    ax[0, 1].cla()
    ax[0, 1].plot(time_list, np.array(joint_vel_list)[:, :-1])
    ax[0, 2].cla()
    ax[0, 2].plot(time_list, wrench_list)
    ax[0, 3].cla()
    ax03_twin.cla()
    ax[0, 3].plot(time_list, np.array(command_eef_list)[:, :3], linestyle="--", linewidth=3)
    ax[0, 3].set_prop_cycle(None)
    ax[0, 3].plot(time_list, np.array(measured_eef_list)[:, :3])
    ax03_twin.plot(time_list, np.array(command_eef_list)[:, 3:], linestyle="--", linewidth=3)
    ax03_twin.set_prop_cycle(None)
    ax03_twin.plot(time_list, np.array(measured_eef_list)[:, 3:])

    dist_thre_list = (3.0, 3.0, 0.8) # [m]
    for ax_idx, (rgb_key, depth_key) in enumerate(
            ((DataKey.FRONT_RGB_IMAGE, DataKey.FRONT_DEPTH_IMAGE),
             (DataKey.SIDE_RGB_IMAGE, DataKey.SIDE_DEPTH_IMAGE),
             (DataKey.HAND_RGB_IMAGE, DataKey.HAND_DEPTH_IMAGE)),
            start=1):
        ax[ax_idx, 0].axis("off")
        rgb_image = record_manager.getSingleData(rgb_key, time_idx)
        rgb_image_skip = 4
        ax[ax_idx, 0].imshow(rgb_image[::rgb_image_skip, ::rgb_image_skip])

        ax[ax_idx, 1].axis("off")
        depth_image = record_manager.getSingleData(depth_key, time_idx)
        depth_iamge_skip = 4
        ax[ax_idx, 1].imshow(depth_image[::depth_iamge_skip, ::depth_iamge_skip])

        point_cloud_skip = 10
        small_depth_image = depth_image[::point_cloud_skip, ::point_cloud_skip]
        small_rgb_image = rgb_image[::point_cloud_skip, ::point_cloud_skip]
        fovy = record_manager.data_seq[f"{depth_key.key()}_fovy"].tolist()
        xyz_array, rgb_array = convertDepthImageToPointCloud(
            small_depth_image, fovy=fovy, rgb_image=small_rgb_image, dist_thre=dist_thre_list[ax_idx - 1])
        if scatter_list[ax_idx - 1] is None:
            get_min_max = lambda v_min, v_max: (0.75 * v_min + 0.25 * v_max, 0.25 * v_min + 0.75 * v_max)
            ax[ax_idx, 2].view_init(elev=-90, azim=-90)
            ax[ax_idx, 2].set_xlim(*get_min_max(xyz_array[:, 0].min(), xyz_array[:, 0].max()))
            ax[ax_idx, 2].set_ylim(*get_min_max(xyz_array[:, 1].min(), xyz_array[:, 1].max()))
            ax[ax_idx, 2].set_zlim(*get_min_max(xyz_array[:, 2].min(), xyz_array[:, 2].max()))
        else:
            scatter_list[ax_idx - 1].remove()
        ax[ax_idx, 2].axis("off")
        ax[ax_idx, 2].set_aspect("equal")
        scatter_list[ax_idx - 1] = ax[ax_idx, 2].scatter(
            xyz_array[:, 0], xyz_array[:, 1], xyz_array[:, 2], c=rgb_array)

    plt.draw()
    plt.pause(0.001)
    fig.canvas.mpl_connect("key_press_event", key_event)

plt.show()
