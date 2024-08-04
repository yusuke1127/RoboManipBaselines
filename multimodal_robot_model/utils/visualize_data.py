import argparse
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2
from multimodal_robot_model.demos.Utils_UR5eCableEnv import RecordKey, RecordManager, convertDepthImageToPointCloud

parser = argparse.ArgumentParser()
parser.add_argument("teleop_filename", type=str, help="whether to enable 3d plot")
args = parser.parse_args()

fig, ax = plt.subplots(4, 3)
for ax_idx in range(1,4):
    ax[ax_idx, -1].remove()
    ax[ax_idx, -1] = fig.add_subplot(4, 3, 3 * (ax_idx + 1), projection="3d")
fig.tight_layout(pad=0.1)

record_manager = RecordManager(env=None)
record_manager.loadData(args.teleop_filename)

time_range = (record_manager.data_seq["time"][0], record_manager.data_seq["time"][-1])
ax[0, 0].set_xlim(*time_range)
ax[0, 1].set_xlim(*time_range)
ax[0, 2].set_xlim(*time_range)
action_data = record_manager.data_seq[RecordKey.ACTION.key()]
joint_pos_data = record_manager.data_seq[RecordKey.JOINT_POS.key()]
ax[0, 0].set_ylim(np.min((joint_pos_data[:, :-1], action_data[:, :-1])), np.max((joint_pos_data[:, :-1], action_data[:, :-1])))
ax00_twin = ax[0, 0].twinx()
ax00_twin.set_ylim(np.min((joint_pos_data[:, -1], action_data[:, -1])), np.max((joint_pos_data[:, -1], action_data[:, -1])))
joint_vel_data = record_manager.data_seq[RecordKey.JOINT_VEL.key()]
ax[0, 1].set_ylim(joint_vel_data.min(), joint_vel_data.max())
wrench_data = record_manager.data_seq[RecordKey.WRENCH.key()]
ax[0, 2].set_ylim(wrench_data.min(), wrench_data.max())

scatter_list = [None] * 3
time_list = []
action_list = []
joint_pos_list = []
joint_vel_list = []
wrench_list = []

time_skip = 10
for time_idx in range(0, len(record_manager.data_seq["time"]), time_skip):
    time_list.append(record_manager.data_seq["time"][time_idx])
    action_list.append(record_manager.getSingleData(RecordKey.ACTION, time_idx))
    joint_pos_list.append(record_manager.getSingleData(RecordKey.JOINT_POS, time_idx))
    joint_vel_list.append(record_manager.getSingleData(RecordKey.JOINT_VEL, time_idx))
    wrench_list.append(record_manager.getSingleData(RecordKey.WRENCH, time_idx))
    ax[0, 0].cla()
    ax00_twin.cla()
    ax[0, 0].plot(time_list, np.array(action_list)[:, :-1], linestyle="--")
    ax[0, 0].set_prop_cycle(None)
    ax[0, 0].plot(time_list, np.array(joint_pos_list)[:, :-1])
    ax00_twin.plot(time_list, np.array(action_list)[:, [-1]], linestyle="--")
    ax00_twin.set_prop_cycle(None)
    ax00_twin.plot(time_list, np.array(joint_pos_list)[:, [-1]])
    ax[0, 1].cla()
    ax[0, 1].plot(time_list, np.array(joint_vel_list)[:, :-1])
    ax[0, 2].cla()
    ax[0, 2].plot(time_list, wrench_list)

    for ax_idx, (record_rgb_key, record_depth_key) in enumerate(
            ((RecordKey.FRONT_RGB_IMAGE, RecordKey.FRONT_DEPTH_IMAGE),
             (RecordKey.SIDE_RGB_IMAGE, RecordKey.SIDE_DEPTH_IMAGE),
             (RecordKey.HAND_RGB_IMAGE, RecordKey.HAND_DEPTH_IMAGE)),
            start=1):
        ax[ax_idx, 0].axis("off")
        rgb_image = record_manager.getSingleData(record_rgb_key, time_idx)
        rgb_image_skip = 4
        ax[ax_idx, 0].imshow(rgb_image[::rgb_image_skip, ::rgb_image_skip])

        ax[ax_idx, 1].axis("off")
        depth_image = record_manager.getSingleData(record_depth_key, time_idx)
        depth_iamge_skip = 4
        ax[ax_idx, 1].imshow(depth_image[::depth_iamge_skip, ::depth_iamge_skip])

        point_cloud_skip = 10
        small_depth_image = depth_image[::point_cloud_skip, ::point_cloud_skip]
        fovy = record_manager.data_seq[f"{record_depth_key.key()}_fovy"].tolist()
        xyz_array = convertDepthImageToPointCloud(small_depth_image, fovy=fovy, dist_thre=3.0)
        if scatter_list[ax_idx - 1] is None:
            ax[ax_idx, 2].view_init(elev=-90, azim=-90)
            ax[ax_idx, 2].set_xlim(xyz_array[:, 0].min(), xyz_array[:, 0].max())
            ax[ax_idx, 2].set_ylim(xyz_array[:, 1].min(), xyz_array[:, 1].max())
            ax[ax_idx, 2].set_zlim(xyz_array[:, 2].min(), xyz_array[:, 2].max())
        else:
            scatter_list[ax_idx - 1].remove()
        ax[ax_idx, 2].set_aspect("equal")
        scatter_list[ax_idx - 1] = ax[ax_idx, 2].scatter(
            xyz_array[:, 0], xyz_array[:, 1], xyz_array[:, 2], c=xyz_array[:, 2], cmap="viridis")

    plt.draw()
    plt.pause(0.001)

plt.show()
