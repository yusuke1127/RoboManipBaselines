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

break_flag = False


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("teleop_filename", type=str)
    parser.add_argument("--skip", default=10, type=int, help="skip", required=False)
    parser.add_argument(
        "--save_gif_filename",
        type=str,
        default="./output_" + os.path.splitext(os.path.basename(__file__))[0] + ".gif",
    )
    return parser.parse_args()


def key_event(event):
    if event.key == "q" or event.key == "escape":
        global break_flag
        break_flag = True


class VisualizeData:
    def __init__(self, teleop_filename, skip, save_gif_filename):
        self.skip = skip
        self.save_gif_filename = save_gif_filename

        self.data_manager = DataManager(env=None)
        self.data_manager.load_data(teleop_filename)

        camera_names = self.data_manager.get_meta_data("camera_names").tolist()
        tactile_names = self.data_manager.get_meta_data("tactile_names").tolist()
        self.sensor_names = camera_names + tactile_names

        plt.rcParams["keymap.quit"] = ["q", "escape"]
        self.frames = []
        self.fig, self.ax = plt.subplots(
            len(self.sensor_names) + 1, 4, constrained_layout=True
        )
        for ax_idx in range(1, len(self.sensor_names) + 1):
            self.ax[ax_idx, 2].remove()
            self.ax[ax_idx, 3].remove()
            self.ax[ax_idx, 2] = self.fig.add_subplot(
                len(self.sensor_names) + 1, 4, 4 * (ax_idx + 1) - 1, projection="3d"
            )

        time_range = (
            self.data_manager.get_data_seq(DataKey.TIME)[0],
            self.data_manager.get_data_seq(DataKey.TIME)[-1],
        )
        self.ax[0, 0].set_xlim(*time_range)
        self.ax[0, 1].set_xlim(*time_range)
        self.ax[0, 2].set_xlim(*time_range)

        action_data = self.data_manager.get_data_seq(DataKey.COMMAND_JOINT_POS)
        joint_pos_data = self.data_manager.get_data_seq(DataKey.MEASURED_JOINT_POS)
        q_data = np.concatenate([action_data, joint_pos_data])
        self.ax[0, 0].set_ylim(q_data[:, :-1].min(), q_data[:, :-1].max())
        self.ax00_twin = self.ax[0, 0].twinx()
        self.ax00_twin.set_ylim(q_data[:, -1].min(), q_data[:, -1].max())
        joint_vel_data = self.data_manager.get_data_seq(DataKey.MEASURED_JOINT_VEL)
        self.ax[0, 1].set_ylim(joint_vel_data.min(), joint_vel_data.max())
        wrench_data = self.data_manager.get_data_seq(DataKey.MEASURED_EEF_WRENCH)
        self.ax[0, 2].set_ylim(wrench_data.min(), wrench_data.max())
        measured_eef_data = self.data_manager.get_data_seq(DataKey.MEASURED_EEF_POSE)
        command_eef_data = self.data_manager.get_data_seq(DataKey.COMMAND_EEF_POSE)
        eef_data = np.concatenate([measured_eef_data, command_eef_data])
        self.ax[0, 3].set_ylim(eef_data[:, 0:3].min(), eef_data[:, 0:3].max())
        self.ax03_twin = self.ax[0, 3].twinx()
        self.ax03_twin.set_ylim(-1.0, 1.0)

        self.scatter_list = [None] * 3
        self.time_list = []
        self.action_list = []
        self.joint_pos_list = []
        self.joint_vel_list = []
        self.wrench_list = []
        self.command_eef_list = []
        self.measured_eef_list = []

    def handle_rgb_image(self, ax_idx, time_idx, rgb_key):
        self.ax[ax_idx, 0].axis("off")
        rgb_image = self.data_manager.get_single_data(rgb_key, time_idx)
        rgb_image_skip = 4
        self.ax[ax_idx, 0].imshow(rgb_image[::rgb_image_skip, ::rgb_image_skip])
        return rgb_image

    def handle_depth_image(self, ax_idx, time_idx, sensor_name, depth_key):
        depth_names_count = len(
            [
                k
                for k in self.data_manager.all_data_seq.keys()
                if k.startswith(f"{sensor_name}_") and ("_depth" in k)
            ]
        )
        assert depth_names_count in (0, 1)
        if not depth_names_count:
            if self.ax[ax_idx, 1] in self.fig.axes:
                self.ax[ax_idx, 1].remove()
            return None
        self.ax[ax_idx, 1].axis("off")
        depth_image = self.data_manager.get_single_data(depth_key, time_idx)
        depth_image_skip = 4
        self.ax[ax_idx, 1].imshow(depth_image[::depth_image_skip, ::depth_image_skip])
        return depth_image

    def handle_point_cloud(
        self,
        ax_idx,
        scatter_list,
        far_clip_list,
        depth_key,
        rgb_image,
        depth_image,
    ):
        if f"{depth_key}_fovy" not in self.data_manager.meta_data.keys():
            if self.ax[ax_idx, 2] in self.fig.axes:
                self.ax[ax_idx, 2].remove()
            return
        point_cloud_skip = 10
        small_depth_image = depth_image[::point_cloud_skip, ::point_cloud_skip]
        small_rgb_image = rgb_image[::point_cloud_skip, ::point_cloud_skip]
        fovy = self.data_manager.get_meta_data(f"{depth_key}_fovy")
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

            self.ax[ax_idx, 2].view_init(elev=-90, azim=-90)
            self.ax[ax_idx, 2].set_xlim(
                *get_min_max(xyz_array[:, 0].min(), xyz_array[:, 0].max())
            )
            self.ax[ax_idx, 2].set_ylim(
                *get_min_max(xyz_array[:, 1].min(), xyz_array[:, 1].max())
            )
            self.ax[ax_idx, 2].set_zlim(
                *get_min_max(xyz_array[:, 2].min(), xyz_array[:, 2].max())
            )
        else:
            scatter_list[ax_idx - 1].remove()
        self.ax[ax_idx, 2].axis("off")
        self.ax[ax_idx, 2].set_box_aspect(np.ptp(xyz_array, axis=0))
        scatter_list[ax_idx - 1] = self.ax[ax_idx, 2].scatter(
            xyz_array[:, 0], xyz_array[:, 1], xyz_array[:, 2], c=rgb_array
        )

    def plot(self):
        for time_idx in tqdm(
            range(0, len(self.data_manager.get_data_seq(DataKey.TIME)), self.skip),
            desc=self.ax[0, 0].plot.__name__,
        ):
            if break_flag:
                break

            self.time_list.append(
                self.data_manager.get_single_data(DataKey.TIME, time_idx)
            )
            self.action_list.append(
                self.data_manager.get_single_data(DataKey.COMMAND_JOINT_POS, time_idx)
            )
            self.joint_pos_list.append(
                self.data_manager.get_single_data(DataKey.MEASURED_JOINT_POS, time_idx)
            )
            self.joint_vel_list.append(
                self.data_manager.get_single_data(DataKey.MEASURED_JOINT_VEL, time_idx)
            )
            self.wrench_list.append(
                self.data_manager.get_single_data(DataKey.MEASURED_EEF_WRENCH, time_idx)
            )
            self.command_eef_list.append(
                self.data_manager.get_single_data(DataKey.COMMAND_EEF_POSE, time_idx)
            )
            self.measured_eef_list.append(
                self.data_manager.get_single_data(DataKey.MEASURED_EEF_POSE, time_idx)
            )

            self.ax[0, 0].cla()
            self.ax00_twin.cla()
            self.ax[0, 0].plot(
                self.time_list,
                np.array(self.action_list)[:, :-1],
                linestyle="--",
                linewidth=3,
            )
            self.ax[0, 0].set_prop_cycle(None)
            self.ax[0, 0].plot(self.time_list, np.array(self.joint_pos_list)[:, :-1])
            self.ax00_twin.plot(
                self.time_list,
                np.array(self.action_list)[:, [-1]],
                linestyle="--",
                linewidth=3,
            )
            self.ax00_twin.set_prop_cycle(None)
            self.ax00_twin.plot(self.time_list, np.array(self.joint_pos_list)[:, [-1]])
            self.ax[0, 1].cla()
            self.ax[0, 1].plot(self.time_list, np.array(self.joint_vel_list)[:, :-1])
            self.ax[0, 2].cla()
            self.ax[0, 2].plot(self.time_list, self.wrench_list)
            self.ax[0, 3].cla()
            self.ax03_twin.cla()
            self.ax[0, 3].plot(
                self.time_list,
                np.array(self.command_eef_list)[:, :3],
                linestyle="--",
                linewidth=3,
            )
            self.ax[0, 3].set_prop_cycle(None)
            self.ax[0, 3].plot(self.time_list, np.array(self.measured_eef_list)[:, :3])
            self.ax03_twin.plot(
                self.time_list,
                np.array(self.command_eef_list)[:, 3:],
                linestyle="--",
                linewidth=3,
            )
            self.ax03_twin.set_prop_cycle(None)
            self.ax03_twin.plot(self.time_list, np.array(self.measured_eef_list)[:, 3:])

            far_clip_list = (3.0, 3.0, 0.8)  # [m]
            for ax_idx, sensor_name in enumerate(self.sensor_names, start=1):
                rgb_key = DataKey.get_rgb_image_key(sensor_name)
                depth_key = DataKey.get_depth_image_key(sensor_name)

                rgb_image = self.handle_rgb_image(ax_idx, time_idx, rgb_key)

                depth_image = self.handle_depth_image(
                    ax_idx, time_idx, sensor_name, depth_key
                )

                self.handle_point_cloud(
                    ax_idx,
                    self.scatter_list,
                    far_clip_list,
                    depth_key,
                    rgb_image,
                    depth_image,
                )

            plt.draw()
            plt.pause(0.001)

            buf = io.BytesIO()
            self.fig.savefig(buf, format="jpg")
            buf.seek(0)
            img = imageio.imread(buf)
            self.frames.append(img)
            buf.close()

            self.fig.canvas.mpl_connect("key_press_event", key_event)

        os.makedirs(os.path.dirname(self.save_gif_filename), exist_ok=True)
        imageio.mimsave(self.save_gif_filename, self.frames, fps=10)
        print(
            f"[{self.__class__.__name__}] File '{self.save_gif_filename}' has been successfully saved."
        )
        print(f"[{self.__class__.__name__}] Press 'Q' or 'Esc' to quit.")

        plt.show()


if __name__ == "__main__":
    args = parse_argument()
    viz = VisualizeData(args.teleop_filename, args.skip, args.save_gif_filename)
    viz.plot()
