import argparse
import sys
import time
from abc import ABCMeta, abstractmethod

import cv2
import matplotlib
import matplotlib.pylab as plt
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg

from robo_manip_baselines.common import (
    DataKey,
    MotionManager,
    Phase,
    PhaseManager,
    PhaseOrder,
)


class RolloutBase(metaclass=ABCMeta):
    def __init__(self):
        self.setup_args()

        self.setup_policy()

        self.setup_env()

        self.setup_plot()

        # Setup motion manager
        self.motion_manager = MotionManager(self.env)

        # Setup phase manager
        self.phase_manager = PhaseManager(self.env, PhaseOrder.ROLLOUT)

    def run(self):
        self.obs, self.info = self.env.reset(seed=self.args.seed)

        inference_duration_list = []
        while True:
            if self.phase_manager.phase == Phase.ROLLOUT:
                inference_start_time = time.time()
                inference_called = self.infer_policy()
                inference_duration = time.time() - inference_start_time
                if inference_called:
                    inference_duration_list.append(inference_duration)

            self.set_arm_command()
            self.set_gripper_command()

            action = self.motion_manager.get_command_data(DataKey.COMMAND_JOINT_POS)
            self.obs, _, _, _, self.info = self.env.step(action)

            if self.phase_manager.phase == Phase.ROLLOUT:
                self.draw_plot()

            # Manage phase
            key = cv2.waitKey(1)
            if self.phase_manager.phase == Phase.INITIAL:
                initial_duration = 1.0  # [s]
                if (
                    not self.args.wait_before_start
                    and self.phase_manager.get_phase_elapsed_duration()
                    > initial_duration
                ) or (self.args.wait_before_start and key == ord("n")):
                    self.phase_manager.set_next_phase()
            elif self.phase_manager.phase == Phase.PRE_REACH:
                pre_reach_duration = 0.7  # [s]
                if self.phase_manager.get_phase_elapsed_duration() > pre_reach_duration:
                    self.phase_manager.set_next_phase()
            elif self.phase_manager.phase == Phase.REACH:
                reach_duration = 0.3  # [s]
                if self.phase_manager.get_phase_elapsed_duration() > reach_duration:
                    self.phase_manager.set_next_phase()
            elif self.phase_manager.phase == Phase.GRASP:
                grasp_duration = 0.5  # [s]
                if self.phase_manager.get_phase_elapsed_duration() > grasp_duration:
                    self.auto_time_idx = 0
                    print("[RolloutBase] Press the 'n' key to finish policy rollout.")
                    self.phase_manager.set_next_phase()
            elif self.phase_manager.phase == Phase.ROLLOUT:
                self.auto_time_idx += 1
                if key == ord("n"):
                    print("[RolloutBase] Statistics on policy inference")
                    policy_model_size = self.calc_model_size()
                    print(
                        f"  - Policy model size [MB] | {policy_model_size / 1024**2:.2f}"
                    )
                    gpu_memory_usage = torch.cuda.max_memory_reserved()
                    print(
                        f"  - GPU memory usage [GB] | {gpu_memory_usage / 1024**3:.3f}"
                    )
                    inference_duration_list = np.array(inference_duration_list)
                    print(
                        "  - Inference duration [s] | "
                        f"mean: {inference_duration_list.mean():.2e}, std: {inference_duration_list.std():.2e} "
                        f"min: {inference_duration_list.min():.2e}, max: {inference_duration_list.max():.2e}"
                    )
                    print("[RolloutBase] Press the 'n' key to exit.")
                    self.phase_manager.set_next_phase()
            elif self.phase_manager.phase == Phase.END:
                if key == ord("n"):
                    break
            if key == 27:  # escape key
                break

        # self.env.close()

    def setup_args(self, parser=None, argv=None):
        if parser is None:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )

        parser.add_argument(
            "--world_idx",
            type=int,
            default=0,
            help="index of the simulation world (0-5)",
        )
        parser.add_argument(
            "--skip",
            type=int,
            help="step interval to infer policy",
        )
        parser.add_argument(
            "--skip_draw",
            type=int,
            help="step interval to draw the plot",
        )
        parser.add_argument(
            "--scale_dt",
            type=float,
            help="dt scale of environment (used only in real-world environments)",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="random seed",
        )
        parser.add_argument(
            "--win_xy_policy",
            type=int,
            nargs=2,
            help="xy position of window to plot policy information",
        )
        parser.add_argument(
            "--wait_before_start",
            action="store_true",
            help="whether to wait a key input before starting simulation",
        )

        if argv is None:
            argv = sys.argv
        self.args = parser.parse_args(argv[1:])

    @abstractmethod
    def setup_policy(self):
        pass

    @abstractmethod
    def setup_env(self):
        pass

    def setup_plot(self, fig_ax=None):
        matplotlib.use("agg")
        if fig_ax is None:
            self.fig, self.ax = plt.subplots(
                1, 1, figsize=(13.5, 6.0), dpi=60, squeeze=False
            )
        else:
            self.fig, self.ax = fig_ax
        for _ax in np.ravel(self.ax):
            _ax.cla()
            _ax.axis("off")
        self.canvas = FigureCanvasAgg(self.fig)
        self.canvas.draw()
        policy_image = np.asarray(self.canvas.buffer_rgba())
        cv2.imshow("Policy image", cv2.cvtColor(policy_image, cv2.COLOR_RGB2BGR))
        if self.args.win_xy_policy is not None:
            cv2.moveWindow("Policy image", *self.args.win_xy_policy)
        cv2.waitKey(1)

    def calc_model_size(self):
        # https://discuss.pytorch.org/t/finding-model-size/130275/2
        model = self.policy
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return param_size + buffer_size

    def set_arm_command(self):
        if self.phase_manager.phase == Phase.ROLLOUT:
            self.motion_manager.arm_joint_pos = self.pred_action[
                self.env.unwrapped.arm_joint_idxes
            ]

    def set_gripper_command(self):
        if self.phase_manager.phase == Phase.GRASP:
            self.motion_manager.gripper_joint_pos = self.env.action_space.high[
                self.env.unwrapped.gripper_joint_idxes
            ]
        elif self.phase_manager.phase == Phase.ROLLOUT:
            self.motion_manager.gripper_joint_pos = np.clip(
                self.pred_action[self.env.unwrapped.gripper_joint_idxes],
                self.env.action_space.low[self.env.unwrapped.gripper_joint_idxes],
                self.env.action_space.high[self.env.unwrapped.gripper_joint_idxes],
            )

    @abstractmethod
    def infer_policy(self):
        pass

    @abstractmethod
    def draw_plot(self):
        pass
