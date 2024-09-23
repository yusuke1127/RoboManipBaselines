import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pylab as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import cv2
import pinocchio as pin
from multimodal_robot_model.common import MotionManager, RecordStatus, RecordManager

class RolloutBase(object):
    def __init__(self):
        self.setupArgs()

        self.setupPolicy()

        self.setupEnv()

        self.setupPlot()

        # Setup motion manager
        self.motion_manager = MotionManager(self.env)
        RecordStatus.TELEOP._name_ = "AUTO"

        # Setup record manager
        self.record_manager = RecordManager(self.env)
        self.record_manager.setupSimWorld(self.args.world_idx)

    def run(self):
        self.obs, self.info = self.env.reset(seed=self.args.seed)

        while True:
            if self.record_manager.status == RecordStatus.TELEOP:
                self.inferPolicy()

            self.setCommand()

            action = self.motion_manager.getAction()
            self.obs, _, _, _, self.info = self.env.step(action)

            if self.record_manager.status == RecordStatus.TELEOP:
                self.drawPlot()

            # Manage status
            key = cv2.waitKey(1)
            if self.record_manager.status == RecordStatus.INITIAL:
                initial_duration = 1.0 # [s]
                if (not self.args.wait_before_start and self.record_manager.status_elapsed_duration > initial_duration) or \
                   (self.args.wait_before_start and key == ord("n")):
                    self.record_manager.goToNextStatus()
            elif self.record_manager.status == RecordStatus.PRE_REACH:
                pre_reach_duration = 0.7 # [s]
                if self.record_manager.status_elapsed_duration > pre_reach_duration:
                    self.record_manager.goToNextStatus()
            elif self.record_manager.status == RecordStatus.REACH:
                reach_duration = 0.3 # [s]
                if self.record_manager.status_elapsed_duration > reach_duration:
                    self.record_manager.goToNextStatus()
            elif self.record_manager.status == RecordStatus.GRASP:
                grasp_duration = 0.5 # [s]
                if self.record_manager.status_elapsed_duration > grasp_duration:
                    self.auto_time_idx = 0
                    self.record_manager.goToNextStatus()
                    print("- Press the 'n' key to finish policy rollout.")
            elif self.record_manager.status == RecordStatus.TELEOP:
                self.auto_time_idx += 1
                if key == ord("n"):
                    self.record_manager.goToNextStatus()
                    print("- Press the 'n' key to exit.")
            elif self.record_manager.status == RecordStatus.END:
                if key == ord("n"):
                    break
            if key == 27: # escape key
                break

        # self.env.close()

    def setupArgs(self, parser=None, argv=None):
        if parser is None:
            parser = argparse.ArgumentParser()

        parser.add_argument("--world_idx", type=int, default=0, help="index of the simulation world (0-5)")
        parser.add_argument("--skip", type=int, help="step interval to infer policy", required=False)
        parser.add_argument("--skip_draw", type=int, help="step interval to draw the plot", required=False)
        parser.add_argument("--scale_dt", type=float,
                            help="dt scale of environment (used only in real-world environments)", required=False)
        parser.add_argument("--seed", type=int, default=42, help="random seed", required=False)
        parser.add_argument("--win_xy_policy", type=int, nargs=2,
                            help="xy position of window to plot policy information", required=False)
        parser.add_argument("--wait_before_start", action="store_true", help="whether to wait a key input before starting simulation")

        if argv is None:
            argv = sys.argv
        self.args = parser.parse_args(argv[1:])

    def setupPolicy(self):
        raise NotImplementedError()

    def setupEnv(self):
        raise NotImplementedError()

    def setupPlot(self, fig_ax=None):
        matplotlib.use("agg")
        if fig_ax is None:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(10.0, 5.0), dpi=60, squeeze=False)
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

    def inferPolicy(self):
        raise NotImplementedError()

    def setCommand(self):
        raise NotImplementedError()

    def drawPlot(self):
        raise NotImplementedError()
