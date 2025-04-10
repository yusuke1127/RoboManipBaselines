import gymnasium as gym


class OperationMujocoHsrTidyup:
    def setup_env(self):
        self.env = gym.make(
            "robo_manip_baselines/MujocoHsrTidyupEnv-v0", render_mode="human"
        )

    def get_pre_motion_phases(self):
        return []
