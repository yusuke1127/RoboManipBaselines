import gymnasium as gym


class OperationMujocoG1Bottles:
    def setup_env(self):
        self.env = gym.make(
            "robo_manip_baselines/MujocoG1BottlesEnv-v0", render_mode="human"
        )

    def get_pre_motion_phases(self):
        return []
