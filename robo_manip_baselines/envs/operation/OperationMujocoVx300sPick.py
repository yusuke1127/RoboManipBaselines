import gymnasium as gym


class OperationMujocoVx300sPick:
    def setup_env(self):
        self.env = gym.make(
            "robo_manip_baselines/MujocoVx300sPickEnv-v0", render_mode="human"
        )
