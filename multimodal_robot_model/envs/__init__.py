from . import mujoco

from gymnasium.envs.registration import register

register(
    id="multimodal_robot_model/UR5eCableEnv-v0",
    entry_point="multimodal_robot_model.envs.mujoco:UR5eCableEnv",
)
