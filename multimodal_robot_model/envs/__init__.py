from . import mujoco

from gymnasium.envs.registration import register

register(
    id="multimodal_robot_model/UR5eCableEnv-v0",
    entry_point="multimodal_robot_model.envs.mujoco:UR5eCableEnv",
)
register(
    id="multimodal_robot_model/UR5eRingEnv-v0",
    entry_point="multimodal_robot_model.envs.mujoco:UR5eRingEnv",
)
register(
    id="multimodal_robot_model/UR5eScoopEnv-v0",
    entry_point="multimodal_robot_model.envs.mujoco:UR5eScoopEnv",
)
register(
    id="multimodal_robot_model/UR5eClothEnv-v0",
    entry_point="multimodal_robot_model.envs.mujoco:UR5eClothEnv",
)
