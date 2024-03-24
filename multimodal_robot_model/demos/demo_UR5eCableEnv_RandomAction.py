import numpy as np
import gymnasium as gym
import multimodal_robot_model

env = gym.make("multimodal_robot_model/UR5eCableEnv-v0", render_mode="human")
obs, info = env.reset(seed=42)

for _ in range(1000):
   action = env.action_space.sample()
   obs, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      obs, info = env.reset()
      print("reset environment. terminated: {}, truncated: {}".format(terminated, truncated))

env.close()
