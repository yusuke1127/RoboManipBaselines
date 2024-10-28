from multimodal_robot_model.sarnn import RolloutSarnn
from multimodal_robot_model.common.rollout import RolloutRealUR5eGear

class RolloutSarnnRealUR5eGear(RolloutSarnn, RolloutRealUR5eGear):
    pass

if __name__ == "__main__":
    robot_ip = "192.168.11.4"
    camera_ids = {"front": "145522067924",
                  "side": None,
                  "hand": None}
    rollout = RolloutSarnnRealUR5eGear(robot_ip, camera_ids)
    rollout.run()
