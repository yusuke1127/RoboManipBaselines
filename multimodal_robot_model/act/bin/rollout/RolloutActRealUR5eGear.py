from multimodal_robot_model.act import RolloutAct
from multimodal_robot_model.common.rollout import RolloutRealUR5eGear

class RolloutActRealUR5eGear(RolloutAct, RolloutRealUR5eGear):
    pass

if __name__ == "__main__":
    robot_ip = "192.168.11.4"
    camera_ids = {"front": "145522067924",
                  "side": None,
                  "hand": "153122070885"}
    rollout = RolloutActRealUR5eGear(robot_ip, camera_ids)
    rollout.run()
