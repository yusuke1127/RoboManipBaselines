from multimodal_robot_model.act import RolloutAct
from multimodal_robot_model.common.tasks import RolloutRealUR5eGear

class RolloutActRealUR5eGear(RolloutAct, RolloutRealUR5eGear):
    pass

if __name__ == "__main__":
    robot_ip = "192.168.11.4"
    rollout = RolloutActRealUR5eGear(robot_ip)
    rollout.run()
