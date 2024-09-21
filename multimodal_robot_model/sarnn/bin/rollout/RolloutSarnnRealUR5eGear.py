from multimodal_robot_model.sarnn import RolloutSarnn
from multimodal_robot_model.common.tasks import RolloutRealUR5eGear

class RolloutSarnnRealUR5eGear(RolloutSarnn, RolloutRealUR5eGear):
    pass

if __name__ == "__main__":
    robot_ip = "192.168.11.4"
    rollout = RolloutSarnnRealUR5eGear(robot_ip)
    rollout.run()
