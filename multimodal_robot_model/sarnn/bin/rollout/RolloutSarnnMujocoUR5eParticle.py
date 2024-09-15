from multimodal_robot_model.sarnn import RolloutSarnn
from multimodal_robot_model.common.tasks import RolloutMujocoUR5eParticle

class RolloutSarnnMujocoUR5eParticle(RolloutSarnn, RolloutMujocoUR5eParticle):
    pass

if __name__ == "__main__":
    rollout = RolloutSarnnMujocoUR5eParticle()
    rollout.run()
