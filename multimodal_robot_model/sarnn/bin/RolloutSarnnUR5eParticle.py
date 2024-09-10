from multimodal_robot_model.sarnn import RolloutSarnn
from multimodal_robot_model.common.tasks import RolloutUR5eParticle

class RolloutSarnnUR5eParticle(RolloutSarnn, RolloutUR5eParticle):
    pass

if __name__ == "__main__":
    rollout = RolloutSarnnUR5eParticle()
    rollout.run()
