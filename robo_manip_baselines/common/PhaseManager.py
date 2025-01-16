import cv2
import numpy as np

from .Phase import Phase


class PhaseManager(object):
    """Phase Manager."""

    def __init__(self, env, phase_order, phase=None):
        self.env = env
        self.phase_order = phase_order

        if phase is None:
            self.initial_phase = self.phase_order[0]
        else:
            self.initial_phase = phase

        self.reset()

    def reset(self):
        """Reset."""
        self.set_phase(self.initial_phase)

    def set_phase(self, phase):
        """Set phase."""
        self.phase = phase

        if self.env is None:
            self.phase_start_time = 0.0
        else:
            self.phase_start_time = self.env.unwrapped.get_time()

    def set_next_phase(self):
        """Set the next phase."""
        idx = self.phase_order.index(self.phase)

        if idx == len(self.phase_order) - 1:
            raise ValueError(
                "[PhaseManager] Cannot go from the last phase to the next."
            )

        self.set_phase(self.phase_order[idx + 1])

    def get_phase_image(self, size=(50, 320)):
        """Get the image corresponding to the phase."""
        phase_image = np.zeros(size + (3,), dtype=np.uint8)

        if self.phase == Phase.INITIAL:
            phase_image[:, :] = np.array([200, 255, 200])
        elif self.phase in (
            Phase.PRE_REACH,
            Phase.REACH,
            Phase.GRASP,
        ):
            phase_image[:, :] = np.array([255, 255, 200])
        elif self.phase in (Phase.TELEOP, Phase.ROLLOUT):
            phase_image[:, :] = np.array([255, 200, 200])
        elif self.phase == Phase.END:
            phase_image[:, :] = np.array([200, 200, 255])
        else:
            raise ValueError(f"[PhaseManager] Unknown phase: {self.phase}")

        cv2.putText(
            phase_image,
            self.phase.upper(),
            (5, 35),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (0, 0, 0),
            2,
        )

        return phase_image

    def get_phase_elapsed_duration(self):
        """Get the elapsed duration of the current phase."""
        return self.env.unwrapped.get_time() - self.phase_start_time
