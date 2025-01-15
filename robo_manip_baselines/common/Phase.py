class Phase(object):
    """Phase of robot operation."""

    # Initial phase
    INITIAL = "initial"

    # Pre-reach phase
    PRE_REACH = "pre_reach"
    # Reach phase
    REACH = "reach"
    # Grasp phase
    GRASP = "grasp"

    # Teleoperation phase
    TELEOP = "teleop"
    # Policy rollout phase
    ROLLOUT = "rollout"

    # End phase
    END = "end"


class PhaseOrder(object):
    """Phase order of robot operation."""

    # Phase order for teleoperation
    TELEOP = [
        Phase.INITIAL,
        Phase.PRE_REACH,
        Phase.REACH,
        Phase.GRASP,
        Phase.TELEOP,
        Phase.END,
    ]

    # Phase order for policy rollout
    ROLLOUT = [
        Phase.INITIAL,
        Phase.PRE_REACH,
        Phase.REACH,
        Phase.GRASP,
        Phase.ROLLOUT,
        Phase.END,
    ]
