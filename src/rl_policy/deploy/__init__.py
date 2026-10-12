"""Real-robot side of the PPO policies. Imports without Isaac Lab / Isaac Sim.

Deployment is OFFLINE ROLLOUT (see INTERFACE.md): the policies are rolled out
in the simulator on a GPU machine and the robot executes the resulting joint
trajectory. This package holds what the Jetson needs for that:

    plan_opening(...)                  detected incision line -> both-arm trajectory
    incision_trajectory_from_csv(...)  exported incision rollout -> trajectory
    validate_trajectory(...)           safety checks; raises TrajectorySafetyError
    trajectory_to_csv(...)             write the CSV the AGX replay script reads
"""

from .incision import incision_trajectory_from_csv
from .opening import OpeningLibrary, PlacementOutOfRange, plan_opening
from .safety import TrajectorySafetyError, validate_trajectory
from .trajectory import load_trajectory, save_trajectory, trajectory_to_csv

__all__ = [
    "OpeningLibrary",
    "PlacementOutOfRange",
    "TrajectorySafetyError",
    "incision_trajectory_from_csv",
    "load_trajectory",
    "plan_opening",
    "save_trajectory",
    "trajectory_to_csv",
    "validate_trajectory",
]
