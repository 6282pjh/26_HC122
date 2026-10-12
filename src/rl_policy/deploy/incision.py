"""Incision: turn a simulator rollout of the PPO policy into a robot trajectory.

The incision policy observes the blade-tip position and its distance to the
line, which only the simulator provides, so it cannot be rolled out on the
Jetson. The rollout is done on a GPU machine:

    python scripts/export_trajectory.py <policy.pt> --heading H --length L --cx X --cy Y --out traj.npz
    python scripts/check_trajectory_safety.py traj.npz

and the CSV it writes (radians, 60 Hz, right arm) is converted here.
"""

import csv
import datetime

from .safety import validate_trajectory
from .servo import sim_to_servo
from .trajectory import I2C_BUS


def incision_trajectory_from_csv(path, rate_hz=3.0, sim_hz=60.0, grip_deg=180.0, end_stop_margin_deg=4.0,
                                 sim_check_passed=None, line_robot_base_mm=None):
    """Right-arm incision trajectory from the CSV of scripts/export_trajectory.py.

    rate_hz: commands per second after down-sampling the 60 Hz rollout.
    grip_deg: gripper servo angle held throughout (the simulated gripper joint
        oscillates and must not be replayed onto a gripper clamping the blade).
    end_stop_margin_deg: joints 1-4 are pulled this far off a servo end stop.
    sim_check_passed: outcome of scripts/check_trajectory_safety.py for this
        rollout (True / False), or None if it was not run. False raises.
    """
    if sim_check_passed is False:
        raise ValueError("this rollout failed scripts/check_trajectory_safety.py and must not be executed")
    rows = []
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if not row or row[0].lstrip().startswith("#"):
                continue
            try:
                vals = [float(x) for x in row[:6]]
            except ValueError:
                continue
            if len(vals) == 6:
                rows.append(vals)
    if not rows:
        raise ValueError("no numeric rows in %s" % path)
    stride = max(1, int(round(sim_hz / rate_hz)))
    joints = []
    for q in rows[::stride]:
        servo = sim_to_servo(q)
        for j in range(4):
            servo[j] = min(max(servo[j], end_stop_margin_deg), 180.0 - end_stop_margin_deg)
        servo[4] = min(max(servo[4], 0.0), 270.0)
        joints.append(servo + [float(grip_deg)])
    traj = {
        "source": "rl_policy",
        "source_detail": "incision PPO policy (Dofbot-Incision-Direct-v3) rolled out in simulation: %s" % path,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "motion": "incision",
        "units": "servo_deg",
        "dt_s": 1.0 / rate_hz,
        "joint_order": ["j1", "j2", "j3", "j4", "j5", "grip"],
        "line_robot_base_mm": line_robot_base_mm,
        "sim_check": {"passed": sim_check_passed, "where": "scripts/check_trajectory_safety.py"},
        "arms": {"right": {"i2c_bus": I2C_BUS["right"], "joints": joints}},
    }
    traj["safety_check"] = validate_trajectory(traj, end_stop_margin_deg=end_stop_margin_deg)
    return traj
