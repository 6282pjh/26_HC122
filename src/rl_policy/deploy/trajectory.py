"""Trajectory files: the JSON format of INTERFACE.md section 7, and the CSV
that agx/replay_opening_on_agx.py reads."""

import json

I2C_BUS = {"right": 8, "left": 1}


def save_trajectory(traj, path):
    with open(path, "w") as fh:
        json.dump(traj, fh, indent=1)


def load_trajectory(path):
    with open(path) as fh:
        return json.load(fh)


def trajectory_to_csv(traj, path):
    """Write a both-arm trajectory as the 12-column CSV the AGX replay reads."""
    right, left = traj["arms"]["right"]["joints"], traj["arms"]["left"]["joints"]
    with open(path, "w") as fh:
        fh.write("# %s, both arms, SERVO DEGREES. right = I2C bus 8, left = I2C bus 1\n" % traj.get("motion", ""))
        fh.write("# source: %s (%s)\n" % (traj.get("source"), traj.get("source_detail")))
        fh.write("# r1,r2,r3,r4,r5,r_grip,l1,l2,l3,l4,l5,l_grip\n")
        for r, l in zip(right, left):
            fh.write(",".join("%.1f" % v for v in list(r) + list(l)) + "\n")
