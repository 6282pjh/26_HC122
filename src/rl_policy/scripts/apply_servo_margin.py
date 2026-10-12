"""Pull an exported incision rollout off the servo end stops before it is checked.

The simulator holds a joint at its limit for free, so a rollout can sit exactly
on a limit for most of the cut; a real servo commanded there leans on its
mechanical stop. This clamps joints 1-4 a few degrees inside their range and
pins the gripper at one value, and writes a new .npz / .csv. The result changes
the arm's geometry slightly, so it MUST be re-checked:

    python scripts/apply_servo_margin.py traj.npz --margin-deg 4 --out traj_margin.npz
    python scripts/check_trajectory_safety.py traj_margin.npz

Needs numpy only.
"""

import argparse
import math

import numpy as np

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("trajectory")
parser.add_argument("--out", required=True)
parser.add_argument("--margin-deg", type=float, default=4.0)
parser.add_argument("--grip-rad", type=float, default=-0.9, help="gripper joint held at this value (closed on the tool)")
args = parser.parse_args()

data = dict(np.load(args.trajectory, allow_pickle=True))
traj = np.array(data["joints_rad"], dtype=np.float64)
names = [str(n) for n in data["joint_names"]]
limit = math.pi / 2 - math.radians(args.margin_deg)
changed = 0
for col, name in enumerate(names):
    if name in ("arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4"):
        clipped = np.clip(traj[:, col], -limit, limit)
        changed += int((np.abs(clipped - traj[:, col]) > 1e-9).sum())
        traj[:, col] = clipped
    elif name == "grip_joint":
        traj[:, col] = args.grip_rad
data["joints_rad"] = traj
np.savez(args.out, **data)
csv_path = args.out.rsplit(".", 1)[0] + ".csv"
with open(csv_path, "w") as fh:
    fh.write("# arm_joint1,arm_joint2,arm_joint3,arm_joint4,arm_joint5,grip_joint (radians)\n")
    fh.write("# servo margin %.1f deg applied to joints 1-4, gripper pinned at %.2f rad\n" % (args.margin_deg, args.grip_rad))
    for row in traj:
        fh.write(",".join("%.6f" % v for v in row) + "\n")
print("margin %.1f deg: %d of %d joint angles pulled off a stop" % (args.margin_deg, changed, traj.shape[0] * 4))
print("saved -> %s and %s" % (args.out, csv_path))
