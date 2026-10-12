"""Replace an unsafe approach with a straight joint-space descent.

The policy overshoots on its first descent — it dives past cut depth, then
settles. In sim nothing stops it (no ground collision, no blade geometry); on
hardware the wrist would hit the table. The cut itself is fine, and the approach
is not part of the task, so trimming the overshoot and interpolating in from the
home pose gives a hardware-safe trajectory without retraining.

Run check_trajectory_safety.py on the OUTPUT before sending it to the robot.

Usage:
    python scripts/trim_trajectory.py in.npz --start 95 --out out.npz
"""

import argparse

import numpy as np

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("trajectory")
parser.add_argument("--start", type=int, required=True,
                    help="first step of the policy trajectory to keep")
parser.add_argument("--out", required=True)
parser.add_argument("--approach-steps", type=int, default=120,
                    help="steps to interpolate from home into the start pose")
args = parser.parse_args()

d = dict(np.load(args.trajectory, allow_pickle=True))
traj = d["joints_rad"]
if not 0 <= args.start < len(traj):
    raise SystemExit(f"--start must be in [0,{len(traj)})")

kept = traj[args.start:]
home = np.zeros_like(kept[0])
home[5] = kept[0][5]  # hold the gripper where the trajectory has it

# Linear joint-space blend: both ends are clear of the table and every joint
# moves monotonically, so the arm descends without the overshoot it learned.
t = np.linspace(0.0, 1.0, args.approach_steps).reshape(-1, 1)
approach = home[None, :] * (1.0 - t) + kept[0][None, :] * t

out = np.concatenate([approach, kept], axis=0)
d["joints_rad"] = out
if "blade_tip_local" in d:
    # No longer meaningful for the synthesised prefix; the safety check
    # recomputes it from the joints anyway.
    del d["blade_tip_local"]
np.savez(args.out, **d)

csv_path = args.out.rsplit(".", 1)[0] + ".csv"
with open(csv_path, "w") as fh:
    fh.write("# arm_joint1,arm_joint2,arm_joint3,arm_joint4,arm_joint5,grip_joint (radians)\n")
    fh.write(f"# trimmed from {args.trajectory} at step {args.start}, "
             f"{args.approach_steps}-step interpolated approach\n")
    for row in out:
        fh.write(",".join(f"{v:.6f}" for v in row) + "\n")

print(f"원본 {len(traj)}스텝 -> 접근 {args.approach_steps} + 절개 {len(kept)} = {len(out)}스텝")
print(f"saved -> {args.out}")
print(f"saved -> {csv_path}  (AGX로 복사할 파일)")
print("\n반드시 check_trajectory_safety.py 로 검사한 뒤 실물에 보낼 것.")
