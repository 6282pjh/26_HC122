"""Roll the opening policy out in sim and save BOTH arms' trajectory for replay.

Writes servo DEGREES directly (the conversion uses the same mapping the
environment used to read the taught JSON, so it round-trips exactly), then
checks the result against the taught motion before anything goes near hardware:

  * did the rollout complete the opening in sim
  * how far the servo angles stray from the taught joint path
  * whether any command sits on a servo end stop
  * the largest change between consecutive commands
  * the smallest gap between the two arms

With --line the opening is moved to where the camera saw the incision. The
motion was taught for ONE incision (--ref, by default the line the taught
incision motion cuts); a detected line is expressed as how far it is shifted and
turned from that one, and the policy is rolled out for that placement. Only
placements inside the range the policy was trained on are accepted.

Usage:
    python scripts/export_opening_trajectory.py <checkpoint.pt> \
        --out opening_traj.csv [--stride 6] [--device cuda:0]
    python scripts/export_opening_trajectory.py <checkpoint.pt> \
        --line 0.166 0.012 104.0 --out opening_cam.csv
"""

import argparse
import json
import math
import os
import sys

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("checkpoint", type=str)
parser.add_argument("--out", type=str, default="opening_traj.csv")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--stride", type=int, default=6,
                    help="keep every Nth 60 Hz sim step as one servo command")
parser.add_argument("--max-step-deg", type=float, default=25.0)
parser.add_argument("--servo-margin", type=float, default=1.0,
                    help="keep joints 1-4 this many degrees off the servo end stops. The "
                         "policy parks arm_joint3 exactly on its stop (servo 0) where the "
                         "taught motion holds it at 1-12; 1 deg is the closest the taught "
                         "motion itself ever goes.")
parser.add_argument("--max-deviation-deg", type=float, default=15.0,
                    help="refuse if any joint strays further than this from the taught path")
parser.add_argument("--min-gap", type=float, default=0.060,
                    help="refuse if the arm skeletons come closer than this (m)")
parser.add_argument("--line", type=float, nargs=3, default=None, metavar=("CX", "CY", "HEADING"),
                    help="detected incision: centre (m, right-arm frame) and direction (deg)")
# Default reference = the line the taught INCISION motion cuts, from FK of
# incision_dual.json: (154, +44) -> (168, -29) mm. The opening was taught for
# that cut. Replace it with a camera reading of the same cut when there is one
# (line_to_opening.py --set-reference) -- that cancels the FK error entirely.
parser.add_argument("--ref", type=float, nargs=3, default=(0.161, 0.0075, 101.0),
                    metavar=("CX", "CY", "HEADING"), help="the incision the motion was taught for")
parser.add_argument("--line-ends", type=float, nargs=4, default=None, metavar=("X1", "Y1", "X2", "Y2"),
                    help="detected incision as its two end points (mm, right-arm base frame), "
                         "for a detector that already reports robot coordinates")
parser.add_argument("--ref-ends", type=float, nargs=4, default=None, metavar=("X1", "Y1", "X2", "Y2"),
                    help="the taught incision as two end points (mm), from the same detector")
args = parser.parse_args()


def _from_ends(e):
    x1, y1, x2, y2 = (v / 1000.0 for v in e)
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0, math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0]


if args.line_ends is not None:
    args.line = _from_ends(args.line_ends)
if args.ref_ends is not None:
    args.ref = _from_ends(args.ref_ends)

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(argparse.Namespace(headless=True, device=args.device, enable_cameras=False,
                                     livestream=-1, kit_args="", experience="")).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import dofbot.tasks  # noqa: F401,E402
from skrl.utils.runner.torch import Runner  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402

cfg = parse_env_cfg("Dofbot-Opening-Direct-v0", device=args.device, num_envs=1)
TRAINED = {"x shift": tuple(cfg.line_shift_x_range), "y shift": tuple(cfg.line_shift_y_range),
           "turn": tuple(cfg.line_yaw_range_deg)}
# The incision exactly where it was taught, starting exactly at READY.
cfg.line_shift_x_range = cfg.line_shift_y_range = (0.0, 0.0)
cfg.line_yaw_range_deg = (0.0, 0.0)
cfg.reset_joint_noise = 0.0
cfg.rsi_prob = 0.0
env = gym.make("Dofbot-Opening-Direct-v0", cfg=cfg)
u = env.unwrapped
env_w = SkrlVecEnvWrapper(env, ml_framework="torch")

here = os.path.dirname(os.path.abspath(__file__))
acfg = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "task", "dofbot", "dofbot", "tasks", "direct", "dofbot", "agents", "skrl_ppo_opening_cfg.yaml")))
acfg["trainer"]["close_environment_at_exit"] = False
acfg["agent"]["experiment"]["write_interval"] = 0
acfg["agent"]["experiment"]["checkpoint_interval"] = 0
runner = Runner(env_w, acfg)
runner.agent.load(args.checkpoint)
if hasattr(runner.agent, "set_mode"):
    runner.agent.set_mode("eval")

motion = json.load(open(cfg.motion_json))
taught = {k: [wp[k]["joints"] for wp in motion["waypoints"]] for k in ("right", "left")}
ARMS = ("right", "left")


def to_servo(q4, arm):
    """Inverse of DofbotOpeningEnv._load_motion for arm joints 1-4."""
    inv = cfg.servo_inverted[arm]
    return [(-math.degrees(q) if inv[i] else math.degrees(q)) + 90.0 for i, q in enumerate(q4)]


# --- skeleton gap between the arms (same measure as clearance_check.py) -------
names = u.robot_incise.body_names
chain = [names.index(f"arm_link{i}") for i in range(1, 6)]
pairs = [(chain[i], chain[i + 1]) for i in range(4)]
pairs += [(chain[4], names.index("rlink2")), (chain[4], names.index("llink2"))]
a_idx = torch.tensor([p[0] for p in pairs], device=u.device)
b_idx = torch.tensor([p[1] for p in pairs], device=u.device)


def seg_dist(p1, q1, p2, q2):
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a = (d1 * d1).sum(-1).clamp(min=1e-9)
    e = (d2 * d2).sum(-1).clamp(min=1e-9)
    f, c, b = (d2 * r).sum(-1), (d1 * r).sum(-1), (d1 * d2).sum(-1)
    s = ((b * f - c * e) / (a * e - b * b).clamp(min=1e-9)).clamp(0.0, 1.0)
    t = ((b * s + f) / e).clamp(0.0, 1.0)
    s = ((b * t - c) / a).clamp(0.0, 1.0)
    return torch.linalg.norm((p1 + s.unsqueeze(-1) * d1) - (p2 + t.unsqueeze(-1) * d2), dim=-1)


def arm_gap():
    pr, pl = u.robot_incise.data.body_pos_w[0], u.robot_retract.data.body_pos_w[0]
    d = seg_dist(pr[a_idx][:, None], pr[b_idx][:, None], pl[a_idx][None], pl[b_idx][None])
    return float(d.min())


placement = "taught placement"
if args.line is not None:
    # The env places the path as  x' = c0 + R(yaw)(x - c0) + shift,  c0 being the
    # midpoint of the two tips at the first anchored waypoint. The rigid motion
    # that carries the reference line onto the detected one is
    # x' = p_line + R(yaw)(x - p_ref), hence the shift below.
    yaw = (args.line[2] - args.ref[2] + 90.0) % 180.0 - 90.0      # a line has no sense
    c, s_ = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    c0 = u._ref_pts[:, u._anchor_from, :2].mean(dim=0).tolist()
    rx, ry = args.ref[0] - c0[0], args.ref[1] - c0[1]
    shift = (args.line[0] - c0[0] - (c * rx - s_ * ry), args.line[1] - c0[1] - (s_ * rx + c * ry))
    placement = (f"shift ({shift[0]*1000:+.1f}, {shift[1]*1000:+.1f}) mm, turn {yaw:+.1f} deg "
                 f"from the taught incision")
    bad = []
    for name, value, rng, unit, k in (("x shift", shift[0], u.cfg.line_shift_x_range, "mm", 1000.0),
                                      ("y shift", shift[1], u.cfg.line_shift_y_range, "mm", 1000.0),
                                      ("turn", yaw, u.cfg.line_yaw_range_deg, "deg", 1.0)):
        lo, hi = TRAINED[name]
        if not (lo - 1e-9 <= value <= hi + 1e-9):
            bad.append(f"{name} {value*k:+.1f} {unit} is outside the trained {lo*k:+.0f}..{hi*k:+.0f} {unit}")
    if bad:
        print(f"\nREFUSED: {placement}")
        for b in bad:
            print("  - " + b)
        print("  The policy was never trained there. Move the patch, or retrain with wider ranges.")
        sys.stdout.flush()
        os._exit(1)
    u.cfg.line_shift_x_range = (shift[0], shift[0])
    u.cfg.line_shift_y_range = (shift[1], shift[1])
    u.cfg.line_yaw_range_deg = (yaw, yaw)

obs, _ = env_w.reset()
rows, gaps, done_step = [], [], None
steps = int(u.max_episode_length) - 2
for step in range(steps):
    with torch.inference_mode():
        try:
            out = runner.agent.act(obs, None, timestep=0, timesteps=0)
        except TypeError:
            out = runner.agent.act(obs, timestep=0, timesteps=0)
        obs, _, term, _, _ = env_w.step(out[-1].get("mean_actions", out[0]))
    if bool(term.any()):
        print(f"ABORT: the rollout crashed or strayed at step {step}.")
        sys.stdout.flush()
        os._exit(1)
    row = []
    for a, robot in enumerate((u.robot_incise, u.robot_retract)):
        q = robot.data.joint_pos[0, u._arm_dof_idx].tolist()
        # Wrist roll and gripper are not policy outputs; send the taught values.
        row += to_servo(q, ARMS[a]) + [taught[ARMS[a]][-1][4], taught[ARMS[a]][-1][5]]
    rows.append(row)
    gaps.append(arm_gap())
    if done_step is None and bool(u._completed[0]):
        done_step = step
        break

problems = []
if done_step is None:
    problems.append("the rollout did NOT complete the opening in sim")
    done_step = len(rows) - 1

# Commands: every Nth step, always ending on the settled final pose.
keep = list(range(0, done_step + 1, args.stride))
if keep[-1] != done_step:
    keep.append(done_step)
cmds = [rows[i] for i in keep]
# Start from the taught READY pose itself rather than the first sim sample.
cmds.insert(0, taught["right"][0][:] + taught["left"][0][:])

clamped = 0
for c in cmds:
    for j in (0, 1, 2, 3, 6, 7, 8, 9):
        v = min(max(c[j], args.servo_margin), 180.0 - args.servo_margin)
        if abs(v - c[j]) > 1e-6:
            clamped += 1
        c[j] = v

# --- checks ------------------------------------------------------------------
# Distance of each command from the taught joint path (joint-interpolated
# between waypoints, as the servos move), per arm, joints 1-4, in servo degrees.
dense = {}
for arm in ARMS:
    pts = []
    w = taught[arm]
    for k in range(len(w) - 1):
        for i in range(50):
            f = i / 50.0
            pts.append([w[k][j] * (1 - f) + w[k + 1][j] * f for j in range(4)])
    pts.append(w[-1][:4])
    dense[arm] = torch.tensor(pts)
worst_dev, worst_at = 0.0, None
for n, c in enumerate(cmds):
    for a, arm in enumerate(ARMS):
        q = torch.tensor(c[a * 6: a * 6 + 4])
        diff = (dense[arm] - q).abs().amax(dim=-1)            # worst joint, per path point
        dev = float(diff.min())
        if dev > worst_dev:
            worst_dev, worst_at = dev, (n, arm)
biggest = max(max(abs(cmds[i][j] - cmds[i - 1][j]) for j in range(12)) for i in range(1, len(cmds)))
limits = [(0.0, 180.0)] * 4 + [(0.0, 270.0), (0.0, 180.0)]
out_of_range = sum(1 for c in cmds for j in range(12)
                   if not (limits[j % 6][0] <= c[j] <= limits[j % 6][1]))
at_margin = sum(1 for c in cmds for j in (0, 1, 2, 3, 6, 7, 8, 9)
                if c[j] <= args.servo_margin + 0.05 or c[j] >= 180.0 - args.servo_margin - 0.05)
min_gap = min(gaps)
end_err = {arm: max(abs(cmds[-1][a * 6 + j] - taught[arm][-1][j]) for j in range(4))
           for a, arm in enumerate(ARMS)}

# The taught joint path is for the taught placement; a moved incision is
# supposed to differ from it, so the limit only applies when nothing was moved.
if args.line is None and worst_dev > args.max_deviation_deg:
    problems.append(f"strays {worst_dev:.1f} deg from the taught path (limit {args.max_deviation_deg:.0f})")
if biggest > args.max_step_deg:
    problems.append(f"largest command step {biggest:.1f} deg (limit {args.max_step_deg:.0f}); lower --stride")
if out_of_range:
    problems.append(f"{out_of_range} commanded angles outside the servo range")
if min_gap < args.min_gap:
    problems.append(f"arms come within {min_gap*1000:.0f} mm (limit {args.min_gap*1000:.0f})")

print(f"\nplacement: {placement}")
print(f"rollout: completed at step {done_step} ({done_step / 60.0:.1f} s of sim time)")
print(f"commands: {len(cmds)} (READY + every {args.stride}th step)")
print(f"largest change between commands: {biggest:.1f} deg")
print(f"furthest from the taught joint path: {worst_dev:.1f} deg"
      + (f" (command {worst_at[0]}, {worst_at[1]} arm)" if worst_at else ""))
print(f"final pose vs taught INSERT_02: right {end_err['right']:.1f} deg, left {end_err['left']:.1f} deg")
print(f"servo-margin {args.servo_margin:.0f} deg: {clamped} of {len(cmds) * 8} angles pulled off a stop; "
      f"{at_margin} now sit at the margin")
print(f"smallest gap between the arms (skeleton): {min_gap * 1000:.0f} mm")
print("\n  n   right j1..j4              left j1..j4")
for n, c in enumerate(cmds):
    print(f"{n:>3}  " + " ".join(f"{v:6.1f}" for v in c[0:4]) + "    " + " ".join(f"{v:6.1f}" for v in c[6:10]))

if problems:
    print("\nNOT WRITTEN. This trajectory must not be run on hardware:")
    for s in problems:
        print("  - " + s)
    sys.stdout.flush()
    os._exit(1)

with open(args.out, "w") as fh:
    fh.write("# opening, both arms, SERVO DEGREES. right = I2C bus 8, left = I2C bus 1\n")
    fh.write("# r1,r2,r3,r4,r5,r_grip,l1,l2,l3,l4,l5,l_grip\n")
    fh.write(f"# {placement}\n")
    fh.write(f"# checkpoint {os.path.basename(args.checkpoint)}; {len(cmds)} commands; "
             f"max step {biggest:.1f} deg; max deviation from taught {worst_dev:.1f} deg; "
             f"min arm gap {min_gap * 1000:.0f} mm\n")
    for c in cmds:
        fh.write(",".join(f"{v:.1f}" for v in c) + "\n")
print(f"\nsaved -> {args.out}")
sys.stdout.flush()
os._exit(0)
