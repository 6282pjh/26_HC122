"""Pre-compute the opening motion for every nearby incision placement.

The V0 policy completes the opening for incisions within about +-10 mm /
+-20 mm / +-8 deg of the taught one, but it needs the simulator to run. This
rolls it out once for every placement on a fine grid over that range, checks
each result (completed, arm clearance, servo range, step size) and stores the
servo commands in one .npz. The Jetson then just looks up the grid cell nearest
to the detected incision -- no simulator, no PyTorch, and every trajectory it
can ever play has already passed the checks here.

Usage:
    python scripts/export_opening_library.py <v0_checkpoint.pt> \
        --out weights/opening_library.npz [--device cuda:0]
"""

import argparse
import json
import math
import os
import sys

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("checkpoint", type=str)
parser.add_argument("--out", type=str, default="opening_library.npz")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--step-mm", type=float, default=2.5)
parser.add_argument("--step-deg", type=float, default=2.0)
parser.add_argument("--stride", type=int, default=6)
parser.add_argument("--servo-margin", type=float, default=1.0)
parser.add_argument("--max-step-deg", type=float, default=25.0)
parser.add_argument("--min-gap", type=float, default=0.060)
# The incision the opening was taught for: FK of incision_dual.json, (154,+44)->(168,-29) mm.
parser.add_argument("--ref", type=float, nargs=3, default=(0.161, 0.0075, 101.0), metavar=("CX", "CY", "HEADING"))
args = parser.parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(argparse.Namespace(headless=True, device=args.device, enable_cameras=False,
                                     livestream=-1, kit_args="", experience="")).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import dofbot.tasks  # noqa: F401,E402
from skrl.utils.runner.torch import Runner  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402

base = parse_env_cfg("Dofbot-Opening-Direct-v0", device=args.device, num_envs=1)


def axis(rng, step, scale):
    lo, hi = rng[0] * scale, rng[1] * scale
    n = int(round((hi - lo) / step))
    return np.linspace(lo, hi, n + 1) / scale


# Exactly the range the policy was trained on, nothing beyond it.
dxs = axis(base.line_shift_x_range, args.step_mm, 1000.0)
dys = axis(base.line_shift_y_range, args.step_mm, 1000.0)
yaws = axis(base.line_yaw_range_deg, args.step_deg, 1.0)
G = np.stack(np.meshgrid(yaws, dxs, dys, indexing="ij"), -1).reshape(-1, 3)     # turn, dx, dy
N = len(G)
print(f"\ngrid: {len(yaws)} turns x {len(dxs)} x-shifts x {len(dys)} y-shifts = {N} placements")

cfg = parse_env_cfg("Dofbot-Opening-Direct-v0", device=args.device, num_envs=N)
cfg.reset_joint_noise = 0.0
cfg.rsi_prob = 0.0
cfg.scene.env_spacing = 1.0
env = gym.make("Dofbot-Opening-Direct-v0", cfg=cfg)
u = env.unwrapped
grid_t = torch.as_tensor(G, dtype=torch.float32, device=u.device)
first = {"used": False}


def fixed_placement(m):
    # The first full reset places every env on its own grid cell. Any later
    # (partial) reset belongs to an env that has already been scored.
    if m == N and not first["used"]:
        first["used"] = True
        return grid_t[:, 1:3].clone(), torch.deg2rad(grid_t[:, 0])
    return torch.zeros((m, 2), device=u.device), torch.zeros(m, device=u.device)


u._sample_placement = fixed_placement
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


obs, _ = env_w.reset()
assert first["used"], "the grid placements were not applied"
steps = 300
q_log = torch.zeros((steps, N, 2, 4), device=u.device)
gap = torch.full((N,), 9.0, device=u.device)
done_step = torch.full((N,), -1, dtype=torch.long, device=u.device)
dead = torch.zeros(N, dtype=torch.bool, device=u.device)
for step in range(steps):
    with torch.inference_mode():
        try:
            out = runner.agent.act(obs, None, timestep=0, timesteps=0)
        except TypeError:
            out = runner.agent.act(obs, timestep=0, timesteps=0)
        obs, _, term, trunc, _ = env_w.step(out[-1].get("mean_actions", out[0]))
    live = ~dead & (done_step < 0)
    for a, robot in enumerate(u.robots):
        q_log[step, :, a] = robot.data.joint_pos[:, u._arm_dof_idx]
    pr, pl = u.robot_incise.data.body_pos_w, u.robot_retract.data.body_pos_w
    d = seg_dist(pr[:, a_idx][:, :, None], pr[:, b_idx][:, :, None],
                 pl[:, a_idx][:, None], pl[:, b_idx][:, None]).flatten(1).min(dim=-1).values
    gap = torch.where(live, torch.minimum(gap, d), gap)
    newly = u._completed & live
    done_step = torch.where(newly, torch.full_like(done_step, step), done_step)
    dead |= (term.view(-1).bool() | trunc.view(-1).bool()) & (done_step < 0)

motion = json.load(open(cfg.motion_json))
ready = motion["waypoints"][0]
last = motion["waypoints"][-1]
ARMS = ("right", "left")
inv = np.array([cfg.servo_inverted[k][:4] for k in ARMS])
q_np = np.degrees(q_log.cpu().numpy())                                   # (T, N, 2, 4)
servo = np.where(inv[None, None], -q_np, q_np) + 90.0
done_np, gap_np = done_step.cpu().numpy(), gap.cpu().numpy()

all_cmds, valid, reasons = [], np.zeros(N, dtype=bool), {"not completed": 0, "arm gap": 0, "step size": 0}
for e in range(N):
    if done_np[e] < 0:
        reasons["not completed"] += 1
        all_cmds.append(None)
        continue
    keep = list(range(0, done_np[e] + 1, args.stride))
    if keep[-1] != done_np[e]:
        keep.append(int(done_np[e]))
    rows = [ready["right"]["joints"][:] + ready["left"]["joints"][:]]
    for i in keep:
        row = []
        for a, arm in enumerate(ARMS):
            s4 = np.clip(servo[i, e, a], args.servo_margin, 180.0 - args.servo_margin)
            row += [float(v) for v in s4] + [last[arm]["joints"][4], last[arm]["joints"][5]]
        rows.append(row)
    rows = np.array(rows)
    biggest = np.abs(np.diff(rows, axis=0)).max()
    if gap_np[e] < args.min_gap:
        reasons["arm gap"] += 1
    elif biggest > args.max_step_deg:
        reasons["step size"] += 1
    else:
        valid[e] = True
    all_cmds.append(rows)

L = max(len(c) for c in all_cmds if c is not None)
cmds = np.zeros((N, L, 12), dtype=np.float32)
length = np.zeros(N, dtype=np.int32)
for e, c in enumerate(all_cmds):
    if c is not None and valid[e]:
        cmds[e, :len(c)] = c
        cmds[e, len(c):] = c[-1]
        length[e] = len(c)

shape = (len(yaws), len(dxs), len(dys))
print(f"usable placements: {int(valid.sum())}/{N} = {valid.mean() * 100:.1f}%")
for k, v in reasons.items():
    if v:
        print(f"  rejected, {k}: {v}")
print(f"arm skeleton gap over the usable ones: mean {gap_np[valid].mean() * 1000:.0f} mm, "
      f"smallest {gap_np[valid].min() * 1000:.0f} mm")
print(f"commands per trajectory: {length[valid].min()}..{length[valid].max()}")
vg = valid.reshape(shape)
print("usable share by turn: " + "  ".join(f"{y:+.0f}:{vg[i].mean() * 100:.0f}%" for i, y in enumerate(yaws)))
if not valid.any():
    print("NOT WRITTEN: nothing usable.")
    sys.stdout.flush()
    os._exit(1)
np.savez_compressed(
    args.out, cmds=cmds.reshape(shape + (L, 12)), length=length.reshape(shape), valid=vg,
    yaws=yaws, dxs=dxs, dys=dys,
    centre=u._ref_pts[:, u._anchor_from, :2].mean(dim=0).cpu().numpy(),
    ref_line=np.array(args.ref, dtype=np.float64),
)
print(f"saved -> {args.out}  ({os.path.getsize(args.out) / 1024:.0f} KB)")
sys.stdout.flush()
os._exit(0)
