"""How close do the two arms come to each other during the opening motion?

Runs the taught JSON motion (joint-interpolated, no policy) or a trained policy
from READY and reports the smallest gap between the two arms' skeletons, when
it happens, and how far each tip swings sideways -- the taught motion swings
the arms out to opposite sides on purpose so they do not meet over the patch.

The gap is between link SKELETONS (segments joining consecutive link origins,
plus both finger links). The links are roughly 25-30 mm thick each side of that
line, so a skeleton gap under ~60 mm means the bodies are touching or close to it.

Usage:
    python scripts/clearance_check.py                       # taught motion
    python scripts/clearance_check.py --checkpoint <agent.pt>
"""

import argparse
import os
import sys

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--no-randomize", action="store_true")
args = parser.parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(argparse.Namespace(headless=True, device=args.device, enable_cameras=False,
                                     livestream=-1, kit_args="", experience="")).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import dofbot.tasks  # noqa: F401,E402

N = args.envs
cfg = parse_env_cfg("Dofbot-Opening-Direct-v0", device=args.device, num_envs=N)
cfg.seed = 1234
cfg.rsi_prob = 0.0
if args.no_randomize or not args.checkpoint:
    cfg.line_shift_x_range = cfg.line_shift_y_range = (0.0, 0.0)
    cfg.line_yaw_range_deg = (0.0, 0.0)
if not args.checkpoint:
    cfg.reset_joint_noise = 0.0
env = gym.make("Dofbot-Opening-Direct-v0", cfg=cfg)
u = env.unwrapped

if args.checkpoint:
    from skrl.utils.runner.torch import Runner
    from isaaclab_rl.skrl import SkrlVecEnvWrapper
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
    obs, _ = env_w.reset()
else:
    env.reset()

names = u.robot_incise.body_names
chain = [names.index(f"arm_link{i}") for i in range(1, 6)]
pairs = [(chain[i], chain[i + 1]) for i in range(4)]
pairs += [(chain[4], names.index("rlink2")), (chain[4], names.index("llink2"))]
a_idx = torch.tensor([p[0] for p in pairs], device=u.device)
b_idx = torch.tensor([p[1] for p in pairs], device=u.device)
seg_names = ["link1-2", "link2-3", "link3-4", "link4-5", "wrist-fingerR", "wrist-fingerL"]


def segments(robot):
    p = robot.data.body_pos_w
    return p[:, a_idx], p[:, b_idx]                      # (N, S, 3)


def seg_dist(p1, q1, p2, q2):
    """Closest distance between segments p1q1 and p2q2, batched."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a = (d1 * d1).sum(-1).clamp(min=1e-9)
    e = (d2 * d2).sum(-1).clamp(min=1e-9)
    f = (d2 * r).sum(-1)
    c = (d1 * r).sum(-1)
    b = (d1 * d2).sum(-1)
    den = (a * e - b * b).clamp(min=1e-9)
    s = ((b * f - c * e) / den).clamp(0.0, 1.0)
    t = ((b * s + f) / e).clamp(0.0, 1.0)
    s = ((b * t - c) / a).clamp(0.0, 1.0)
    return torch.linalg.norm((p1 + s.unsqueeze(-1) * d1) - (p2 + t.unsqueeze(-1) * d2), dim=-1)


S = len(pairs)
best = torch.full((N,), 9.0, device=u.device)
best_step = torch.zeros(N, dtype=torch.long, device=u.device)
best_pair = torch.zeros(N, dtype=torch.long, device=u.device)
y_max = torch.zeros((N, 2), device=u.device)
trace = []
way = u._way_q[:, :, :4]
legs = way.shape[1] - 1
per_leg = 60

for step in range(args.steps):
    if args.checkpoint:
        with torch.inference_mode():
            try:
                out = runner.agent.act(obs, None, timestep=0, timesteps=0)
            except TypeError:
                out = runner.agent.act(obs, timestep=0, timesteps=0)
            obs, _, _, _, _ = env_w.step(out[-1].get("mean_actions", out[0]))
    else:
        leg = min(step // per_leg, legs)
        if leg < legs:
            f = (step % per_leg + 1) / per_leg
            want = way[:, leg] * (1 - f) + way[:, leg + 1] * f
        else:
            want = way[:, -1]
        cur = u._joint_targets[:, :, u._arm_dof_idx]
        act = ((want.unsqueeze(0) - cur) / cfg.joint_delta_scale).clamp(-1.0, 1.0)
        env.step(act.reshape(N, 8))
    p1, q1 = segments(u.robot_incise)
    p2, q2 = segments(u.robot_retract)
    d = seg_dist(p1[:, :, None], q1[:, :, None], p2[:, None], q2[:, None])   # (N, S, S)
    flat = d.reshape(N, -1)
    dmin, arg = flat.min(dim=-1)
    better = dmin < best
    best = torch.where(better, dmin, best)
    best_step = torch.where(better, torch.full_like(best_step, step), best_step)
    best_pair = torch.where(better, arg, best_pair)
    tips, _ = u._tips()
    y_max = torch.maximum(y_max, tips[:, :, 1].abs())
    trace.append((step, float(dmin.mean()), float(dmin.min()), float(u._progress().mean())))

label = f"policy {os.path.basename(args.checkpoint)}" if args.checkpoint else "taught JSON motion"
print(f"\n=== {label}, {N} envs ===")
print(f"smallest skeleton gap: mean {float(best.mean())*1000:.0f} mm | worst env {float(best.min())*1000:.0f} mm "
      f"| best env {float(best.max())*1000:.0f} mm")
w = int(best.argmin())
print(f"  worst env: step {int(best_step[w])}, right {seg_names[int(best_pair[w]) // S]} vs "
      f"left {seg_names[int(best_pair[w]) % S]}")
print(f"  envs under 60 mm: {int((best < 0.060).sum())}/{N}   under 40 mm: {int((best < 0.040).sum())}/{N}")
print(f"sideways swing (max |y| of tip): right {float(y_max[:, 0].mean())*1000:.0f} mm, "
      f"left {float(y_max[:, 1].mean())*1000:.0f} mm")
print(f"completed: {int(u._completed.sum())}/{N}")
print("\n step  gap mean  gap min  progress")
for s, m, lo, pr in trace[::15]:
    print(f"{s:>5} {m*1000:>8.0f} {lo*1000:>8.0f} {pr:>9.2f}")
sys.stdout.flush()
os._exit(0)
