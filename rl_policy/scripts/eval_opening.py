"""Completion-rate evaluation for Dofbot-Opening-Direct-v0.

Reward is not a health metric here either. What counts is whether both arms got
through the insertion, reached the reference spread and HELD it, and if not, at
which leg of the reference path each arm stopped.

Usage:
    python scripts/eval_opening.py <checkpoint.pt> [--device cuda:0] [--envs 64]
"""

import argparse
import os
import sys

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("checkpoint", type=str, help="path to an agent .pt file")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=1234)
args = parser.parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(
    argparse.Namespace(
        headless=True, device=args.device, enable_cameras=False,
        livestream=-1, kit_args="", experience="",
    )
).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import dofbot.tasks  # noqa: F401,E402
from skrl.utils.runner.torch import Runner  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402

N = args.envs
cfg = parse_env_cfg("Dofbot-Opening-Direct-v0", device=args.device, num_envs=N)
cfg.seed = args.seed
cfg.rsi_prob = 0.0          # every episode is the full motion, from READY
env = gym.make("Dofbot-Opening-Direct-v0", cfg=cfg)
u = env.unwrapped
env_w = SkrlVecEnvWrapper(env, ml_framework="torch")

here = os.path.dirname(os.path.abspath(__file__))
acfg = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dofbot", "tasks", "agents", "skrl_ppo_opening_cfg.yaml")))
acfg["trainer"]["close_environment_at_exit"] = False
acfg["agent"]["experiment"]["write_interval"] = 0
acfg["agent"]["experiment"]["checkpoint_interval"] = 0
runner = Runner(env_w, acfg)
runner.agent.load(args.checkpoint)
if hasattr(runner.agent, "set_mode"):
    runner.agent.set_mode("eval")

obs, _ = env_w.reset()
best = torch.zeros((N, 2), device=u.device)            # furthest weighted progress
best_leg = torch.zeros((N, 2), dtype=torch.long, device=u.device)
width_done = torch.zeros(N, device=u.device)             # when the hold completed
width_end = torch.zeros(N, device=u.device)              # at the end of the episode
held_after = torch.zeros(N, device=u.device)             # steps still holding after that
steps_after = torch.zeros(N, device=u.device)
completed = torch.zeros(N, dtype=torch.bool, device=u.device)
ended = torch.zeros(N, dtype=torch.bool, device=u.device)   # first episode over
comp_step: dict[int, int] = {}

steps = int(u.max_episode_length) - 2
for step in range(steps):
    with torch.inference_mode():
        try:
            out = runner.agent.act(obs, None, timestep=0, timesteps=0)
        except TypeError:
            out = runner.agent.act(obs, timestep=0, timesteps=0)
        obs, _, term, trunc, _ = env_w.step(out[-1].get("mean_actions", out[0]))
    live = ~ended
    prog = u._progress()
    leg = u._path_at(u._s.unsqueeze(-1))[1].squeeze(-1) // u._sub
    better = (prog > best) & live.unsqueeze(-1)
    best = torch.where(better, prog, best)
    best_leg = torch.where(better, leg, best_leg)
    tips, _ = u._tips()
    w = torch.sum((tips[:, 1, :2] - tips[:, 0, :2]) * u._spread_dir, dim=-1)
    width_end = torch.where(live, w, width_end)
    newly = u._completed & live & ~completed
    width_done = torch.where(newly, w, width_done)
    after = completed & live
    steps_after += after.float()
    held_after += (after & (u._hold > 0)).float()
    for e in newly.nonzero().flatten().tolist():
        comp_step[e] = step
    completed |= newly
    # An early termination (crash / strayed) resets the env; only the first
    # episode of each env is scored.
    ended |= term.view(-1).bool() | trunc.view(-1).bool()

b = best.cpu().numpy()
bl = best_leg.cpu().numpy()
n_done = int(completed.sum())
names = u._ref_names
target = float(torch.linalg.norm(u._ref_pts[1, -1, :2] - u._ref_pts[0, -1, :2])) * 1000

print(f"\nCOMPLETION {n_done}/{N} = {100 * n_done / N:.0f}%   (spread reached and held {cfg.hold_steps} steps)")
if comp_step:
    d = sorted(comp_step.values())
    print(f"  steps: median {np.median(d):.0f} | fastest {d[0]} | slowest {d[-1]}")
print(f"  PATH PROGRESS  right mean {b[:, 0].mean() * 100:.0f}% median {np.median(b[:, 0]) * 100:.0f}%"
      f" | left mean {b[:, 1].mean() * 100:.0f}% median {np.median(b[:, 1]) * 100:.0f}%")
print(f"  crashed or strayed before the end: {int((ended & ~completed).sum())}/{N}")
if n_done:
    c = completed
    kept = (held_after[c] / steps_after[c].clamp(min=1)).mean()
    print(f"  opening width: {float(width_done[c].mean()) * 1000:.0f} mm when completed, "
          f"{float(width_end[c].mean()) * 1000:.0f} mm at episode end (reference {target:.0f} mm)")
    print(f"  still holding afterwards: {float(kept) * 100:.0f}% of the remaining steps")

print("\nwhere each arm stopped (leg of the reference path it was on at its furthest):")
for a, arm in enumerate(("right", "left")):
    row = "  ".join(f"{names[k]}->{names[k + 1]}: {(bl[:, a] == k).sum():>2}" for k in range(len(names) - 1))
    print(f"  {arm:<5} {row}")

sys.stdout.flush()
os._exit(0)
