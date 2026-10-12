"""Completion-rate evaluation for Dofbot-Incision-Direct-v3.

Reward is NOT a health metric for this task: a policy can post an all-time-high
reward while completing 31% of lines, because the dense posture/proximity terms
dominate and get farmed on headings that can never finish. Judge runs with this
instead.

Usage:
    python scripts/eval_incision.py <checkpoint.pt> [--device cuda:0] [--envs 64]

Prints overall completion plus, per travel-direction bucket, which of the four
waypoint gates (xy / depth / vertical / blade) is the one staying shut.
"""

import argparse
import os

parser = argparse.ArgumentParser(description=__doc__)
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
cfg = parse_env_cfg("Dofbot-Incision-Direct-v3", device=args.device, num_envs=N)
cfg.seed = args.seed
env = gym.make("Dofbot-Incision-Direct-v3", cfg=cfg)
u = env.unwrapped
env_w = SkrlVecEnvWrapper(env, ml_framework="torch")

acfg = yaml.safe_load(
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "task", "dofbot", "dofbot", "tasks", "direct", "dofbot", "agents", "skrl_ppo_incision_v3_cfg.yaml"))
)
acfg["trainer"]["close_environment_at_exit"] = False
acfg["agent"]["experiment"]["write_interval"] = 0
acfg["agent"]["experiment"]["checkpoint_interval"] = 0
runner = Runner(env_w, acfg)
runner.agent.load(args.checkpoint)
if hasattr(runner.agent, "set_mode"):
    runner.agent.set_mode("eval")

obs, _ = env_w.reset()
ang = (torch.rad2deg(torch.atan2(u._line_dir[:, 1], u._line_dir[:, 0])) % 360.0).cpu().numpy()

comp_step: dict[int, int] = {}
gate = torch.zeros((N, 4), device=u.device)
best_xy = torch.full((N,), 9.0, device=u.device)
best_blade = torch.full((N,), -9.0, device=u.device)
best_down = torch.full((N,), -9.0, device=u.device)
best_cut = torch.zeros((N,), device=u.device)

steps = int(u.max_episode_length) - 2
for step in range(steps):
    with torch.inference_mode():
        try:
            out = runner.agent.act(obs, None, timestep=0, timesteps=0)
        except TypeError:
            out = runner.agent.act(obs, timestep=0, timesteps=0)
        obs, _, _, _, _ = env_w.step(out[-1].get("mean_actions", out[0]))
    tip = u._blade_tip_pos_w()
    delta = tip - u._current_target_pos_w()
    # Distance from the LINE, not from the target point: while stroking, the tip
    # runs ahead of where the cut currently ends and that is correct behaviour.
    _, xy = u._tip_line_coords()
    z = delta[:, 2].abs()
    dn = u._tool_downward_cos()
    bl = u._blade_align_cos()
    best_xy = torch.minimum(best_xy, xy)
    best_cut = torch.maximum(best_cut, u._cut_s / u._line_len.clamp(min=1e-6))
    best_down = torch.maximum(best_down, dn)
    best_blade = torch.maximum(best_blade, bl)
    gate[:, 0] += (xy < u.cfg.cut_lateral_tol).float()
    gate[:, 1] += (z < u.cfg.waypoint_z_tol).float()
    gate[:, 2] += (dn > u.cfg.vertical_cos_tol).float()
    gate[:, 3] += (bl > u.cfg.blade_align_tol).float()
    for e in range(N):
        if e not in comp_step and bool(u._completed[e].item()):
            comp_step[e] = step

comp = u._completed.cpu().numpy()
g = gate.cpu().numpy()
bx, bd, bb, bc = (t.cpu().numpy() for t in (best_xy, best_down, best_blade, best_cut))
done = sorted(comp_step.values())

print(f"\nCOMPLETION {len(comp_step)}/{N} = {100 * len(comp_step) / N:.0f}%")
if done:
    print(f"  steps: median {np.median(done):.0f} | fastest {done[0]} | slowest {done[-1]}")
print(f"  mean blade align {u._blade_align_cos().mean():.3f} | "
      f"mean downward {u._tool_downward_cos().mean():.3f}")
print(f"  CUT FRACTION mean {bc.mean() * 100:.0f}% | median {np.median(bc) * 100:.0f}% "
      f"| best {bc.max() * 100:.0f}%   <- how much of the line was actually cut")

print(f"\nby travel direction (gates are steps open out of {steps}):")
edges = list(range(0, 360, 15))
for lo in edges:
    m = (ang >= lo) & (ang < lo + 15)
    if m.sum() == 0:
        continue
    print(f"  [{lo:>3},{lo + 15:>3})  n={m.sum():>2}  done={comp[m].sum():>2}   "
          f"cut={bc[m].mean() * 100:>3.0f}% off-line={bx[m].min() * 1000:>4.1f}mm "
          f"down={bd[m].max():.2f} blade={bb[m].max():.2f}   "
          f"gates line={g[m, 0].max():>3.0f} z={g[m, 1].max():>3.0f} "
          f"down={g[m, 2].max():>3.0f} blade={g[m, 3].max():>3.0f}")

env_w.close()
app.close()
