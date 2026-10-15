"""Render one successful incision rollout to mp4.

Picks an env whose line runs in a heading the policy handles, aims the viewport
at it, and records. Note `env.reset()` does NOT resample the line, so the
heading has to be chosen by scanning many envs rather than by resetting one.

Usage:
    python scripts/render_incision.py <checkpoint.pt> \
        [--device cuda:0] [--out videos/incision] [--lo 85] [--hi 120]

On a shared GPU, run this under `timeout` — Isaac's render path touches GPU 0
even when told to use another device, and stalls indefinitely when GPU 0 is
saturated.
"""

import argparse
import os

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("checkpoint", type=str, help="path to an agent .pt file")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--out", type=str, default="videos/incision")
parser.add_argument("--envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=420)
parser.add_argument("--lo", type=float, default=85.0, help="min heading (deg) to film")
parser.add_argument("--hi", type=float, default=120.0, help="max heading (deg) to film")
args = parser.parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(
    argparse.Namespace(
        headless=True, device=args.device, enable_cameras=True,
        livestream=-1, kit_args="", experience="",
    )
).app

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
env = gym.make("Dofbot-Incision-Direct-v3", cfg=cfg, render_mode="rgb_array")
env = gym.wrappers.RecordVideo(
    env, video_folder=args.out, step_trigger=lambda s: s == 0,
    video_length=args.steps, disable_logger=True,
)
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
ang = torch.rad2deg(torch.atan2(u._line_dir[:, 1], u._line_dir[:, 0])) % 360.0
cand = [i for i in range(N) if args.lo <= float(ang[i]) <= args.hi]
pick = cand[0] if cand else 0
o = u.scene.env_origins[pick].cpu().numpy()
print(f"filming env {pick}, heading {float(ang[pick]):.0f} deg")
u.viewport_camera_controller.update_view_location(
    (o[0] + 0.5, o[1] + 0.4, o[2] + 0.32), (o[0] + 0.2, o[1], o[2] + 0.04)
)

events, last = [], 0
for step in range(args.steps):
    with torch.inference_mode():
        try:
            out = runner.agent.act(obs, None, timestep=0, timesteps=0)
        except TypeError:
            out = runner.agent.act(obs, timestep=0, timesteps=0)
        obs, _, _, _, _ = env_w.step(out[-1].get("mean_actions", out[0]))
    idx = int(u._target_idx[pick].item())
    if idx != last:
        events.append((step, idx))
        last = idx

print("waypoints:", events)
print("completed:", bool(u._completed[pick].item()),
      f"| batch {int(u._completed.sum().item())}/{N}")
print(f"blade align {float(u._blade_align_cos()[pick]):.2f} | "
      f"downward {float(u._tool_downward_cos()[pick]):.2f}")
print(f"wrote {args.out}/rl-video-step-0.mp4")

env_w.close()
app.close()
