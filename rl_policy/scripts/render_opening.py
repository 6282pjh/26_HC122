"""Render one opening rollout to mp4, for either opening task.

Usage:
    python scripts/render_opening.py <checkpoint.pt> --task Dofbot-Opening-Direct-v0 \
        [--device cuda:0] [--out videos/opening] [--steps 600]

On a shared GPU, run this under `timeout` -- Isaac's render path touches GPU 0
even when told to use another device, and stalls when GPU 0 is saturated.
"""

import argparse
import os
import sys

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("checkpoint", type=str)
parser.add_argument("--task", default="Dofbot-Opening-Direct-v0")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--out", type=str, default="videos/opening")
parser.add_argument("--envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--top", action="store_true", help="look straight down on the rig")
parser.add_argument("--iso", action="store_true", help="look down at 45 degrees from a front corner")
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

V1 = args.task.endswith("v1")
cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.envs)
if not V1:
    cfg.rsi_prob = 0.0
env = gym.make(args.task, cfg=cfg, render_mode="rgb_array")
env = gym.wrappers.RecordVideo(
    env, video_folder=args.out, step_trigger=lambda s: s == 0,
    video_length=args.steps, disable_logger=True,
)
u = env.unwrapped
env_w = SkrlVecEnvWrapper(env, ml_framework="torch")

here = os.path.dirname(os.path.abspath(__file__))
name = "skrl_ppo_opening_v1_cfg.yaml" if V1 else "skrl_ppo_opening_cfg.yaml"
acfg = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dofbot", "tasks", "agents", name)))
acfg["trainer"]["close_environment_at_exit"] = False
acfg["agent"]["experiment"]["write_interval"] = 0
acfg["agent"]["experiment"]["checkpoint_interval"] = 0
runner = Runner(env_w, acfg)
runner.agent.load(args.checkpoint)
if hasattr(runner.agent, "set_mode"):
    runner.agent.set_mode("eval")

obs, _ = env_w.reset()
o = u.scene.env_origins[0].cpu().numpy()
if args.top:
    # A hair off vertical so the camera's up vector stays defined.
    u.viewport_camera_controller.update_view_location(
        (o[0] + 0.185, o[1] - 0.002, o[2] + 1.05), (o[0] + 0.185, o[1], o[2])
    )
elif args.iso:
    # 45 deg elevation, 45 deg round from the front: 0.50 m out along each
    # horizontal axis gives 0.71 m horizontally, matched by 0.71 m of height.
    # Aimed at mid-height of the folded arms so READY is in frame too.
    u.viewport_camera_controller.update_view_location(
        (o[0] + 0.185 + 0.50, o[1] - 0.50, o[2] + 0.14 + 0.71), (o[0] + 0.185, o[1], o[2] + 0.14)
    )
else:
    u.viewport_camera_controller.update_view_location(
        (o[0] + 0.185, o[1] - 0.62, o[2] + 0.42), (o[0] + 0.185, o[1], o[2] + 0.10)
    )

for step in range(args.steps):
    with torch.inference_mode():
        try:
            out = runner.agent.act(obs, None, timestep=0, timesteps=0)
        except TypeError:
            out = runner.agent.act(obs, timestep=0, timesteps=0)
        obs, _, _, _, _ = env_w.step(out[-1].get("mean_actions", out[0]))
    if step % 60 == 59 or step == args.steps - 1:
        log = u.extras.get("log", {})
        keys = [k for k in log if any(s in k for s in ("progress", "completed", "error_rad", "width", "gap_m", "holding"))]
        print(f"step {step + 1:>4}  " + "  ".join(f"{k.split('/')[-1]}={float(log[k]):.3f}" for k in keys))

print(f"wrote {args.out}/rl-video-step-0.mp4")
sys.stdout.flush()
env.close()
os._exit(0)
