"""Roll the trained policy out in sim and save the joint trajectory for replay.

First real-world tests should replay a trajectory, not run the policy live: the
policy is closed-loop on the blade-tip position, which on hardware needs servo
readback plus forward kinematics. Validating the servo mapping and the mechanics
first, with those two out of the picture, keeps failures diagnosable.

Usage:
    python scripts/export_trajectory.py <checkpoint.pt> \
        --heading 96 --out traj.npz [--device cuda:0]

Saves joint targets in RADIANS, in sim joint order (arm_joint1..5, grip_joint).
"""

import argparse
import os

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("checkpoint", type=str)
parser.add_argument("--out", type=str, default="traj.npz")
parser.add_argument("--heading", type=float, default=96.0,
                    help="line travel direction in deg; must be in the trained window [60,150)")
parser.add_argument("--length", type=float, default=0.08, help="line length in m")
parser.add_argument("--cx", type=float, default=0.20, help="line centre x (robot-local, m)")
parser.add_argument("--cy", type=float, default=0.0, help="line centre y (robot-local, m)")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--steps", type=int, default=420)
args = parser.parse_args()

if not (60.0 <= args.heading < 150.0):
    raise SystemExit(
        f"heading {args.heading} is outside the trained window [60,150). "
        "Rotate the skin jig 90 deg and re-derive the heading instead."
    )

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(
    argparse.Namespace(headless=True, device=args.device, enable_cameras=False,
                       livestream=-1, kit_args="", experience="")
).app

import math  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import dofbot.tasks  # noqa: F401,E402
from skrl.utils.runner.torch import Runner  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402

cfg = parse_env_cfg("Dofbot-Incision-Direct-v3", device=args.device, num_envs=1)
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

# Force the exact line we want to cut, instead of whatever was sampled.
dev = u.device
n_pts = u.cfg.num_waypoints
theta = math.radians(args.heading)
dx = 0.5 * args.length * math.cos(theta)
dy = 0.5 * args.length * math.sin(theta)
start = torch.tensor([[args.cx - dx, args.cy - dy]], device=dev)
end = torch.tensor([[args.cx + dx, args.cy + dy]], device=dev)
t = torch.linspace(0.0, 1.0, n_pts, device=dev).view(1, n_pts, 1)
xy = start.unsqueeze(1) * (1.0 - t) + end.unsqueeze(1) * t
z = torch.full((1, n_pts, 1), u.cfg.skin_top_z - u.cfg.incision_depth, device=dev)
u._waypoints_local[:] = torch.cat((xy, z), dim=-1)
d = end - start
u._line_dir[:] = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
u._target_idx[:] = 0
u._completed[:] = False
# The cut is defined by the line's start, direction and length.
u._line_start[:] = start
u._line_len[:] = float(args.length)
u._cut_s[:] = 0.0
u._was_cutting[:] = False
# The observation returned by reset() was built for the sampled line.
obs = u._get_observations()["policy"]

arm_ids = u._arm_dof_idx
grip_id = u._grip_dof_idx[0]
traj, tips, events = [], [], []
last = 0
for step in range(args.steps):
    with torch.inference_mode():
        try:
            out = runner.agent.act(obs, None, timestep=0, timesteps=0)
        except TypeError:
            out = runner.agent.act(obs, timestep=0, timesteps=0)
        obs, _, _, _, _ = env_w.step(out[-1].get("mean_actions", out[0]))
    q = u.robot_incise.data.joint_pos[0]
    traj.append([float(q[i]) for i in arm_ids] + [float(q[grip_id])])
    tips.append((u._blade_tip_pos_w()[0] - u.scene.env_origins[0]).cpu().numpy())
    idx = int(u._target_idx[0].item())
    if idx != last:
        events.append((step, idx))
        last = idx

traj = np.asarray(traj, dtype=np.float64)
tips = np.asarray(tips, dtype=np.float64)
completed = bool(u._completed[0].item())
print(f"completed: {completed} | waypoints: {events}")
print(f"trajectory: {traj.shape[0]} steps x {traj.shape[1]} joints (rad, sim order)")
print(f"blade tip z: min {tips[:, 2].min():.4f} max {tips[:, 2].max():.4f}")

np.savez(
    args.out,
    joints_rad=traj,
    blade_tip_local=tips,
    waypoints_local=u._waypoints_local[0].cpu().numpy(),
    line_start_local=start[0].cpu().numpy(),
    line_end_local=end[0].cpu().numpy(),
    heading_deg=args.heading,
    control_hz=1.0 / (u.cfg.sim.dt * u.cfg.decimation),
    completed=completed,
    joint_names=np.array(
        [u.robot_incise.joint_names[i] for i in arm_ids]
        + [u.robot_incise.joint_names[grip_id]]
    ),
)
print(f"saved -> {args.out}")

# Also write CSV: the AGX replays from this, and reading it needs nothing but
# the Python standard library, so the Jetson stays at its stock image.
csv_path = args.out.rsplit(".", 1)[0] + ".csv"
with open(csv_path, "w") as fh:
    fh.write("# arm_joint1,arm_joint2,arm_joint3,arm_joint4,arm_joint5,grip_joint (radians)\n")
    fh.write(f"# heading {args.heading} deg, recorded at "
             f"{1.0 / (u.cfg.sim.dt * u.cfg.decimation):.0f} Hz, completed={completed}\n")
    for row in traj:
        fh.write(",".join(f"{v:.6f}" for v in row) + "\n")
print(f"saved -> {csv_path}  (copy this one to the AGX)")

if not completed:
    print("WARNING: this rollout did not finish the line; pick another heading or re-run.")

env_w.close()
app.close()
