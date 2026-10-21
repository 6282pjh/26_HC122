"""Play the reference opening path through the opening environment.

No policy: the reference waypoints are interpolated in joint space, the way the
servos move, and fed in as actions. This checks the TASK, not a policy -- if
the reference path itself does not score as a completed opening, the task
definition is wrong and training against it would be pointless.

Usage:
    python scripts/replay_reference_in_env.py --device cuda:0
    python scripts/replay_reference_in_env.py --randomize   # with line shifts
"""

import argparse
import os

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps_per_leg", type=int, default=60, help="60 = 1 s per move, as in the reference")
parser.add_argument("--randomize", action="store_true", help="keep the incision placement randomization")
args = parser.parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(argparse.Namespace(headless=True, device=args.device, enable_cameras=False,
                                     livestream=-1, kit_args="", experience="")).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import dofbot.tasks  # noqa: F401,E402

cfg = parse_env_cfg("Dofbot-Opening-Direct-v0", device=args.device, num_envs=args.num_envs)
if not args.randomize:
    cfg.line_shift_x_range = (0.0, 0.0)
    cfg.line_shift_y_range = (0.0, 0.0)
    cfg.line_yaw_range_deg = (0.0, 0.0)
    cfg.reset_joint_noise = 0.0
cfg.rsi_prob = 0.0          # the replay always starts at READY
env = gym.make("Dofbot-Opening-Direct-v0", cfg=cfg)
u = env.unwrapped
env.reset()

ref = u._way_q[:, :, :4]                      # (2, waypoints, 4)
legs = ref.shape[1] - 1
hold = cfg.hold_steps + 30
total_reward = torch.zeros(u.num_envs, device=u.device)
print(f"\n{'step':>5} {'leg':<16}{'progress R/L':>14}{'width mm':>10}{'on path':>9}{'hold':>6}")
step = 0
for leg in range(legs + 1):
    n = args.steps_per_leg if leg < legs else hold
    for i in range(n):
        if leg < legs:
            a = (i + 1) / n
            want = ref[:, leg] * (1 - a) + ref[:, leg + 1] * a
        else:
            want = ref[:, -1]
        cur = u._joint_targets[:, :, u._arm_dof_idx]                  # (n, 2, 4)
        act = ((want.unsqueeze(0) - cur) / cfg.joint_delta_scale).clamp(-1.0, 1.0)
        _, r, term, trunc, _ = env.step(act.reshape(u.num_envs, 8))
        total_reward += r
        step += 1
        if (i + 1) % 15 == 0 or i == n - 1:
            log = u.extras["log"]
            name = u._ref_names[leg + 1] if leg < legs else "(hold)"
            print(f"{step:>5} {name:<16}{float(log['open_progress_right']):>7.2f}"
                  f"{float(log['open_progress_left']):>7.2f}{float(log['opening_width_m'])*1000:>10.0f}"
                  f"{float(log['on_path_frac']):>9.2f}{int(u._hold.float().mean()):>6}")
        if bool(term.any()):
            print(f"  !! crashed at step {step}: {int(term.sum())} envs")

done = int(u._completed.sum())
tips, down = u._tips()
err = torch.linalg.norm(tips - u._path[:, :, -1], dim=-1) * 1000
print(f"\n완료(개복 유지 {cfg.hold_steps} 스텝): {done}/{u.num_envs}")
print(f"최종 끝점 오차  오른팔 {float(err[:, 0].mean()):.1f} mm / 왼팔 {float(err[:, 1].mean()):.1f} mm")
print(f"최종 수직도     오른팔 {float(down[:, 0].mean()):.2f} / 왼팔 {float(down[:, 1].mean()):.2f}")
print(f"에피소드 보상   {float(total_reward.mean()):.0f}")
import sys  # noqa: E402
sys.stdout.flush()
os._exit(0)
