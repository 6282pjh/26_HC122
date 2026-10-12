"""Replay an exported trajectory in sim and check it is safe on real hardware.

The sim disables collisions on the skin patch and gives the blade no collision
geometry at all, so a rollout can drive the arm straight through the skin and
the table without anything objecting. On hardware that is a crash. This replays
the recorded joint angles and reports the lowest point of every arm link, plus
the blade tip, so the trajectory can be cleared (or rejected) before it is sent.

Usage:
    python scripts/check_trajectory_safety.py traj.npz [--device cuda:1]
"""

import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("trajectory", type=str)
parser.add_argument("--device", type=str, default="cuda:1")
parser.add_argument("--floor-margin", type=float, default=0.005,
                    help="minimum clearance above z=0 to consider safe (m)")
args = parser.parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(
    argparse.Namespace(headless=True, device=args.device, enable_cameras=False,
                       livestream=-1, kit_args="", experience="")
).app

import numpy as np  # noqa: E402
import torch
from isaaclab.utils.math import matrix_from_quat  # noqa: E402
import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import dofbot.tasks  # noqa: F401,E402

data = np.load(args.trajectory, allow_pickle=True)
traj = data["joints_rad"]
names = [str(n) for n in data["joint_names"]]

cfg = parse_env_cfg("Dofbot-Incision-Direct-v3", device=args.device, num_envs=1)
env = gym.make("Dofbot-Incision-Direct-v3", cfg=cfg)
u = env.unwrapped
robot = u.robot_incise
env.reset()

ids = [robot.joint_names.index(n) for n in names]
origin = u.scene.env_origins[0]
skin_top = u.cfg.skin_top_z

# base_link sits at z=0 by construction, so including it flags every step and
# the verdict becomes meaningless. Guard the moving links only — same set the
# env's floor penalty uses.
guard = [i for i, n in enumerate(robot.body_names) if n != "base_link"]
guard_names = [robot.body_names[i] for i in guard]
# ...and the FINGERTIPS as points. Link origins alone are not a safety check:
# rlink2's origin sits 50.5mm above its own fingertip, so an origin-only test
# passes trajectories that already have the fingers in the table. Same trap that
# hid the blade_length error (2026-08-26).
finger_idx, _ = robot.find_bodies(["rlink2", "llink2"])
FINGER_OFF = torch.tensor([u.cfg.finger_length, 0.0, 0.0], device=u.device).view(1, 3, 1)

min_link_z = np.full(len(traj), np.inf)
tip_z = np.zeros(len(traj))
worst_link = [""] * len(traj)

for k, q_row in enumerate(traj):
    q = robot.data.joint_pos.clone()
    for i, v in zip(ids, q_row):
        q[:, i] = float(v)
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    robot.set_joint_position_target(q)
    u.scene.write_data_to_sim()
    u.sim.step(render=False)
    u.scene.update(u.physics_dt)
    z = (robot.data.body_pos_w[0, guard, 2] - origin[2]).cpu().numpy()
    rot = matrix_from_quat(robot.data.body_quat_w[0][finger_idx])
    tips = robot.data.body_pos_w[0][finger_idx] + (rot @ FINGER_OFF).squeeze(-1)
    ftz = (tips[:, 2] - origin[2]).cpu().numpy()
    if ftz.min() < z.min():
        min_link_z[k] = ftz.min()
        worst_link[k] = ["rlink2", "llink2"][int(ftz.argmin())] + " 손끝"
    else:
        min_link_z[k] = z.min()
        worst_link[k] = guard_names[int(z.argmin())]
    tip_z[k] = float(u._blade_tip_pos_w()[0, 2] - origin[2])

lo = int(min_link_z.argmin())
print(f"\n스텝 {len(traj)}개 검사")
print(f"팔 링크 최저점 : {min_link_z.min():+.4f} m  (스텝 {lo}, {worst_link[lo]})")
print(f"칼날 끝 최저점 : {tip_z.min():+.4f} m  (스텝 {int(tip_z.argmin())})")
print(f"피부 표면      : {skin_top:+.4f} m")

# Three outcomes, not one. Conflating them made every valid cut fail: with
# blade_length 60.5 and finger_length 50.5 the fingertip rides exactly 10 mm
# above the blade, so cutting 5 mm deep puts the fingers at +5 mm -- right on
# the old 5 mm margin, by construction.
#
#   DANGER  blade or finger reaching the TABLE. This breaks hardware.
#   NOTE    finger dipping below the skin SURFACE. The patch is 10 mm of
#           compliant foam, so light contact is not a crash; depth decides.
table_z = u.cfg.blade_floor_z
blade_hits_table = (tip_z < table_z).sum()
link_hits_table = (min_link_z < table_z).sum()
finger_in_skin = ((min_link_z < 0.0) & (min_link_z >= table_z)).sum()

print(f"\n[위험] 칼끝이 수술대({table_z*1000:+.0f}mm) 아래   : {blade_hits_table} / {len(traj)}")
print(f"[위험] 팔 링크가 수술대 아래           : {link_hits_table} / {len(traj)}")
print(f"[참고] 손끝이 피부 표면 아래           : {finger_in_skin} / {len(traj)}"
      f"   (최대 {abs(min(0.0, min_link_z.min()))*1000:.1f}mm)")
print(f"(참고) blade_length = {u.cfg.blade_length*1000:.1f}mm, finger_length = {u.cfg.finger_length*1000:.1f}mm")

# Joint-limit saturation. Sim clamps a joint at its limit for free; a real
# serial servo commanded to the end of its range pushes against the mechanical
# stop, draws stall current and heats, with no torque feedback to stop it.
# Measured on the first exported trajectory: arm_joint3 sat at servo 0 deg for
# every frame of the cut, i.e. 7 s leaning on the stop.
lim = robot.data.soft_joint_pos_limits[0]
sat_frames = {}
for col, name in enumerate(names):
    j = robot.joint_names.index(name)
    lo, hi = float(lim[j, 0]), float(lim[j, 1])
    span = max(hi - lo, 1e-6)
    n = sum(1 for v in traj[:, col] if (v - lo) / span < 0.02 or (hi - v) / span < 0.02)
    if n:
        sat_frames[name] = n
print()
if sat_frames:
    for n_, c in sorted(sat_frames.items(), key=lambda kv: -kv[1]):
        print(f"[위험] {n_} 이 가동범위 끝에 붙은 스텝 : {c} / {len(traj)}")
else:
    print("[확인] 가동범위 끝에 붙는 관절 없음")

if blade_hits_table or link_hits_table or sat_frames:
    print("\n=> 실물 재생 금지.")
    if blade_hits_table or link_hits_table:
        print("   칼날 또는 팔이 수술대에 닿습니다. sim은 칼끝이 계산된 점이라")
        print("   테이블을 그냥 통과하지만 실물은 부딪힙니다.")
    if sat_frames:
        print("   관절이 가동범위 끝에 눌려 있습니다. 실물 서보는 스토퍼를 밀며")
        print("   과전류·발열 상태가 되고, 토크 피드백이 없어 스스로 멈추지 않습니다.")
elif finger_in_skin:
    print(f"\n=> 재생 가능. 손끝이 피부를 최대 {abs(min_link_z.min())*1000:.1f}mm 누르지만")
    print("   10mm 발포 패치 안이고 수술대에는 닿지 않습니다.")
    print("   첫 실행은 칼날 제거 + 피부 제거 상태로 확인할 것.")
else:
    print("\n=> 재생 가능. 피부·수술대 어디에도 닿지 않습니다.")
    print("   첫 실행은 칼날 제거 상태로 확인할 것.")

env.close()
app.close()
