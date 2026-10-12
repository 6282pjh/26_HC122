"""Which incision placements can both arms physically reach?

The opening motion is the taught one moved rigidly onto the incision. Whether
that works for a given placement is decided by kinematics, not by training, so
measure it before choosing a training range.

Method: with the tool held (near) vertical the arm has three free joints, so
its reachable tip positions are a 2-D region in (distance from base axis,
height), swept around the base by joint 1. That region is sampled exhaustively
by FK in the simulator. A placement is feasible when every point of the insert
and spread legs, for both arms, lands inside the region and within joint 1's
travel.

Usage:
    python scripts/reach_map.py --device cuda:0 --out motions/opening_reach.npz
"""

import argparse
import math
import os
import sys

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--out", default="motions/opening_reach.npz")
parser.add_argument("--cell", type=float, default=0.002, help="resolution of the reach region (m)")
args = parser.parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(argparse.Namespace(headless=True, device=args.device, enable_cameras=False,
                                     livestream=-1, kit_args="", experience="")).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import dofbot.tasks  # noqa: F401,E402

N = 4096
cfg = parse_env_cfg("Dofbot-Opening-Direct-v0", device=args.device, num_envs=N)
env = gym.make("Dofbot-Opening-Direct-v0", cfg=cfg)
u = env.unwrapped
env.reset()
dev = u.device

# --- sample the vertical-tool configurations -----------------------------------
# Tool vertical <=> j2 + j3 + j4 = 180 deg in sim angles (taught end poses: 179, 176).
# A tilt of up to 20 deg is allowed, as the task itself allows (cos > 0.85).
deg = torch.arange(-90.0, 90.01, 1.5)
j2, j3 = torch.meshgrid(deg, deg, indexing="ij")
configs = []
for tilt in (-20.0, -10.0, 0.0, 10.0, 20.0):
    j4 = 180.0 + tilt - j2 - j3
    ok = (j4 >= -90.0) & (j4 <= 90.0)
    configs.append(torch.stack((j2[ok], j3[ok], j4[ok]), dim=-1))
configs = torch.deg2rad(torch.cat(configs)).to(dev)
print(f"\nFK samples per arm: {len(configs)}")

base = [torch.tensor(r.cfg.init_state.pos[:2], device=dev) for r in u.robots]
reach, forward = [], []
for a, robot in enumerate(u.robots):
    pts = []
    for i in range(0, len(configs), N):
        chunk = configs[i:i + N]
        q = robot.data.default_joint_pos.clone()
        q[:len(chunk), u._arm_dof_idx[0]] = 0.0
        for k in range(3):
            q[:len(chunk), u._arm_dof_idx[k + 1]] = chunk[:, k]
        q[:, u._j5_dof_idx] = u._ref_q[a, -1, 4]
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        u.sim.forward()
        u.scene.update(1e-5)
        tips, down = u._tips()
        t = tips[:len(chunk), a]
        d = t[:, :2] - base[a]
        pts.append(torch.stack((torch.linalg.norm(d, dim=-1), t[:, 2], down[:len(chunk), a],
                                torch.atan2(d[:, 1], d[:, 0])), dim=-1))
    pts = torch.cat(pts).cpu().numpy()
    pts = pts[(pts[:, 2] > 0.85) & (pts[:, 0] > 0.03)]
    reach.append(pts[:, :2])
    # Direction the arm points with joint 1 at zero (circular mean of azimuths).
    forward.append(math.atan2(np.sin(pts[:, 3]).mean(), np.cos(pts[:, 3]).mean()))
    print(f"{('right', 'left')[a]} arm: forward {math.degrees(forward[a]):+.0f} deg, "
          f"reach r {pts[:, 0].min()*1000:.0f}..{pts[:, 0].max()*1000:.0f} mm, "
          f"z {pts[:, 1].min()*1000:+.0f}..{pts[:, 1].max()*1000:+.0f} mm ({len(pts)} vertical samples)")

cell = args.cell
R0, Z0, NR, NZ = 0.0, -0.20, int(0.45 / cell), int(0.65 / cell)
occ = []
for pts in reach:
    g = np.zeros((NR, NZ), dtype=bool)
    ir = np.clip(((pts[:, 0] - R0) / cell).astype(int), 0, NR - 1)
    iz = np.clip(((pts[:, 1] - Z0) / cell).astype(int), 0, NZ - 1)
    g[ir, iz] = True
    # close the gaps between FK samples (one cell each way)
    d = g.copy()
    for dr in (-1, 0, 1):
        for dz in (-1, 0, 1):
            d |= np.roll(np.roll(g, dr, 0), dz, 1)
    occ.append(d)

# --- the part of the taught path that has to be reached with a vertical tool ----
k_from = u._num_pts - 1 - u._sub - u._sub // 2          # second half of the descent + the spread
ref = u._ref_pts.cpu().numpy()                          # (2, P, 3)
centre = ref[:, u._anchor_from, :2].mean(axis=0)
need = ref[:, k_from:]                                  # (2, K, 3)
j1_lim = math.radians(90.0)


def feasible(dx, dy, yaw_deg):
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    for a in range(2):
        rel = need[a, :, :2] - centre
        xy = centre + np.stack((c * rel[:, 0] - s * rel[:, 1], s * rel[:, 0] + c * rel[:, 1]), -1) + (dx, dy)
        d = xy - base[a].cpu().numpy()
        r = np.linalg.norm(d, axis=-1)
        az = (np.arctan2(d[:, 1], d[:, 0]) - forward[a] + math.pi) % (2 * math.pi) - math.pi
        if (np.abs(az) > j1_lim).any():
            return False
        ir = ((r - R0) / cell).astype(int)
        iz = ((need[a, :, 2] - Z0) / cell).astype(int)
        if (ir < 0).any() or (ir >= NR).any() or (iz < 0).any() or (iz >= NZ).any():
            return False
        if not occ[a][ir, iz].all():
            return False
    return True


dxs = np.arange(-0.080, 0.0801, 0.005)
dys = np.arange(-0.120, 0.1201, 0.005)
yaws = np.arange(-60.0, 60.01, 4.0)
grid = np.zeros((len(yaws), len(dxs), len(dys)), dtype=bool)
for i, yaw in enumerate(yaws):
    for j, dx in enumerate(dxs):
        for k, dy in enumerate(dys):
            grid[i, j, k] = feasible(dx, dy, yaw)

print(f"\ntaught placement feasible: {feasible(0.0, 0.0, 0.0)}")
print("\nfeasible shifts per turn (mm): x range at y=0 | y range at x=0 | share of the swept box")
j0, k0 = len(dxs) // 2, len(dys) // 2
for i, yaw in enumerate(yaws):
    xs = dxs[grid[i, :, k0]] * 1000
    ys = dys[grid[i, j0, :]] * 1000
    fx = f"{xs.min():+4.0f}..{xs.max():+4.0f}" if len(xs) else "   none   "
    fy = f"{ys.min():+4.0f}..{ys.max():+4.0f}" if len(ys) else "   none   "
    print(f"  turn {yaw:+4.0f} deg   x {fx}   y {fy}   {grid[i].mean() * 100:4.0f}%")

# Largest axis-aligned box of placements around the taught one that is feasible throughout.
best = None
for hy in range(0, len(yaws) // 2 + 1):
    for hx in range(0, j0 + 1):
        for hk in range(0, k0 + 1):
            i0 = len(yaws) // 2
            box = grid[i0 - hy:i0 + hy + 1, j0 - hx:j0 + hx + 1, k0 - hk:k0 + hk + 1]
            if box.all():
                vol = (2 * hy + 1) * (2 * hx + 1) * (2 * hk + 1)
                if best is None or vol > best[0]:
                    best = (vol, hx * 5, hk * 5, hy * 4)
if best:
    print(f"\nlargest fully feasible box around the taught placement: "
          f"x +-{best[1]} mm, y +-{best[2]} mm, turn +-{best[3]} deg")
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
np.savez(args.out, grid=grid, dxs=dxs, dys=dys, yaws=yaws)
print(f"saved -> {args.out}")
sys.stdout.flush()
os._exit(0)
