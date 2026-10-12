"""Where do the hand-taught JSON waypoints put the tools? Kinematic replay in sim.

Reads motion files such as motions/incision_dual.json (servo degrees per arm)
and prints, for every waypoint and both arms, the simulated joint angles, the
blade-tip and fingertip positions, how vertical the tool is and which way the
blade edge and the jaw axis point. Use it to compare a taught motion against
what a task assumes.

Servo deg -> sim rad uses the bus-8 calibration (j1-4 inverted about 90,
j5 = servo - 225, grip 0..180 -> -1.54..0). Bus 1 is ASSUMED to match.
Each segment is interpolated linearly in joint space, as the servos move;
--out saves every interpolated sample.

Usage:
    python scripts/json_fk.py motions/incision_dual.json motions/opening_dual.json
"""
import argparse, json, math
p = argparse.ArgumentParser()
p.add_argument("files", nargs="+")
p.add_argument("--device", default="cuda:0")
p.add_argument("--sub", type=int, default=10)
p.add_argument("--out", default=None)
args = p.parse_args()
from isaaclab.app import AppLauncher
app = AppLauncher(argparse.Namespace(headless=True, device=args.device, enable_cameras=False,
                                     livestream=-1, kit_args="", experience="")).app
import torch, gymnasium as gym
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
import dofbot.tasks  # noqa
from isaaclab.utils.math import matrix_from_quat

cfg = parse_env_cfg("Dofbot-Incision-Direct-v3", device=args.device, num_envs=1)
env = gym.make("Dofbot-Incision-Direct-v3", cfg=cfg)
u = env.unwrapped
env.reset()
dev = u.device
org = u.scene.env_origins[0]

def servo_to_sim(j):
    q = [math.radians(90.0 - j[i]) for i in range(4)]
    q.append(math.radians(j[4] - 225.0))
    g = -1.54 + j[5] / 180.0 * 1.54
    return q, g

def pose(robot, j):
    q, g = servo_to_sim(j)
    full = robot.data.default_joint_pos.clone()
    arm, _ = robot.find_joints(["arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5"], preserve_order=True)
    grip, _ = robot.find_joints(["grip_joint"])
    for k, i in enumerate(arm):
        full[0, i] = q[k]
    full[0, grip[0]] = g
    # finger linkage mimics grip_joint with multiplier +/-1 (see cfg note)
    for name in robot.joint_names:
        if "link_joint" in name:
            i = robot.find_joints([name])[0][0]
            d = robot.data.default_joint_pos[0, i]; g0 = robot.data.default_joint_pos[0, grip[0]]
            sign = 0.0 if abs(float(g0)) < 1e-6 else float(d / g0)
            full[0, i] = sign * g
    robot.write_joint_state_to_sim(full, torch.zeros_like(full))
    robot.set_joint_position_target(full)
    lim = robot.data.joint_pos_limits[0, arm]
    qq = torch.tensor(q, device=dev)
    over = [(k + 1, math.degrees(float(qq[k] - lim[k, 1])) if qq[k] > lim[k, 1] else math.degrees(float(qq[k] - lim[k, 0])))
            for k in range(5) if qq[k] > lim[k, 1] + 1e-4 or qq[k] < lim[k, 0] - 1e-4]
    return q, over

def measure(robot):
    wi = u._wrist_body_idx[0]
    ei = u._ee_body_idx[0]
    fi, _ = robot.find_bodies(["rlink2", "llink2"])
    pos = robot.data.body_pos_w[0] - org
    quat = robot.data.body_quat_w[0]
    Rw = matrix_from_quat(quat[wi].unsqueeze(0))[0]
    tool, edge, jaw = Rw[:, 2], Rw[:, 0], Rw[:, 1]
    blade = pos[ei] + u.cfg.blade_length * tool
    Rf = matrix_from_quat(quat[fi])
    off = torch.tensor([u.cfg.finger_length, 0.0, 0.0], device=dev)
    tips = pos[fi] + Rf @ off
    mid = tips.mean(0)
    lowz = float(pos[1:, 2].min())
    return dict(blade=blade.tolist(), tips=tips.tolist(), mid=mid.tolist(),
                down=float(-tool[2]), edge_deg=math.degrees(math.atan2(float(edge[1]), float(edge[0]))) % 180.0,
                jaw_deg=math.degrees(math.atan2(float(jaw[1]), float(jaw[0]))) % 180.0,
                tool=tool.tolist(), min_link_z=lowz)

def refresh():
    u.sim.forward()
    u.scene.update(1e-5)

print("\nBODIES", u.robot_incise.body_names)
print("JOINTS", u.robot_incise.joint_names)
print("DEFAULT", [round(float(v), 3) for v in u.robot_incise.data.default_joint_pos[0]])
result = {}
for f in args.files:
    d = json.load(open(f))
    print(f"\n================ {d['sequence_name']} ================")
    wps = d["waypoints"]
    rows = []
    for n, w in enumerate(wps):
        prev = wps[n - 1] if n else w
        steps = args.sub if n else 1
        for s in range(1, steps + 1):
            a = s / steps
            jr = [prev["right"]["joints"][i] * (1 - a) + w["right"]["joints"][i] * a for i in range(6)]
            jl = [prev["left"]["joints"][i] * (1 - a) + w["left"]["joints"][i] * a for i in range(6)]
            qr, over_r = pose(u.robot_incise, jr)
            ql, over_l = pose(u.robot_retract, jl)
            refresh()
            rows.append(dict(wp=w["index"], name=w["name"], a=a, servo_r=jr, servo_l=jl,
                             q_r=qr, q_l=ql, over_r=over_r, over_l=over_l,
                             R=measure(u.robot_incise), L=measure(u.robot_retract)))
    result[d["sequence_name"]] = rows
    mm = lambda v: "(" + ",".join(f"{x*1000:+6.0f}" for x in v) + ")"
    for arm in ("R", "L"):
        print(f"\n--- {'RIGHT bus8 (knife base, origin)' if arm=='R' else 'LEFT bus1 (base x=370)'} ---  mm, knife-arm frame, z=0 skin top")
        print(f"{'wp':<20}{'sim deg j1..j5':<32}{'blade tip':<24}{'finger mid':<24}{'down':>6}{'edge':>6}{'jaw':>6}{'minZ':>6}  limit")
        for r in rows:
            if r["a"] < 1.0: continue
            m = r[arm]; q = r["q_r" if arm == "R" else "q_l"]; ov = r["over_r" if arm == "R" else "over_l"]
            print(f"{r['name']:<20}{' '.join(f'{math.degrees(x):+5.0f}' for x in q):<32}{mm(m['blade']):<24}{mm(m['mid']):<24}"
                  f"{m['down']:6.2f}{m['edge_deg']:6.0f}{m['jaw_deg']:6.0f}{m['min_link_z']*1000:6.0f}  "
                  + (" ".join(f"j{k}{d:+.0f}deg" for k, d in ov) if ov else "ok"))
    # tip-to-tip distance between the arms
    print("\n--- 양팔 손끝 중점 간 거리 / 최저점 ---")
    for r in rows:
        if r["a"] < 1.0: continue
        a = torch.tensor(r["R"]["mid"]); b = torch.tensor(r["L"]["mid"])
        print(f"{r['name']:<20} 거리 {float((a-b).norm())*1000:6.0f} mm   dxy {mm((b-a).tolist())}")
if args.out:
    json.dump(result, open(args.out, "w"))
import os, sys
sys.stdout.flush()
os._exit(0)
