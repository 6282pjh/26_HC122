#!/usr/bin/env python3
"""Replay a trajectory directly on the DOFBOT. RUNS ON THE JETSON AGX.

No network, no TCP server: this drives Arm_Lib itself. Copy the .csv produced by
export_trajectory.py onto the AGX (USB stick is fine) and run it there.

Only the Python standard library plus Arm_Lib is required — no numpy, no torch,
no Isaac. The AGX needs nothing installed beyond the stock DOFBOT image.

SAFETY
  * Run --dry-run first and read the printed servo angles.
  * First live run: NO BLADE fitted, skin removed, workspace clear.
  * --rate downsamples the 60 Hz sim trajectory; servos need ~300 ms per move,
    so 5 Hz is the default. --max-step-deg refuses oversized jumps.

Usage:
    python3 replay_on_agx.py traj.csv --dry-run
    python3 replay_on_agx.py traj.csv --grip-deg 180
"""

import argparse
import csv
import json
import math
import sys
import time

# Sim joint limits (rad) in order arm_joint1..5, grip_joint, from the DOFBOT
# URDF, paired with servo ranges from Yahboom manual section 6.5.
JOINTS = [
    ("arm_joint1", -1.5708, 1.5708, 0.0, 180.0, False),
    ("arm_joint2", -1.5708, 1.5708, 0.0, 180.0, False),
    ("arm_joint3", -1.5708, 1.5708, 0.0, 180.0, False),
    ("arm_joint4", -1.5708, 1.5708, 0.0, 180.0, False),
    ("arm_joint5", -1.5708, 3.14159, 0.0, 270.0, False),
    ("grip_joint", -1.54, 0.0, 0.0, 180.0, False),
]


def connect_arm(bus, addr):
    """Attach Arm_Lib to a specific DOFBOT.

    Arm_Device() opens I2C bus 1 unconditionally. With two robots on the AGX
    that silently drives the wrong arm, so the bus/address are swapped in here.
    Right arm measured as bus 8 / 0x15; find the other with `i2cdetect -y <bus>`.
    """
    from Arm_Lib import Arm_Device
    import smbus

    print(f"DOFBOT 연결: I2C 버스 {bus}, 주소 0x{addr:02X}")
    arm = Arm_Device()
    try:
        arm.bus.close()
    except Exception:
        pass
    arm.bus = smbus.SMBus(bus)
    arm.addr = addr
    time.sleep(0.1)
    return arm


def load_calibration(path):
    """Per-joint direction flags from calibrate_joints.py.

    The nominal mapping comes from Yahboom's reference client, which targeted a
    different USD conversion; on this arm some joints run the opposite way, which
    makes it fold backward where the sim reached forward. Measure, don't assume.
    """
    if not path:
        return {}
    with open(path) as f:
        cal = json.load(f)
    # Keys like _i2c / _note carry metadata, not joint entries — skip anything
    # that is not a joint dict so the file can stay self-documenting.
    flags = {
        name: bool(v.get("inverted", False))
        for name, v in cal.items()
        if not name.startswith("_") and isinstance(v, dict)
    }
    flipped = [n for n, v in flags.items() if v]
    print("calibration: " + (("inverted " + ", ".join(flipped)) if flipped
                             else "no joints inverted"))
    return flags


def to_servo_deg(q_rad, calibration=None):
    calibration = calibration or {}
    out = []
    for value, (name, lo, hi, a, b, inverted) in zip(q_rad, JOINTS):
        inverted = calibration.get(name, inverted)
        v = max(lo, min(hi, float(value)))
        deg = (v - lo) * (b - a) / (hi - lo) + a
        if inverted:
            deg = b - (deg - a)
        out.append(deg)
    return out


def read_pose(arm):
    """Current servo angles, or None if the arm will not report them."""
    try:
        pose = [float(arm.Arm_serial_servo_read(i + 1)) for i in range(6)]
    except Exception as exc:
        print("servo readback failed: %s: %s" % (type(exc).__name__, exc))
        return None
    if any(v is None for v in pose):
        return None
    return pose


def retreat(arm, args):
    """Lift the blade clear before letting go of the arm.

    Servos hold position, so simply exiting leaves the blade wherever it
    stopped -- inside the skin if the cut was in progress. Straighten the elbow
    and wrist to raise the tool without moving the base, so the blade comes up
    along its own axis rather than dragging sideways through the material.
    """
    if args.no_retreat:
        print("--no-retreat: 팔을 그대로 둡니다. 칼날이 재료에 박혀 있을 수 있습니다.")
        return
    try:
        here = read_pose(arm)
        if here is None:
            print("후퇴 생략: 현재 자세를 읽지 못했습니다. 전원을 확인하세요.")
            return
        up = list(here)
        up[1] = min(max(here[1] - args.retreat_deg, 0.0), 180.0)   # j2
        up[2] = min(max(here[2] + args.retreat_deg, 0.0), 180.0)   # j3
        print("후퇴: j2 %.0f -> %.0f, j3 %.0f -> %.0f"
              % (here[1], up[1], here[2], up[2]))
        arm.Arm_serial_servo_write6(*(up + [800]))
        time.sleep(1.0)
    except Exception as exc:
        print("후퇴 실패: %s: %s — 수동으로 팔을 들어 올리세요." % (type(exc).__name__, exc))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trajectory", help="CSV from export_trajectory.py (radians)")
    p.add_argument("--rate", type=float, default=5.0, help="commands per second")
    p.add_argument("--sim-hz", type=float, default=60.0, help="rate the CSV was recorded at")
    p.add_argument("--max-step-deg", type=float, default=25.0)
    p.add_argument("--grip-deg", type=float, default=None,
                   help="hold the gripper at this servo angle instead of using the CSV "
                        "(recommended: the sim gripper joint oscillates)")
    p.add_argument("--servo-ms", type=int, default=300, help="Arm_Lib move duration per command")
    p.add_argument("--bus", type=int, default=8, help="I2C bus (right arm = 8)")
    p.add_argument("--addr", type=lambda s: int(s, 0), default=0x15,
                   help="I2C address (right arm = 0x15)")
    p.add_argument("--j5-offset", type=float, default=135.0,
                   help="degrees added to the wrist-roll servo command. MEASURED "
                        "on hardware 2026-08-31: with offset 0 the blade dragged "
                        "on its SIDE, and +135 is what puts the edge along the "
                        "cut. sim j5 and the servo are 135 deg out of step, so "
                        "sim's lower limit (-90 deg, where the policy sits 68% "
                        "of every episode) is actually servo 135 -- the middle "
                        "of the real travel, not a wall.")
    p.add_argument("--servo-margin", type=float, default=0.0,
                   help="keep commands this many degrees clear of each servo's "
                        "end stops. The sim clamps a joint at its limit for "
                        "free, so a policy trained without a limit penalty will "
                        "park against one -- the 08-31 trajectory held j3 at 0 "
                        "deg for 315 of 420 frames, which on hardware is the "
                        "servo leaning on its mechanical stop for seven "
                        "seconds. Use 10 for a no-blade pipeline test: the "
                        "motion shifts slightly where the clamp bites, but no "
                        "servo can stall.")
    p.add_argument("--calibration", default=None,
                   help="joint_calibration_bus<N>.json from calibrate_joints.py")
    p.add_argument("--approach-steps", type=int, default=8,
                   help="steps used to reach the first pose when the current "
                        "pose cannot be read back")
    p.add_argument("--retreat-deg", type=float, default=20.0,
                   help="degrees to straighten j2/j3 on abort, lifting the blade "
                        "out of the material before the arm is released")
    p.add_argument("--no-retreat", action="store_true",
                   help="leave the arm where it stopped on abort (not advised "
                        "with a blade fitted)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cal = load_calibration(args.calibration)

    rows = []
    with open(args.trajectory, newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].lstrip().startswith("#"):
                continue
            try:
                vals = [float(x) for x in row[:6]]
            except ValueError:
                continue  # header line
            if len(vals) == 6:
                rows.append(vals)
    if not rows:
        print("ERROR: no numeric rows found in", args.trajectory)
        return 1

    stride = max(1, int(round(args.sim_hz / args.rate)))
    frames = rows[::stride]
    cmds = [to_servo_deg(q, cal) for q in frames]
    if args.j5_offset:
        if cal.get("arm_joint5", False):
            print("\n*** 경고: 캘리브레이션이 arm_joint5 를 반전으로 표시하고 있는데")
            print("    --j5-offset 도 적용됩니다. 실측(2026-08-31)은 j5 가 반전이 아니고")
            print("    offset +135 라는 것이었습니다. 둘 다 적용하면 서보 명령이 범위를")
            print("    벗어나 잘립니다. 캘리브레이션에서 arm_joint5 를 false 로 바꾸세요.\n")
        lo5, hi5 = JOINTS[4][3], JOINTS[4][4]
        out_of_range = 0
        for c in cmds:
            v = c[4] + args.j5_offset
            if not (lo5 <= v <= hi5):
                out_of_range += 1
                v = min(max(v, lo5), hi5)
            c[4] = v
        print("j5-offset %+.0f deg applied" % args.j5_offset
              + ("" if not out_of_range
                 else "  (%d of %d clipped to %.0f..%.0f -- the real wrist cannot "
                      "reach that part of sim's range)" % (out_of_range, len(cmds), lo5, hi5)))
    if args.grip_deg is not None:
        for c in cmds:
            c[5] = args.grip_deg
    if args.servo_margin > 0:
        clamped = 0
        for c in cmds:
            for j, (_, _, _, a, b, _) in enumerate(JOINTS):
                v = min(max(c[j], a + args.servo_margin), b - args.servo_margin)
                if abs(v - c[j]) > 1e-9:
                    clamped += 1
                c[j] = v
        print("servo-margin %.0f deg: %d of %d commanded angles pulled off a stop"
              % (args.servo_margin, clamped, len(cmds) * 6))

    print("%d rows @ %.0f Hz -> %d commands @ %.1f Hz (%.1f s of motion)"
          % (len(rows), args.sim_hz, len(cmds), args.rate, len(cmds) / args.rate))

    # The servo needs --servo-ms to complete a move. Issuing the next command
    # sooner means it never arrives where it was told, and the arm tracks a path
    # nobody checked. This used to be advice in the abort text; it is a hard
    # error now, because the DEFAULT rate of 5 violates it (200ms < 300ms).
    period_ms = 1000.0 / args.rate
    if period_ms < args.servo_ms:
        print("ABORT: --rate %.1f gives %.0fms between commands, but --servo-ms is %d."
              % (args.rate, period_ms, args.servo_ms))
        print("  The servo cannot finish a move in time, so the arm never reaches")
        print("  the commanded poses and follows an unverified path.")
        print("  Use --rate %.1f or lower, or shorten --servo-ms if the hardware"
              % (1000.0 / args.servo_ms))
        print("  really does move faster.")
        return 1

    biggest = 0.0
    for i in range(1, len(cmds)):
        biggest = max(biggest, max(abs(cmds[i][j] - cmds[i - 1][j]) for j in range(6)))
    print("largest single-command joint change: %.1f deg" % biggest)
    if biggest > args.max_step_deg:
        print("ABORT: exceeds --max-step-deg (%.0f)." % args.max_step_deg)
        print("  RAISE --rate (more commands, smaller steps) -- lowering it makes")
        print("  the jumps bigger, since stride = sim_hz / rate.")
        print("  If the biggest jump is grip_joint, use --grip-deg instead: the sim")
        print("  gripper oscillates around its target and that motion must not be")
        print("  replayed onto a gripper that is clamping the blade.")
        print("  For this rig: --rate 3 --grip-deg 74.8 keeps steps under 21 deg")
        print("  while leaving 333ms per command against the servo's ~300ms.")
        return 1

    print("first command: " + " ".join("%6.1f" % v for v in cmds[0]))
    print("last  command: " + " ".join("%6.1f" % v for v in cmds[-1]))
    print("servo order: j1 j2 j3 j4 j5 grip (degrees)")

    if args.dry_run:
        print("\n--dry-run: the robot was not moved.")
        return 0

    print(f"\nAbout to MOVE THE REAL ROBOT on I2C bus {args.bus} (0x{args.addr:02X}).")
    print("Confirm: blade removed, skin removed, workspace clear, power reachable.")
    if input("type 'go' to continue: ").strip() != "go":
        print("aborted.")
        return 1

    arm = connect_arm(args.bus, args.addr)
    period = 1.0 / args.rate
    try:
        # The approach to the first pose was never checked: --max-step-deg only
        # compares consecutive commands, so the move from wherever the arm is
        # standing to cmds[0] was unbounded. On this trajectory that is j5
        # swinging 137 deg in one second. Read the current angles and walk there
        # in steps no larger than the same limit.
        here = read_pose(arm)
        if here is None:
            print("WARNING: could not read the current pose; approaching slowly.")
            steps = args.approach_steps
        else:
            gap = max(abs(cmds[0][j] - here[j]) for j in range(6))
            print("current pose: " + " ".join("%6.1f" % v for v in here))
            print("gap to first command: %.1f deg" % gap)
            steps = max(1, int(math.ceil(gap / args.max_step_deg)))
        if steps > 1:
            print("approaching in %d steps of <= %.0f deg" % (steps, args.max_step_deg))
        for k in range(1, steps + 1):
            t = k / steps
            via = ([here[j] + t * (cmds[0][j] - here[j]) for j in range(6)]
                   if here is not None else cmds[0])
            arm.Arm_serial_servo_write6(*(via + [args.servo_ms]))
            time.sleep(max(period, args.servo_ms / 1000.0))
        print("at start pose; holding 3 s...")
        time.sleep(3.0)
        for i, c in enumerate(cmds):
            arm.Arm_serial_servo_write6(*(c + [args.servo_ms]))
            if i % max(1, int(args.rate)) == 0:
                print("  %4d/%d  " % (i, len(cmds)) + " ".join("%6.1f" % v for v in c))
            time.sleep(period)
        print("replay finished.")
    except KeyboardInterrupt:
        print("\nCTRL+C — 재생 중단.")
        retreat(arm, args)
    except Exception as exc:
        # An I2C error used to fall straight through to `finally`, which closed
        # the bus and left the servos holding their last commanded pose. Mid-cut
        # that is the blade sitting in the material.
        print("\n재생 중 오류: %s: %s" % (type(exc).__name__, exc))
        retreat(arm, args)
        raise
    finally:
        try:
            arm.bus.close()
        except Exception:
            pass
        del arm
    return 0


if __name__ == "__main__":
    sys.exit(main())
