#!/usr/bin/env python3
"""Replay the opening trajectory on BOTH DOFBOTs. RUNS ON THE JETSON AGX.

Reads the CSV written by export_opening_trajectory.py: twelve servo angles per
line, already in DEGREES (right arm j1..j5, grip, then left arm). Nothing is
converted here, so no calibration file is needed. Standard library + Arm_Lib only.

SAFETY
  * Run --dry-run first and read the printed angles.
  * First live runs: skin patch removed, nothing between the arms, a hand on
    the power switch.
  * Start with ONE arm (--only right, then --only left). An arm moving alone
    cannot hit the other one, which stays where it is.
  * Then both arms with --stop-at, a few commands further each time.
  * Ctrl+C stops sending. The servos HOLD where they are; nothing moves on abort.

Usage:
    python3 replay_opening_on_agx.py opening_traj.csv --dry-run
    python3 replay_opening_on_agx.py opening_traj.csv --only right
    python3 replay_opening_on_agx.py opening_traj.csv --only left
    python3 replay_opening_on_agx.py opening_traj.csv --stop-at 12
    python3 replay_opening_on_agx.py opening_traj.csv
"""

import argparse
import csv
import math
import sys
import time

LIMITS = [(0.0, 180.0)] * 4 + [(0.0, 270.0), (0.0, 180.0)]
ARMS = ("right", "left")


def connect_arm(bus, addr):
    """Arm_Device() opens I2C bus 1 unconditionally; point it at the right one."""
    from Arm_Lib import Arm_Device
    import smbus

    arm = Arm_Device()
    try:
        arm.bus.close()
    except Exception:
        pass
    arm.bus = smbus.SMBus(bus)
    arm.addr = addr
    time.sleep(0.1)
    # Without this the servos stay limp and ignore position commands. Found on
    # the first hardware run (2026-10-08): nothing moved until torque was on.
    arm.Arm_serial_set_torque(1)
    time.sleep(0.1)
    return arm


def read_pose(arm):
    try:
        pose = [arm.Arm_serial_servo_read(i + 1) for i in range(6)]
    except Exception as exc:
        print("servo readback failed: %s: %s" % (type(exc).__name__, exc))
        return None
    if any(v is None for v in pose):
        return None
    return [float(v) for v in pose]


def fmt(c):
    return " ".join("%6.1f" % v for v in c)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trajectory", help="CSV from export_opening_trajectory.py (servo degrees)")
    p.add_argument("--period", type=float, default=0.25,
                   help="seconds the arm takes between consecutive CSV rows. 0.25 x 30 rows "
                        "= about 7.5 s for the whole motion (taught: 4 s)")
    p.add_argument("--hz", type=float, default=15.0,
                   help="commands actually sent per second. The CSV rows are interpolated up "
                        "to this rate so the servos are fed a steady stream instead of one "
                        "target every --period, which made them stop at every row")
    p.add_argument("--overlap", type=float, default=2.0,
                   help="servo move time as a multiple of the time between commands. Above 1 "
                        "the servo is still moving when the next command arrives, so it never "
                        "comes to rest mid-motion")
    p.add_argument("--no-smooth", action="store_true",
                   help="old behaviour: one command per CSV row, servo stops at each")
    p.add_argument("--servo-ms", type=int, default=300,
                   help="Arm_Lib move duration per command with --no-smooth, and for the "
                        "approach to READY")
    p.add_argument("--max-step-deg", type=float, default=25.0)
    p.add_argument("--only", choices=ARMS, default=None, help="move one arm; the other is not touched")
    p.add_argument("--stop-at", type=int, default=None,
                   help="play only the first N commands, then hold there")
    p.add_argument("--no-return", action="store_true",
                   help="do not offer to play the motion backwards to READY afterwards")
    p.add_argument("--right-bus", type=int, default=8)
    p.add_argument("--left-bus", type=int, default=1)
    p.add_argument("--addr", type=lambda s: int(s, 0), default=0x15)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cmds = []
    with open(args.trajectory, newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].lstrip().startswith("#"):
                continue
            try:
                vals = [float(x) for x in row[:12]]
            except ValueError:
                continue
            if len(vals) == 12:
                cmds.append((vals[:6], vals[6:]))
    if not cmds:
        print("ERROR: no 12-column rows found in", args.trajectory)
        return 1
    if args.stop_at is not None:
        cmds = cmds[:max(1, args.stop_at)]

    active = [a for a in range(2) if args.only in (None, ARMS[a])]

    for n, pair in enumerate(cmds):
        for a in active:
            for j, v in enumerate(pair[a]):
                lo, hi = LIMITS[j]
                if not (lo <= v <= hi):
                    print("ABORT: command %d, %s arm, servo %d = %.1f is outside %.0f..%.0f"
                          % (n, ARMS[a], j + 1, v, lo, hi))
                    return 1
    biggest = 0.0
    for i in range(1, len(cmds)):
        for a in active:
            biggest = max(biggest, max(abs(cmds[i][a][j] - cmds[i - 1][a][j]) for j in range(6)))
    if args.no_smooth and args.period * 1000.0 < args.servo_ms:
        print("ABORT: --period %.2fs is shorter than --servo-ms %d; the servos would never"
              % (args.period, args.servo_ms))
        print("  reach a pose before the next one is sent. Raise --period.")
        return 1
    sub = 1 if args.no_smooth else max(1, int(round(args.period * args.hz)))
    tick = args.period / sub
    move_ms = args.servo_ms if args.no_smooth else int(round(tick * 1000.0 * args.overlap))
    if not args.no_smooth:
        print("smooth: each row split into %d, one command every %.0f ms, servo move time %d ms"
              % (sub, tick * 1000.0, move_ms))

    print("%d commands, %.2f s apart = %.1f s of motion" % (len(cmds), args.period, len(cmds) * args.period))
    print("arms: " + ", ".join("%s (bus %d)" % (ARMS[a], (args.right_bus, args.left_bus)[a]) for a in active))
    print("largest single-command joint change: %.1f deg" % biggest)
    if biggest > args.max_step_deg:
        print("ABORT: exceeds --max-step-deg (%.0f). Re-export with a smaller --stride." % args.max_step_deg)
        return 1
    print("servo order: j1 j2 j3 j4 j5 grip (degrees)")
    for a in active:
        print("  %-5s first " % ARMS[a] + fmt(cmds[0][a]))
        print("  %-5s last  " % ARMS[a] + fmt(cmds[-1][a]))

    if args.dry_run:
        print("\n  n   right                                       left")
        for n, (r, l) in enumerate(cmds):
            print("%3d   %s    %s" % (n, fmt(r), fmt(l)))
        print("\n--dry-run: the robots were not moved.")
        return 0

    print("\nAbout to MOVE THE REAL ROBOT(S): " + ", ".join(ARMS[a] for a in active))
    print("Confirm: skin removed, nothing between the arms, power switch in reach.")
    if input("type 'go' to continue: ").strip() != "go":
        print("aborted.")
        return 1

    buses = (args.right_bus, args.left_bus)
    arms = {a: connect_arm(buses[a], args.addr) for a in active}

    def send(pair, ms):
        for a in active:
            arms[a].Arm_serial_servo_write6(*(pair[a] + [ms]))

    def play(seq, label):
        # Paced against the clock, not by sleeping a fixed amount, so the time
        # the I2C writes take does not stretch the motion or make it uneven.
        start = time.time()
        n = 0
        for i in range(len(seq)):
            steps = sub if i < len(seq) - 1 else 1
            for k in range(steps):
                f = k / float(steps)
                pair = seq[i] if k == 0 else tuple(
                    [seq[i][a][j] + f * (seq[i + 1][a][j] - seq[i][a][j]) for j in range(6)]
                    for a in range(2))
                send(pair, move_ms)
                n += 1
                wait = start + n * tick - time.time()
                if wait > 0:
                    time.sleep(wait)
            print("  %s %3d/%d  " % (label, i, len(seq) - 1)
                  + "   ".join("%s %s" % (ARMS[a][0].upper(), fmt(seq[i][a][:4])) for a in active))
        # let the last move finish before anything else is commanded
        time.sleep(move_ms / 1000.0)

    try:
        # Walk from wherever each arm is standing to the first command in steps
        # no larger than --max-step-deg; the trajectory check above only covers
        # consecutive commands, not this first move.
        here = {a: read_pose(arms[a]) for a in active}
        steps = 8
        if all(here[a] is not None for a in active):
            gap = max(abs(cmds[0][a][j] - here[a][j]) for a in active for j in range(6))
            for a in active:
                print("%-5s now  " % ARMS[a] + fmt(here[a]))
            print("gap to first command: %.1f deg" % gap)
            steps = max(1, int(math.ceil(gap / args.max_step_deg)))
        else:
            print("WARNING: could not read the current pose; approaching in %d steps." % steps)
        for k in range(1, steps + 1):
            t = k / float(steps)
            for a in active:
                via = ([here[a][j] + t * (cmds[0][a][j] - here[a][j]) for j in range(6)]
                       if here[a] is not None else cmds[0][a])
                arms[a].Arm_serial_servo_write6(*(via + [args.servo_ms]))
            time.sleep(max(args.period, args.servo_ms / 1000.0))
        print("at READY; holding 3 s...")
        time.sleep(3.0)

        play(cmds, "open")
        print("opening finished; the arms are holding the spread.")

        if not args.no_return:
            if input("press Enter to go back to READY the way it came (or type 'stay'): ").strip() != "stay":
                play(cmds[::-1], "back")
                print("back at READY.")
    except KeyboardInterrupt:
        print("\nCTRL+C -- stopped sending. The servos are holding their last pose.")
        print("Nothing was moved on abort. To back out, re-run with --stop-at at or")
        print("below the command number printed above and answer Enter to return.")
    finally:
        for a in active:
            try:
                arms[a].bus.close()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
