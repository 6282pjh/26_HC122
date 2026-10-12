"""Dry run: produce a trajectory and run the safety checks. No motor is moved.

    # opening, at a detected incision line (end points in mm, right-arm base frame)
    python -m rl_policy.deploy.dry_run opening --line 154 44 168 -29 --out results/dryrun_opening.json

    # opening, relative to a reference line seen by the same detector
    python -m rl_policy.deploy.dry_run opening --line 160 50 171 -22 --ref 154 44 168 -29 --out out.json

    # incision, from a simulator rollout exported by scripts/export_trajectory.py
    python -m rl_policy.deploy.dry_run incision --csv traj.csv --sim-check passed --out results/dryrun_incision.json
"""

import argparse
import sys

from . import (PlacementOutOfRange, TrajectorySafetyError, incision_trajectory_from_csv, plan_opening,
               save_trajectory, trajectory_to_csv)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="motion", required=True)
    o = sub.add_parser("opening")
    o.add_argument("--line", type=float, nargs=4, required=True, metavar=("X1", "Y1", "X2", "Y2"))
    o.add_argument("--ref", type=float, nargs=4, default=None, metavar=("X1", "Y1", "X2", "Y2"))
    o.add_argument("--library", default=None)
    i = sub.add_parser("incision")
    i.add_argument("--csv", required=True)
    i.add_argument("--sim-check", choices=("passed", "failed", "not-run"), default="not-run")
    i.add_argument("--rate", type=float, default=3.0)
    for s in (o, i):
        s.add_argument("--out", default=None, help="write the trajectory JSON here")
        s.add_argument("--csv-out", default=None, help="also write the CSV the AGX replay script reads (opening)")
    args = p.parse_args(argv)

    try:
        if args.motion == "opening":
            ref = (args.ref[:2], args.ref[2:]) if args.ref else None
            traj = plan_opening(args.line[:2], args.line[2:], ref, args.library)
        else:
            state = {"passed": True, "failed": False, "not-run": None}[args.sim_check]
            traj = incision_trajectory_from_csv(args.csv, rate_hz=args.rate, sim_check_passed=state)
    except (PlacementOutOfRange, TrajectorySafetyError, ValueError) as exc:
        print("REFUSED (%s): %s" % (type(exc).__name__, exc))
        print("No trajectory was produced. Nothing may be executed.")
        return 1

    print("motion: %s   source: %s" % (traj["motion"], traj["source"]))
    if "placement" in traj:
        r, u = traj["placement"]["requested"], traj["placement"]["used"]
        print("placement vs the taught incision: x %+.1f mm, y %+.1f mm, turn %+.1f deg  (reference: %s)"
              % (r["x_mm"], r["y_mm"], r["turn_deg"], traj["reference"]))
        print("library cell used:                x %+.1f mm, y %+.1f mm, turn %+.0f deg"
              % (u["x_mm"], u["y_mm"], u["turn_deg"]))
    s = traj["safety_check"]
    print("safety check: passed, %d rows, largest step %.1f deg, %d angles at the end-stop margin"
          % (s["rows"], s["max_step_deg"], s["angles_at_end_stop_margin"]))
    for name, arm in traj["arms"].items():
        print("  %-5s first %s" % (name, " ".join("%6.1f" % v for v in arm["joints"][0])))
        print("  %-5s last  %s" % (name, " ".join("%6.1f" % v for v in arm["joints"][-1])))
    if args.out:
        save_trajectory(traj, args.out)
        print("saved -> %s" % args.out)
    if args.csv_out and args.motion == "opening":
        trajectory_to_csv(traj, args.csv_out)
        print("saved -> %s" % args.csv_out)
    print("DRY RUN: no motor was moved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
