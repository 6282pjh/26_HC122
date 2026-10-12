"""Checks a trajectory must pass before it is sent to the servos.

These are the checks that need no simulator. The geometric ones (blade or links
below the table, arm-to-arm clearance, opening completed) are done where the
trajectory is generated, in scripts/check_trajectory_safety.py,
scripts/export_opening_trajectory.py and scripts/export_opening_library.py;
their outcome travels with the trajectory in its "sim_check" field.
"""

from .servo import SERVO_LIMITS

MAX_STEP_DEG = 25.0        # largest change of one servo between consecutive rows
END_STOP_MARGIN_DEG = 1.0  # joints 1-4 must stay this far from a servo end stop


class TrajectorySafetyError(ValueError):
    """The trajectory failed a safety check and must not be executed."""


def validate_trajectory(traj, max_step_deg=MAX_STEP_DEG, end_stop_margin_deg=END_STOP_MARGIN_DEG):
    """Return a report dict, or raise TrajectorySafetyError.

    `traj` is the dict described in INTERFACE.md section 7.
    """
    if traj.get("units") != "servo_deg":
        raise TrajectorySafetyError("units must be 'servo_deg', got %r" % traj.get("units"))
    arms = traj.get("arms") or {}
    if not arms:
        raise TrajectorySafetyError("trajectory has no arms")
    lengths = {name: len(arm["joints"]) for name, arm in arms.items()}
    if len(set(lengths.values())) != 1:
        raise TrajectorySafetyError("arms have different numbers of rows: %s" % lengths)

    biggest, at_margin = 0.0, 0
    for name, arm in arms.items():
        rows = arm["joints"]
        if not rows:
            raise TrajectorySafetyError("%s arm has no rows" % name)
        for n, row in enumerate(rows):
            if len(row) != 6:
                raise TrajectorySafetyError("%s arm, row %d: expected 6 servo angles" % (name, n))
            for j, value in enumerate(row):
                lo, hi = SERVO_LIMITS[j]
                if j < 4:
                    lo, hi = lo + end_stop_margin_deg, hi - end_stop_margin_deg
                if not (lo - 1e-6 <= value <= hi + 1e-6):
                    raise TrajectorySafetyError(
                        "%s arm, row %d, servo %d = %.1f deg is outside %.1f..%.1f"
                        % (name, n, j + 1, value, lo, hi))
                if j < 4 and (value <= lo + 0.05 or value >= hi - 0.05):
                    at_margin += 1
            if n:
                step = max(abs(row[j] - rows[n - 1][j]) for j in range(6))
                biggest = max(biggest, step)
                if step > max_step_deg:
                    raise TrajectorySafetyError(
                        "%s arm, row %d: a servo jumps %.1f deg (limit %.1f)" % (name, n, step, max_step_deg))
    return {
        "passed": True,
        "rows": next(iter(lengths.values())),
        "max_step_deg": round(biggest, 2),
        "angles_at_end_stop_margin": at_margin,
        "max_step_limit_deg": max_step_deg,
        "end_stop_margin_deg": end_stop_margin_deg,
    }
