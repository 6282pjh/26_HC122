"""Opening motion for a detected incision line: lookup in the pre-computed library.

The opening policy (PPO, both arms) needs the simulator to run, so it is rolled
out there for every incision placement on a 2.5 mm / 2 deg grid
(scripts/export_opening_library.py) and each rollout is checked (opening
completed, arm-to-arm clearance, servo range, step size). The library holds the
resulting servo commands. Here the grid cell nearest to the detected line is
picked, so the motion is at most 1.25 mm / 1 deg off the line.
"""

import datetime
import math
import os

import numpy as np

from .safety import validate_trajectory
from .trajectory import I2C_BUS

DEFAULT_LIBRARY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights", "opening_library.npz")


class PlacementOutOfRange(ValueError):
    """The detected line is outside the range the policy was trained and checked on."""


class OpeningLibrary:
    def __init__(self, path=DEFAULT_LIBRARY):
        d = np.load(path, allow_pickle=False)
        self.path = path
        self.cmds, self.length, self.valid = d["cmds"], d["length"], d["valid"]
        self.yaws, self.dxs, self.dys = d["yaws"], d["dxs"], d["dys"]
        self.centre, self.ref_line = d["centre"], d["ref_line"]

    def placement(self, line, ref=None):
        """(shift_x, shift_y [m], turn [deg]) carrying the reference incision onto `line`.

        Lines are (cx, cy [m], heading [deg]) in the right arm's base frame.
        The task places the taught path as x' = c0 + R(turn)(x - c0) + shift
        (DofbotOpeningEnv._generate_path); this is the same transform solved for shift.
        """
        ref = self.ref_line if ref is None else ref
        turn = (line[2] - ref[2] + 90.0) % 180.0 - 90.0            # a line has no sense
        c, s = math.cos(math.radians(turn)), math.sin(math.radians(turn))
        rx, ry = ref[0] - self.centre[0], ref[1] - self.centre[1]
        return (line[0] - self.centre[0] - (c * rx - s * ry),
                line[1] - self.centre[1] - (s * rx + c * ry), turn)

    def _cell(self, place):
        idx = []
        for value, axis in ((place[2], self.yaws), (place[0], self.dxs), (place[1], self.dys)):
            step = axis[1] - axis[0]
            # half a cell of slack, so a line right on the boundary is not refused
            if value < axis[0] - step / 2 or value > axis[-1] + step / 2:
                return None
            idx.append(min(max(int(round((value - axis[0]) / step)), 0), len(axis) - 1))
        return tuple(idx)

    def limits(self):
        return {"x_mm": [float(self.dxs[0]) * 1000, float(self.dxs[-1]) * 1000],
                "y_mm": [float(self.dys[0]) * 1000, float(self.dys[-1]) * 1000],
                "turn_deg": [float(self.yaws[0]), float(self.yaws[-1])]}

    def commands(self, place):
        """Servo rows (12 values: right j1..j5, grip, left ...) and the grid placement used."""
        cell = self._cell(place)
        if cell is None or not bool(self.valid[cell]):
            raise PlacementOutOfRange(
                "shift (%+.1f, %+.1f) mm, turn %+.1f deg is outside the checked range %s"
                % (place[0] * 1000, place[1] * 1000, place[2], self.limits()))
        n = int(self.length[cell])
        used = (float(self.dxs[cell[1]]), float(self.dys[cell[2]]), float(self.yaws[cell[0]]))
        return self.cmds[cell][:n].astype(float).tolist(), used


def line_from_ends(start_mm, end_mm):
    """(cx, cy [m], heading [deg]) from two end points given in mm."""
    x1, y1, x2, y2 = start_mm[0] / 1000.0, start_mm[1] / 1000.0, end_mm[0] / 1000.0, end_mm[1] / 1000.0
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0, math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0)


def plan_opening(line_start_mm, line_end_mm, reference_ends_mm=None, library=None):
    """Both-arm opening trajectory for a detected incision line.

    line_start_mm, line_end_mm: (x, y) of the line's end points, mm, right-arm base frame.
    reference_ends_mm: ((x, y), (x, y)) of the incision the motion was taught on, seen by
        the SAME detector. Strongly recommended: only the difference between the two is
        used, so a constant offset of the detector's frame cancels. Without it the
        reference is the taught incision motion's forward kinematics (1-2 cm uncertain).
    Returns the trajectory dict, already validated.
    Raises PlacementOutOfRange or TrajectorySafetyError; nothing is returned in that case.
    """
    lib = library if isinstance(library, OpeningLibrary) else OpeningLibrary(library or DEFAULT_LIBRARY)
    line = line_from_ends(line_start_mm, line_end_mm)
    ref = line_from_ends(*reference_ends_mm) if reference_ends_mm else tuple(float(v) for v in lib.ref_line)
    place = lib.placement(line, ref)
    rows, used = lib.commands(place)
    traj = {
        "source": "rl_policy",
        "source_detail": "opening PPO policy (Dofbot-Opening-Direct-v0) rolled out in simulation; "
                         "pre-computed library %s, nearest grid cell" % os.path.basename(lib.path),
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "motion": "opening",
        "units": "servo_deg",
        "dt_s": 0.25,
        "joint_order": ["j1", "j2", "j3", "j4", "j5", "grip"],
        "line_robot_base_mm": {"start": [float(v) for v in line_start_mm[:2]],
                               "end": [float(v) for v in line_end_mm[:2]]},
        "reference": "detector" if reference_ends_mm else "taught-incision forward kinematics (default)",
        "placement": {"requested": {"x_mm": place[0] * 1000, "y_mm": place[1] * 1000, "turn_deg": place[2]},
                      "used": {"x_mm": used[0] * 1000, "y_mm": used[1] * 1000, "turn_deg": used[2]}},
        "sim_check": {"opening_completed": True, "min_arm_skeleton_gap_mm_required": 60.0,
                      "where": "scripts/export_opening_library.py"},
        "arms": {name: {"i2c_bus": I2C_BUS[name], "joints": [r[k:k + 6] for r in rows]}
                 for name, k in (("right", 0), ("left", 6))},
    }
    traj["safety_check"] = validate_trajectory(traj)
    return traj
