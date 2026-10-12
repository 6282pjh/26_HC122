"""
Opening (개복) V0: both arms spread the incision, following a hand-taught motion.

The reference is `motions/opening_dual.json`: five servo waypoints per arm,
taught on the real rig (bus 8 = right arm at the origin, bus 1 = left arm at
x = 0.370). The environment reads that file and runs forward kinematics on it at
start-up, so the path the policy is paid to follow IS the taught motion -- there
are no hand-copied coordinates here to drift out of date.

What the taught motion does (FK of the JSON, mm, right-arm base frame):

    READY_OPENING   both arms folded up, tips ~380 above the patch
    OPEN_APPROACH   swing out sideways (right y=+118, left y=-85), 150 high
    INSERT_ABOVE    come together over the incision, tips 62 apart, tools vertical
    INSERT_01       descend ~50, tips 87 apart
    INSERT_02       pull apart along x, tips 127 apart        <- the opening

The lessons from the incision task are applied from the start:
  * Progress is CONTINUOUS along the path and only extends from where it
    already is, so the arms have to travel it, not touch its corners.
  * The last two legs (insert, spread) are tight, need the tool vertical and
    need the tip to have been on the path the step before.
  * The path-progress reward dominates; dense shaping is deliberately small.
  * The spread has to be HELD. Reaching the end once is not an opening.
Both arms are policy-controlled here (8 actions), unlike incision V3.
"""

from __future__ import annotations

import os

from isaaclab.utils import configclass

from .dofbot_incision_env_cfg_v3 import DofbotIncisionEnvCfgV3, _incise_actuators

_REPO_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "..")
)


@configclass
class DofbotOpeningEnvCfg(DofbotIncisionEnvCfgV3):
    """Opening environment configuration. Scene and assets come from incision V3."""

    episode_length_s = 10.0
    # 4 arm joints per arm. arm_joint5 is not an action: the taught motion never
    # moves it (servo 89 / 88 at every waypoint), so it is pinned there.
    action_space = 8
    # per arm: q(4) dq(4) prev_action(4) target_rel(3) lookahead_rel(3)
    #          progress(1) tool_down(1) = 20; shared: spread_dir(2) hold(1)
    observation_space = 43

    # --- Reference motion ---------------------------------------------------
    motion_json = os.environ.get(
        "DOFBOT_OPENING_JSON", os.path.join(_REPO_DIR, "motions", "opening_dual.json")
    )
    # Servo degrees -> sim radians. Measured on bus 8 (joint_calibration_bus8.json
    # + find_j5_offset.py): joints 1-4 run inverted about 90, joint 5 is not
    # inverted and sits 135 deg off. Bus 1 has NOT been calibrated; it is the
    # same model of arm and is assumed to match. If calibrate_joints.py --bus 1
    # says otherwise, change it here and nothing else needs touching.
    servo_inverted = {
        "right": (True, True, True, True, False),
        "left": (True, True, True, True, False),
    }
    j5_servo_offset_deg = 135.0

    opening_arm_dof_names = ["arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4"]
    # Extra length of whatever the closed gripper holds, beyond the fingertips,
    # along the tool axis. 0 = the control point is the fingertip midpoint.
    # The JSON keeps both grippers at servo 180, which is what the knife arm
    # reads while clamping the scalpel, so both arms are probably holding a
    # retractor. Its length is not known; it only shifts the control point
    # along a vertical tool, so the taught joint motion is reproduced either way.
    opening_tool_length = 0.0

    # --- Rig height -----------------------------------------------------------
    # The taught motions put the tools ~30 mm lower than this scene's skin:
    # in incision_dual.json the blade drags at z = -33..-36 mm and the holder's
    # fingertips press at -33 mm; here the left tip ends at -24 mm. The table
    # top is at -10 mm with collision ON, so without this the left arm cannot
    # physically reach its taught pose. Whether the real surface is lower or
    # the servo zeros are a few degrees off is not known -- one tape-measure
    # reading (base plate to skin top) settles it. Until then the scene is
    # lowered to where the taught motion says the surface is.
    rig_z_offset = -0.030

    # --- Incision placement (the taught one, plus a little variation) --------
    # Offsets applied to every waypoint from `anchor_from_index` on, i.e. the
    # three that are positioned relative to the incision.
    anchor_from_index = 2
    line_shift_x_range = (-0.010, 0.010)
    line_shift_y_range = (-0.020, 0.020)
    line_yaw_range_deg = (-8.0, 8.0)
    reset_joint_noise = 0.05
    # Share of episodes that start part-way along the taught motion.
    rsi_prob = 0.5
    # Either tip further than this from where its path currently ends = reset.
    stray_limit = 0.12

    # --- Path following -------------------------------------------------------
    # Samples per taught leg when turning the joint-space motion into a path.
    path_substeps = 8
    # Legs from this index on are the insertion and the spread.
    tight_from_leg = 2
    free_corridor = 0.040       # how far off the path the tip may be, free-space legs
    tight_corridor = 0.008      # same, insert + spread
    # Furthest the progress may advance in one step. The taught motion takes 1 s
    # per leg (60 steps); the insert leg is ~50 mm, so ~1 mm/step. Allowing much
    # more lets the arm stab and yank instead of insert and spread.
    free_max_gap = 0.008
    tight_max_gap = 0.002
    path_samples = 8
    opening_vertical_cos_tol = 0.85   # taught: 0.92..1.00 on the tight legs
    # Share of the progress reward each leg carries. The approach is 85% of the
    # path LENGTH; paying by length would make the opening itself a rounding error.
    leg_weights = (1.0, 1.0, 2.0, 4.0)
    # "synchronized_dual_arm": an arm may not run further ahead than this
    # (in weighted progress) of the other.
    sync_max_lead = 0.10

    rew_path_progress_scale = 10000.0
    rew_offpath_penalty = 200.0
    rew_sync_penalty = 1.0
    rew_track_far_scale = 0.3
    track_far_eps = 0.15
    rew_track_fine_scale = 0.7
    track_fine_eps = 0.02
    rew_open_vertical_scale = 0.5

    # --- Holding the opening ---------------------------------------------------
    path_finish_eps = 0.002
    hold_tol = 0.006
    hold_steps = 60             # 1 s
    rew_hold_scale = 5.0
    rew_open_complete_bonus = 1000.0

    # --- Safety ----------------------------------------------------------------
    patch_thickness = 0.010
    rew_tip_dive_penalty_scale = 150.0
    tip_dive_cap = 0.02
    rew_link_floor_penalty_scale = 150.0
    link_floor_cap = 0.08
    crash_margin = 0.010
    min_tip_separation = 0.030  # taught minimum is 62 mm
    rew_tip_collision_penalty = 50.0

    # The taught end pose has arm_joint3 at servo 1 / 5 deg, i.e. 1-5 deg from
    # its stop. That is where the operator put it on real hardware, so the
    # incision task's joint-limit penalty would punish the reference itself.
    rew_joint_limit_penalty = 0.0

    def __post_init__(self):
        # Both arms move here, so the left one gets the same drive as the right.
        self.robot_retract_cfg.actuators = _incise_actuators()
        for prop in (self.skin_cfg, self.table_cfg):
            x, y, z = prop.init_state.pos
            prop.init_state.pos = (x, y, z + self.rig_z_offset)
        # The inherited ground plane sits at z = 0, which is the SKIN surface in
        # this scene, and it has collision. Incision never noticed (only the
        # virtual blade tip goes below 0), but here the left fingertips go to
        # -12 mm and were being stopped by it, and it hid the lowered patch and
        # table in renders. Put it under the support boxes (bottom at -127 mm).
        self.scene.ground.init_state.pos = (0.0, 0.0, -0.15)
