# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Incision V3: clean redesign of the skin-incision task (knife + static holder)

Lessons from the five V2 training runs are baked in:

1. POSITION-TARGET control (was: velocity). The policy outputs joint-target
   DELTAS integrated into a persistent target buffer tracked by a stiff PD.
   This produces inherently smooth motion and matches the real DOFBOT servo
   interface (Arm_serial_servo_write6 takes position commands), removing a
   sim2real mismatch.
2. SMOOTHNESS is rewarded: an action-rate penalty plus the previous action in
   the observation. V2 had no incentive or information for temporal smoothness
   and its motion was jittery.
3. STD CONTROL: V2's policy std always pinned at its ceiling (weak precision
   gradient vs entropy). V3 caps log-std at 0.3, starts at -1.0, and lowers
   the entropy bonus, so the policy can actually converge to precise actions.
4. PRECISION GRADIENT: shaping kernels are sharper near the target and the
   waypoint "cut" test is slightly relaxed so the sparse bonuses actually fire
   (a bonus that never fires teaches nothing).
5. The second arm (holder) is a STATIC prop in a bent skin-holding pose, not
   policy-controlled (two-arm coordination stalled learning in V2).
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


def _finger_actuator() -> ImplicitActuatorCfg:
    # Hold the passive gripper-finger linkage rigid (it has no mimic joint and
    # flops freely if left unactuated).
    return ImplicitActuatorCfg(
        joint_names_expr=["llink_joint[1-3]", "rlink_joint[2-3]"],
        stiffness=30.0,
        damping=5.0,
        effort_limit_sim=20.0,
    )


def _incise_actuators() -> dict:
    # Position-control PD for the knife arm (policy moves the position target).
    return {
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint[1-5]"],
            stiffness=100.0,
            damping=20.0,
            effort_limit_sim=80.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["grip_joint"],
            stiffness=40.0,
            damping=20.0,
            effort_limit_sim=50.0,
        ),
        "fingers": _finger_actuator(),
    }


# Real-rig props, measured from the user's scene script. Defined at module level:
# a helper inside the configclass body is treated as a config attribute and breaks
# config serialisation ("has no attribute '_prop'") when the task is launched.
_BOX_SIZE = (0.260, 0.374, 0.157)     # support box; top face = robot base height (+30mm)
_TABLE_SIZE = (0.160, 0.240, 0.128)   # surgical table; top face = skin patch bottom (-10mm)


def _rig_prop(path: str, size, pos, color) -> RigidObjectCfg:
    """Static prop with collision ON, so the arm cannot plan through it."""
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/{path}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


def _retract_actuators() -> dict:
    # Static holder: stiff position control pinned at its bent holding pose.
    return {
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint[1-5]"],
            stiffness=60.0,
            damping=15.0,
            effort_limit_sim=80.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["grip_joint"],
            stiffness=40.0,
            damping=20.0,
            effort_limit_sim=50.0,
        ),
        "fingers": _finger_actuator(),
    }


@configclass
class DofbotIncisionSceneCfgV3(InteractiveSceneCfg):
    """Scene definition with a shared ground plane."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            color=(0.4, 0.4, 0.4), size=(100.0, 100.0), visible=True
        ),
        collision_group=-1,
    )


@configclass
class DofbotIncisionEnvCfgV3(DirectRLEnvCfg):
    """Incision V3 environment configuration."""

    # Env
    decimation = 2
    episode_length_s = 15.0

    # Spaces
    # obs (29): q(5), dq(5), prev_action(5), ee_pos(3), cur_target_rel(3),
    #           next_target_rel(3), progress(1), tool_downward_cos(1),
    #           line_dir(2), blade_align_cos(1)
    action_space = 5
    observation_space = 29
    state_space = 0

    # Simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # Robot USD path (same asset as V2)
    _ASSETS_DIR = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "assets")
    )
    _LOCAL_USD = os.path.join(_ASSETS_DIR, "dofbot.usd")
    _DEFAULT_USD = os.path.join(_ASSETS_DIR, "dofbot.usda")
    # Yahboom models base_link.STL with the plate's long section extending the
    # same way the arm reaches. On the real robot that section sits BEHIND the
    # turret (it carries the Jetson/expansion board) and the arm works out past
    # the turret end. This copy has base_link's visual+collision yawed 180 deg
    # about the turret axis to match; nothing kinematic changes, so policies
    # trained on the unflipped asset stay valid.
    _PLATE_OUT_USD = os.path.join(_ASSETS_DIR, "dofbot_plate_out.usd")
    _ENV_OVERRIDE = os.environ.get("DOFBOT_ROBOT_USD")

    if _ENV_OVERRIDE and os.path.exists(_ENV_OVERRIDE):
        robot_usd_path = _ENV_OVERRIDE
        print(f"[Incision-V3] Using robot asset from DOFBOT_ROBOT_USD: '{robot_usd_path}'")
    elif os.path.exists(_PLATE_OUT_USD):
        robot_usd_path = _PLATE_OUT_USD
        print(f"[Incision-V3] Using plate-flipped robot asset: '{robot_usd_path}'")
    elif os.path.exists(_DEFAULT_USD):
        robot_usd_path = _DEFAULT_USD
        print(f"[Incision-V3] Using local ASCII robot asset: '{robot_usd_path}'")
    else:
        robot_usd_path = _LOCAL_USD
        print(f"[Incision-V3] Using local binary robot asset: '{robot_usd_path}'")

    # Articulation root (see dofbot_pickplace_env_cfg_v3.py for why "/base_link")
    articulation_root_prim_path = "/base_link"

    # Consistent CLOSED-gripper pose for the passive finger linkage. The real
    # gripper is a mechanical linkage (URDF mimic tags, dropped by the USD
    # importer): each finger joint follows grip_joint with multiplier +/-1.
    # Holding the fingers at 0 while grip_joint=-0.9 left the linkage visually
    # broken ("너덜너덜"); these angles are the mimic-consistent closed pose.
    _closed_finger_joint_pos = {
        "llink_joint1": 0.9,   # -1 * grip(-0.9)
        "llink_joint2": -0.9,  # +1 * grip(-0.9)
        "llink_joint3": 0.9,   # -1 * grip(-0.9)
        "rlink_joint2": 0.9,   # -1 * grip(-0.9)
        "rlink_joint3": -0.9,  # +1 * grip(-0.9)
    }

    # Knife arm (policy-controlled).
    # Base yawed exactly +pi/2 so the arm FACES the patch at j1=0: the URDF
    # mounts the arm yawed ~1.5 rad on base_link, so with an unrotated base the
    # arm had to work at j1~+1.29 — 16 deg from its +90 deg limit, with almost
    # no yaw authority toward one side (measured face_yaw(j1=0) = -87 deg).
    # Centering j1 gives symmetric +/-90 deg yaw range over the incision line
    # and, on the real robot, keeps servo 1 near mid-range instead of its stop.
    # pi/2 (not the 1.35 first tried) also lays the base plate exactly parallel
    # to the knife-holder axis, so the two robots square up facing each other.
    robot_incise_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/RobotIncise",
        spawn=sim_utils.UsdFileCfg(usd_path=robot_usd_path),
        init_state=ArticulationCfg.InitialStateCfg(
            # 받침 박스 윗면 = 실측 World Z +30mm. 이전에는 로봇을 지면에
            # 두고 피부를 -35mm에 둬서 피부가 지면 아래 파묻혀 있었다.
            # 상대 기하는 같지만 실물 배치와 어긋나 렌더가 틀렸다.
            pos=(0.0, 0.0, 0.030),
            # 실물 배치(2026-08-18 확정): 두 판은 평행하게 두되, 오른팔은 반시계
            # 90도 / 왼팔은 시계 90도로 서로 반대로 돌려 엇갈리게 놓는다. 로봇이
            # 피부 양쪽에 있으므로 이러면 두 팔 모두 피부를 향한다.
            # 같은 방향으로 나란히 두면 작업면이 팔의 옆 90도에 놓여 j1이 한계에
            # 붙고 작업면의 49%만 도달 가능해진다(실측).
            rot=(0.7071068, 0.0, 0.0, 0.7071068),  # yaw +pi/2 — 반시계 90도
            joint_pos={
                "arm_joint.*": 0.0,
                "grip_joint": -0.9,  # closed (holding the tool)
                **_closed_finger_joint_pos,
            },
        ),
        actuators=_incise_actuators(),
        articulation_root_prim_path=articulation_root_prim_path,
    )

    # Static holder arm on the opposite side, facing the patch centre-line,
    # frozen in a bent skin-holding pose (tool tip ~(0.28, 0.0, 0.04); the
    # incision waypoints stay at x<=0.25 so it never blocks the knife).
    # Candidate holding poses, measured 2026-08-31 by sweeping the holder's
    # first four joints and binning by where a fingertip lands on the patch.
    # Each entry is (press_x, press_y, j1, j2, j3, j4) with the press point in
    # the knife arm's frame; every one of them actually presses (fingertip
    # 1.9-6.3 mm below the skin surface).
    #
    # The pose is chosen PER EPISODE, farthest from that episode's incision
    # line, instead of being fixed. A fixed pose cost 16 points of completion
    # (56% vs 72% with the holder parked 5 m away) because it sat on the path of
    # strokes at headings above 105 deg. On the real rig the same rule applies:
    # the camera reports where the line is, so the holder can be told where to
    # press rather than always pressing the same spot.
    # FIXED holding pose, pressing the patch's FAR EDGE (user, 2026-08-31).
    #
    # Measured after full settling: fingertip lands at (245.6, +10.3, -6.2) mm,
    # i.e. on the patch and 6.2 mm into it, with ZERO links inside the incision
    # zone. Verified the settled numbers, not the search-time ones -- the arm
    # droops 5-7 mm further out over a full episode than it does after the six
    # settle steps the pose search used.
    #
    # Per-episode selection among candidates was tried and abandoned. Three
    # variants (fixed retune, pick-by-fingertip, pick-by-footprint) all scored
    # 55-56% against 72% with the holder removed, because the holder's base sits
    # at x=0.370 and the patch ends at x=0.265, so its arm must cross x
    # 0.24-0.37 whatever the joints. Worse, "furthest from the line" always
    # picked the extreme-y poses, whose fingertips settle OFF the patch
    # entirely (y -128.8 / +122.0 against a patch edge at -125/+115) -- the
    # holder was pressing thin air. Only moving the base would fix the
    # obstruction, which is a physical change to the rig.

    retract_hold_joint_pos = {
        # Same +pi/2 base-yaw correction as the knife arm (see above). j1 drops
        # by the same pi/2 so the world-frame holding pose is unchanged, but j1
        # now sits near 0 instead of pinned at the +1.5708 limit.
        # Retuned 2026-08-25 by search over 625 candidate poses. The previous
        # angles (j1 -0.071 / j3 1.3 / j4 1.2) were the worst of both worlds:
        # the fingertip sat at z=+2.0mm, i.e. 2mm ABOVE the skin and pressing
        # nothing, while the arm left only 5.4mm of horizontal clearance to the
        # incision zone. Parking the holder 5m away raised completion 41% -> 64%
        # on an unchanged policy, so that clearance was worth 23 points.
        #
        # These press the patch for real (fingertip z = -1.1mm) at its far edge,
        # outside the incision zone, with 23.9mm of clearance -- 4.4x the old.
        "arm_joint1": 0.079,
        "arm_joint2": 0.950,
        "arm_joint3": 1.350,
        "arm_joint4": 0.700,
        "arm_joint5": 0.0,
        "grip_joint": -0.9,
        **_closed_finger_joint_pos,
    }
    robot_retract_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/RobotRetract",
        spawn=sim_utils.UsdFileCfg(usd_path=robot_usd_path),
        init_state=ArticulationCfg.InitialStateCfg(
            # Base pushed back 2cm (0.40 -> 0.42) so the closed gripper pinches
            # the skin patch at its FAR EDGE (x~0.30) instead of poking through
            # the surface mid-patch (the patch has collisions disabled so the
            # knife can enter it; the holder must therefore avoid overlapping
            # the surface visually).
            pos=(0.370, 0.0, 0.030),
            rot=(-0.7071068, 0.0, 0.0, 0.7071068),  # yaw (pi + pi/2) — 반대로 90도
            joint_pos=retract_hold_joint_pos,
        ),
        actuators=_retract_actuators(),
        articulation_root_prim_path=articulation_root_prim_path,
    )

    # Skin patch (static proxy surface)
    skin_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Skin",
        spawn=sim_utils.CuboidCfg(
            size=(0.16, 0.24, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.85, 0.55, 0.45), metallic=0.0
            ),
        ),
        # 피부 표면이 World Z=0 (수술대 위). 두께 10mm이므로 중심 -5mm.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.185, 0.0, -0.005)),
    )

    # ---- 실물 리그 재현 (사용자 세트장 스크립트의 실측 치수) ----------------
    # 지금까지는 치수만 반영하고 물체는 생략했다. 정책이 관측하지 않는 소품이라
    # 학습에는 불필요하지만, 그 결과 sim이 "팔이 자기 받침 박스나 수술대를 친다"는
    # 것을 알 수 없었다. 실물 좌표를 칼팔 기준으로 변환해 충돌체로 넣는다.
    #   변환: env_x = (265 - worldX)mm,  env_y = (115 - worldY)mm
    knife_box_cfg: RigidObjectCfg = _rig_prop(
        "KnifeBox", _BOX_SIZE, (-0.055, -0.005, -0.0485), (0.65, 0.72, 0.72)
    )
    holder_box_cfg: RigidObjectCfg = _rig_prop(
        "HolderBox", _BOX_SIZE, (0.425, -0.005, -0.0485), (0.65, 0.72, 0.72)
    )
    table_cfg: RigidObjectCfg = _rig_prop(
        "SurgicalTable", _TABLE_SIZE, (0.185, -0.005, -0.074), (0.72, 0.72, 0.72)
    )

    # Scene
    scene: InteractiveSceneCfg = DofbotIncisionSceneCfgV3(num_envs=16, env_spacing=2.5)

    # Joint / link names
    arm_dof_names = [
        "arm_joint1",
        "arm_joint2",
        "arm_joint3",
        "arm_joint4",
        "arm_joint5",
    ]
    gripper_dof_name = ["grip_joint"]
    gripper_closed_target = -0.9
    ee_link_name = "rlink2"        # gripper reference (fingers held rigid)
    # Blade clamped in the gripper, measured along the tool approach axis from
    # rlink2 to the cutting tip. The policy tracks the BLADE TIP, so at cut
    # depth (z=0.045) the gripper sits at z=0.075 — clear of the 0.050 patch
    # surface instead of passing through it.
    # Distance from the rlink2 link ORIGIN to the blade edge, along the tool
    # axis. MEASURED, not assumed: rlink2.STL runs 50.51 mm along the link's
    # local +x, and that direction is the tool axis, so the fingertip sits
    # +50.50 mm from the origin (confirmed in-sim against the articulation
    # poses). The user measured 10 mm from fingertip to the blade in the block
    # clamped in the gripper. 50.5 + 10 = 60.5.
    #
    # The old 0.03 put the sim's "blade tip" 20.5 mm BEHIND the fingertip, i.e.
    # inside the gripper body. Driving that point to cut depth pushed the real
    # fingers 20.5 mm further down -- through the 10 mm patch and into the
    # table. It read as safe only because the floor guard watches link ORIGINS,
    # which sit 50.5 mm above the fingertips.
    blade_length = 0.0605
    # rlink2.STL / llink2.STL extent along the link's local +x, used to guard
    # the actual fingertips rather than the link origins.
    finger_length = 0.0505
    wrist_link_name = "arm_link5"  # its local +z is the tool approach axis

    # Position-delta action: target += action * joint_delta_scale per env step.
    # 0.04 rad/step at 60 Hz -> max ~2.4 rad/s, comparable to V2's reach speed
    # but tracked by a PD, so the resulting motion is smooth.
    joint_delta_scale = 0.04

    # Skin surface top z (patch center z + half thickness)
    # 실측: 로봇 베이스 평면이 받침 박스 윗면(World Z=+30mm)이고 피부 표면이
    # World Z=0 이므로, 로봇 기준으로 피부는 30mm '아래'. sim은 +50mm였음(80mm 차이).
    skin_top_z = 0.0

    # Incision line generation: STRAIGHT lines (the real task cuts straight
    # lines drawn on the skin). Random position + random direction + random
    # length every episode, sampled inside the region below; lengths shrink
    # automatically when the angle would push an endpoint out of the region.
    # Cut into fewer, wider-spaced points. Waypoint spacing is length/(n-1) and
    # must stay clear of waypoint_xy_tol (5 mm), or consecutive points sit inside
    # each other's tolerance and "tracing" degenerates into standing still. With
    # 5 points over 40-70 mm the spacing is 10.0-17.5 mm, comfortably clear.
    # Note this does NOT change observation_space: the policy only ever sees the
    # current and next target, not the whole list.
    num_waypoints = 5
    # 작업면 실측 범위는 정면 105~265mm / 좌우 -115~+125mm. 여유를 두고
    # 작업중심(185mm)을 기준으로 잡음.
    # Region the whole line must fit in (robot-local), sized to the 160x240 board
    # as it actually sits: the 160 mm side runs between the robots. Briefly
    # trimmed to y_hi 0.07 so the box would also fit the board turned 90 deg,
    # then reverted — the support boxes leave a 220 mm gap between their inner
    # faces and the board is 240 mm on its long side, so it physically cannot be
    # rotated into that direction. Note this also limits the 0/90 jig indexing
    # that the 90-deg-wide heading band relies on; verify on the rig which line
    # orientations are actually reachable before relying on it.
    incision_x_range = (0.135, 0.235)
    incision_y_range = (-0.08, 0.08)
    line_length_range = (0.04, 0.07)  # 4–7 cm (was 5–12; shortened per rig)
    # Travel-direction band, in degrees. Pulled out of _generate_waypoints so an
    # eval can widen it without editing code -- the whole question of whether the
    # upper band is blocked by the holder or by the wrist is answered by running
    # the same policy over a wider band.
    incision_heading_range = (60.0, 105.0)
    incision_depth = 0.005  # 5mm below skin surface (real depth set by skin height)

    # --- Reward: vertical incision, precise and smooth ------------------------
    # Dense shaping is bounded positive (tanh kernels), TWO-SCALE: a coarse
    # kernel provides gradient from far away (a single sharp kernel saturates
    # beyond ~6cm and would leave early exploration with no approach signal),
    # while the sharp kernel provides real gradient for the last centimetre.
    rew_incise_far_scale = 0.5
    incise_far_eps = 0.15
    rew_incise_xy_scale = 1.0
    incise_xy_eps = 0.02
    incise_xy_align_tol = 0.02     # xy error under which the tool is "over" the point
    # Coarse depth kernel, mirroring the coarse xy kernel above. Without it the
    # sharp 8 mm kernel is saturated at the standoff the arm starts from and the
    # policy has no gradient telling it to descend at all.
    rew_incise_depth_far_scale = 1.0
    incise_depth_far_eps = 0.05
    rew_incise_depth_scale = 1.5
    incise_depth_eps = 0.008
    # Keep the tool vertical (wrist local +z pointing world-down)
    # NOTE (v3 run 1): the policy reached xy 3mm / z 1mm on the first waypoint
    # with downward cos ~0.79 (~38 deg) but the cut test demanded 0.85, so no
    # waypoint ever registered and the line was never traced. ~38 deg appears
    # to be near the 5-DOF kinematic limit at the line's reach, so the GATE is
    # relaxed to 0.6 while the DENSE weight is raised — the policy still gets
    # pushed toward the most vertical posture the kinematics allow.
    rew_vertical_scale = 0.5
    vertical_cos_tol = 0.6         # ~53 deg gate; dense reward drives true posture

    # Blade orientation: the real tool is a directional blade mounted on the
    # gripper — its edge must FACE the travel direction to cut. The blade edge
    # is defined along the wrist (arm_link5) local +x axis; the policy aligns
    # it with the line direction using arm_joint5. Gate kept loose (0.5 ~ 60
    # deg) after the vertical-gate lesson: dense reward shapes the posture,
    # the gate only rejects grossly wrong orientations.
    rew_blade_align_scale = 0.5
    # |cos| of the blade edge against the line. 0.5 meant the edge could sit
    # 60 deg off and still count as a cut, which is a tear, not an incision --
    # the gate was open 836-888 steps of 898, i.e. essentially always. 0.97 is
    # ~14 deg. Unsigned, so both 180-deg-apart blade poses satisfy it.
    # CURRICULUM STEP. 0.97 (~14 deg) applied in one jump made blade alignment
    # the binding gate: measured 17-30 open steps of 898 at headings 120-150,
    # while depth (just fixed) reached 752-803 there. The two trade off against
    # each other on a 5-DOF arm -- the wrist roll that squares the blade costs
    # the posture that reaches depth -- yet best-blade is 1.00 in EVERY bucket,
    # so the pose is reachable and the problem is holding it. 0.90 (~26 deg)
    # first, then tighten back toward 0.97 once completion is non-trivial.
    blade_align_tol = 0.90

    # Waypoint "cut" test. These were loose enough that the policy could satisfy
    # them without performing the motion, so both are now smaller than the
    # physical feature they are supposed to hit:
    #
    #   xy 25mm vs waypoint spacing 8.2-16.9mm -> a stationary blade sat inside
    #     3 waypoints at once (max 6), and 19% of "cuts" happened with under 1mm
    #     of travel. 5mm is below half the minimum spacing, so at most two
    #     waypoints can qualify together.
    #   z 10mm vs a 10mm-thick patch (0.040-0.050) -> the window spanned
    #     0.035-0.055, i.e. 5mm clear above the skin (not cutting) through to 5mm
    #     below the pad (through it into the tray) both counted as a cut. On the
    #     08-12 policy all 8 waypoints were passed at steps 105-128, entirely
    #     below the pad, while the blade only settled at cut depth from step 134.
    #     3mm keeps the blade inside the patch (0.042-0.048).
    waypoint_xy_tol = 0.005
    waypoint_z_tol = 0.003

    # Sparse event bonuses
    # --- Continuous cut ------------------------------------------------------
    # The task is a stroke. The blade counts as cutting while it is within
    # cut_lateral_tol of the LINE (not of a target point -- while stroking, the
    # tip legitimately runs ahead of where the cut currently ends), at depth,
    # vertical and squared up. The cut extends only from where it already ends,
    # by at most cut_max_gap per step, so the policy cannot satisfy the
    # conditions at the far end and claim the whole line.
    cut_lateral_tol = 0.004    # 4 mm off the line and the blade is not on it
    # Posture rewards (vertical, blade alignment) are multiplied by
    # 1 - tanh(lateral/this), so they pay near the line and fade away from it.
    posture_gate_eps = 0.03
    # Furthest the tip may extend the cut in one step. 8 mm was far too loose:
    # a 40-70 mm line at 60 Hz needs only 1-1.75 mm/step, so 8 mm left room to
    # touch, lift, skip ahead and touch again -- a dotted line that scored as a
    # cut. 3 mm still allows a brisk stroke without allowing the skip.
    cut_max_gap = 0.003
    cut_lookahead = 0.015      # how far ahead the "next target" observation looks
    cut_finish_eps = 0.003     # this close to the far end counts as a full cut
    # Paid per millimetre cut, normalised so a whole stroke is worth this much
    # whatever the sampled length.
    #
    # 3000 was sized against the wrong baseline -- against a FULL cut rather
    # than the cut actually achieved. Measured on 2026-08-24_04-15-36: episode
    # reward 8400, of which cutting contributed 3000 x 11% = 330, i.e. 4%. The
    # other 96% came from hovering near the line with good posture, so that is
    # what the policy learned. Dense scales are now roughly a third of what they
    # were and posture is gated on line proximity, while a full stroke pays
    # 10000 + 1000: hovering earns on the order of 2000 an episode, cutting
    # 11000. Cutting has to be the dominant term or it is not the task.
    rew_cut_progress_scale = 10000.0
    # Charged once each time contact breaks mid-stroke. Roughly the value of
    # 5 mm of cut, so re-approaching is worse than simply staying down.
    rew_cut_lift_penalty = 200.0
    rew_waypoint_bonus = 10.0  # unused since the continuous-cut rewrite
    rew_completion_bonus = 1000.0

    # Smoothness: penalise action RATE (squared change vs previous action).
    # This is the main anti-jitter term; the previous action is also part of
    # the observation so the policy can act on it.
    # Keep the arm above the work surface. Nothing in the scene stops it going
    # under: the ground plane is in collision_group -1, the skin patch has
    # collisions disabled, and the blade is a numeric offset with no geometry.
    # A trajectory exported from the previous policy put arm_link5 at z=-0.041
    # during the approach — 4cm below the table, which is a crash on hardware.
    # Only the blade may go below the skin; every link must stay above this.
    # Measured reference: during a legitimate cut the lowest link sits at 0.051,
    # so 0.02 leaves margin without penalising the cutting pose itself.
    # 피부 표면 기준으로 정의한다. 실물은 피부가 로봇 베이스보다 30mm 아래라,
    # 이 값을 베이스 평면(+0.02)에 두면 팔이 피부에 접근하는 것 자체가 위반이 되어
    # 정상 절개를 금지해 버린다. 물리적으로 옳은 제약은 "칼날 외에는 피부 표면
    # 아래로 내려가지 않는다"이므로 표면과 같은 높이로 잡는다. 페널티가 침범
    # 깊이에 비례하므로 1mm 스침은 0.15/step로 무시할 만하고, 20mm 다이빙은
    # 3/step로 확실히 아프다.
    workspace_floor_z = skin_top_z
    # Bounded so a violation cannot swamp the dense terms and kill early
    # learning, the way the run-1 distance penalty did.
    # Scale 40 was measured to be far too weak: after a full retrain the penalty
    # came to ~120 of an 8667 episode reward (1.4%), so the policy simply paid it
    # and still dipped the wrist to -0.022m. 150 puts a typical violation near
    # 10% of episode reward, which it cannot ignore.
    rew_floor_penalty_scale = 150.0
    floor_penalty_cap = 0.08
    # Catastrophe backstop only. Terminating at z<0 was measured to be far too
    # harsh: 63 of 64 episodes died at step ~77, before the arm ever reached the
    # line (waypoints are normally passed around step 105-128), so the policy
    # never got to practise cutting at all — the same bootstrap collapse that
    # killed the 07-29 and 08-01 scratch runs. The dense penalty above does the
    # shaping; this only stops the policy exploring deep dives.
    #
    # Pinned to the surgical table's top face, which is also the skin patch's
    # underside (skin spans -10..0 mm). The table now has collision enabled, so
    # a knife-arm link cannot physically get below it; leaving the threshold at
    # -20 mm put it inside solid geometry, where PhysX blocks the arm before the
    # backstop can ever fire. At the table top the two agree: touching the table
    # ends the episode.
    floor_crash_z = skin_top_z - 0.010

    # --- Blade lower bound --------------------------------------------------
    # The blade tip is a computed point, so the table never stops it. Without a
    # lower bound the policy dove to -15..-23mm against a -5mm target at
    # headings 135-150 and scored 0% cut there, while rollouts that stayed in
    # -3..-11mm cut 30-81%. The patch bottom is the table's top face, which is
    # where a real blade stops.
    blade_floor_z = skin_top_z - 0.010
    rew_blade_dive_penalty_scale = 150.0   # same order as the finger floor penalty
    blade_dive_cap = 0.02                  # bounded, so it cannot swamp learning
    blade_crash_z = blade_floor_z - 0.010  # 10mm into the table ends the episode

    # Keep the arm off its joint stops. Sim clamps for free; a real serial servo
    # held at the end of its range stalls against the mechanical stop and heats,
    # with no torque feedback. The first exported trajectory parked arm_joint3
    # there for 315 of 420 frames. Penalty ramps from 0 to full over the last
    # 8% of each joint's travel, summed over the five arm joints.
    # Wrist-roll range, overridden to match the REAL servo. MEASURED 2026-08-31:
    # at the mid-cut pose the blade squares up at servo 170.8, which is the
    # non-inverted mapping (35.8) plus 135 -- sim's j5 and the servo are 135 deg
    # out of step. Feeding that offset back through the servo's own 0..270 span
    # gives a real reachable range of -225..+45 deg, against the -90..+180 the
    # USD declares.
    #
    # This matters more than a calibration detail. The policy sat on sim's -90
    # limit for 68% of every episode and no penalty could move it -- because on
    # the real arm that pose is servo 135, the MIDDLE of the travel, with 135 deg
    # of roll still available below it. The wrist-roll shortage that shaped this
    # whole task (the "dead zone", the narrowed heading window, the joint-limit
    # penalty that would not take) was in large part a sim artefact.
    j5_limit_override = (-3.9270, 0.7854)   # rad, = -225 .. +45 deg

    # The five passive finger joints come out of the USD conversion with limits
    # of +/-FLT_MAX (+/-3.4e38 rad). PhysX only accepts revolute limits within
    # +/-2Pi, and Isaac Lab's write_joint_position_limit_to_sim re-sends EVERY
    # joint's limits, so touching j5 made PhysX reject these five and very
    # likely drop the j5 change with them. They are held at a fixed closed pose
    # anyway, so a snug range around it costs nothing.
    finger_limit_override = (-1.2, 1.2)     # rad, comfortably around +/-0.9
    j5_servo_offset_deg = 135.0             # for the record; replay applies it

    joint_limit_margin = 0.08
    rew_joint_limit_penalty = 2.0

    rew_action_rate_penalty = 0.05
    # Small magnitude penalty
    rew_action_penalty_scale = 0.005
