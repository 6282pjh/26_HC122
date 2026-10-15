# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Incision V3: clean redesign (position-target control, smooth + precise)

관측 (29D):
    q(5), dq(5), prev_action(5), ee_pos(3),
    current_target_rel(3), next_target_rel(3), progress(1), tool_downward_cos(1),
    line_dir(2), blade_align(1)

액션 (5D):
    - 관절 위치 타깃 델타. target += action * joint_delta_scale, PD가 추종.
      (실제 DOFBOT 서보의 위치 명령 인터페이스와 일치)

RobotRetract는 정적 프롭 (굽은 자세로 피부를 잡고 고정, 정책 무관).
"""

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import SPHERE_MARKER_CFG
from isaaclab.utils.math import matrix_from_quat

from .dofbot_incision_env_cfg_v3 import DofbotIncisionEnvCfgV3


class DofbotIncisionEnvV3(DirectRLEnv):
    """Knife arm traces a random incision line vertically; holder arm is static."""

    cfg: DofbotIncisionEnvCfgV3

    def __init__(
        self, cfg: DofbotIncisionEnvCfgV3, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        self._arm_dof_idx, _ = self.robot_incise.find_joints(
            self.cfg.arm_dof_names, preserve_order=True
        )
        self._finger_dof_idx, _ = self.robot_incise.find_joints(
            ["llink_joint[1-3]", "rlink_joint[2-3]"]
        )
        self._grip_dof_idx, _ = self.robot_incise.find_joints(
            self.cfg.gripper_dof_name, preserve_order=True
        )
        self._ee_body_idx, _ = self.robot_incise.find_bodies(self.cfg.ee_link_name)
        self._wrist_body_idx, _ = self.robot_incise.find_bodies(self.cfg.wrist_link_name)

        # Bodies guarded against dipping below the work surface. base_link is
        # excluded: it is bolted to the support box and never moves, so it can
        # only add noise to the constraint.
        # The two fingertips are guarded as POINTS, not as link origins: rlink2's
        # origin sits 50.5 mm above its own fingertip, so an origin-only guard
        # reports the arm clear while the fingers are already in the skin.
        self._finger_body_idx, _ = self.robot_incise.find_bodies(["rlink2", "llink2"])

        self._guard_body_idx = [
            i for i, name in enumerate(self.robot_incise.body_names) if name != "base_link"
        ]

        self.joint_pos = self.robot_incise.data.joint_pos
        self.joint_vel = self.robot_incise.data.joint_vel

        # Persistent position-target buffer the policy moves via deltas.
        self._joint_targets = self.robot_incise.data.default_joint_pos.clone()
        # Arm-joint soft limits for clamping the targets.
        # Widen the wrist roll to the range the real servo actually covers before
        # anything reads the limits. See cfg.j5_limit_override for the measurement.
        # The finger limits MUST be fixed in the same pass. Isaac Lab re-sends
        # every joint's limits on any write, and the USD conversion left the five
        # passive finger joints at +/-FLT_MAX, which PhysX refuses for a revolute
        # joint (limits must be within +/-2Pi). Sending j5 alone made PhysX
        # reject the whole batch.
        def _set_limits(joint_names, lo, hi):
            idx, _ = self.robot_incise.find_joints(joint_names)
            lim = torch.tensor([[lo, hi]], device=self.device)
            lim = lim.unsqueeze(0).expand(self.num_envs, len(idx), 2).clone()
            for robot in (self.robot_incise, self.robot_retract):
                robot.write_joint_position_limit_to_sim(lim, joint_ids=idx)
            return idx

        if self.cfg.finger_limit_override is not None:
            lof, hif = self.cfg.finger_limit_override
            _set_limits(["llink_joint[1-3]", "rlink_joint[2-3]"], lof, hif)

        if self.cfg.j5_limit_override is not None:
            lo5, hi5 = self.cfg.j5_limit_override
            _set_limits(["arm_joint5"], lo5, hi5)

        # Read back what PhysX actually accepted -- the print used to report the
        # intended value whether or not it was applied.
        _j5, _ = self.robot_incise.find_joints(["arm_joint5"])
        _got = self.robot_incise.data.joint_pos_limits[0, _j5[0]]
        print(f"[Incision-V3] arm_joint5 range -> "
              f"{torch.rad2deg(_got[0]):+.1f}..{torch.rad2deg(_got[1]):+.1f} deg "
              f"(read back from sim; USD default was -90..+180)")

        limits = self.robot_incise.data.soft_joint_pos_limits  # (envs, joints, 2)
        self._arm_lower = limits[:, self._arm_dof_idx, 0]
        self._arm_upper = limits[:, self._arm_dof_idx, 1]

        n = self.cfg.num_waypoints
        self._waypoints_local = torch.zeros((self.num_envs, n, 3), device=self.device)
        # Unit travel direction of the straight incision line (start -> end);
        # the blade edge must face this way.
        self._line_dir = torch.zeros((self.num_envs, 2), device=self.device)
        self._target_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Continuous-cut state. The task is a STROKE, not a set of points: the
        # blade has to stay in contact and drag along the line. _cut_s is how far
        # along the line has actually been cut, and it only extends while the cut
        # conditions hold, so lifting the knife freezes progress where it broke.
        self._line_start = torch.zeros((self.num_envs, 2), device=self.device)
        self._line_len = torch.zeros(self.num_envs, device=self.device)
        self._cut_s = torch.zeros(self.num_envs, device=self.device)
        # Was the blade cutting on the PREVIOUS step? A stroke has to be
        # continuous in time, not just contiguous in space -- see _get_rewards.
        self._was_cutting = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Action buffers for smoothness (rate penalty + prev action in obs)
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._prev_actions = torch.zeros_like(self.actions)
        self._action_rate_sq = torch.zeros(self.num_envs, device=self.device)

        # Debug visualization of the incision line
        marker_cfg = SPHERE_MARKER_CFG.copy()
        marker_cfg.markers["sphere"].radius = 0.006
        marker_cfg.prim_path = "/Visuals/IncisionWaypoints"
        self._waypoint_markers = VisualizationMarkers(marker_cfg)

        # The blade itself has no mesh (it is a rigid offset from the gripper),
        # so render its tip — otherwise the arm looks like it stops above the
        # skin with nothing touching the line.
        tip_cfg = SPHERE_MARKER_CFG.copy()
        tip_cfg.markers["sphere"].radius = 0.004
        tip_cfg.prim_path = "/Visuals/BladeTip"
        self._blade_tip_marker = VisualizationMarkers(tip_cfg)

        print("[Incision-V3] ========================================")
        print("[Incision-V3] Dofbot Skin-Incision V3 (position-target control)")
        print("[Incision-V3] ========================================")
        print(f"[Incision-V3] Number of environments: {self.num_envs}")
        print(f"[Incision-V3] Joint delta scale: {self.cfg.joint_delta_scale} rad/step")
        print(f"[Incision-V3] Waypoints per episode: {self.cfg.num_waypoints}")
        print(f"[Incision-V3] Incision depth: {self.cfg.incision_depth}m")
        print(f"[Incision-V3] Line: STRAIGHT, length {self.cfg.line_length_range}m, random pos+dir")
        print(f"[Incision-V3] Cut test: xy<{self.cfg.waypoint_xy_tol}m, z<{self.cfg.waypoint_z_tol}m, "
              f"downward cos>{self.cfg.vertical_cos_tol}, blade cos>{self.cfg.blade_align_tol}")
        print("[Incision-V3] ========================================")

    # Setup ---------------------------------------------------------------
    def _setup_scene(self):
        self.robot_incise = Articulation(self.cfg.robot_incise_cfg)
        self.robot_retract = Articulation(self.cfg.robot_retract_cfg)
        self.skin = RigidObject(self.cfg.skin_cfg)
        # Real-rig props (support boxes + surgical table). Collision is ON so the
        # arm cannot learn a path that would hit them on hardware — the skin patch
        # deliberately keeps collision OFF so the blade can enter it.
        self.knife_box = RigidObject(self.cfg.knife_box_cfg)
        self.holder_box = RigidObject(self.cfg.holder_box_cfg)
        self.table = RigidObject(self.cfg.table_cfg)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot_incise"] = self.robot_incise
        self.scene.articulations["robot_retract"] = self.robot_retract
        self.scene.rigid_objects["skin"] = self.skin
        self.scene.rigid_objects["knife_box"] = self.knife_box
        self.scene.rigid_objects["holder_box"] = self.holder_box
        self.scene.rigid_objects["table"] = self.table

        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0))
        light_cfg.func("/World/Light", light_cfg)

    # Pre-physics ---------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        actions = torch.clamp(actions, -1.0, 1.0)
        # Smoothness bookkeeping (used by the reward and the observation)
        self._action_rate_sq = torch.sum((actions - self._prev_actions) ** 2, dim=-1)
        self._prev_actions = self.actions.clone()
        self.actions = actions.clone()

        # Integrate the position targets and clamp to the joint limits.
        arm_targets = self._joint_targets[:, self._arm_dof_idx]
        arm_targets = arm_targets + actions * self.cfg.joint_delta_scale
        arm_targets = torch.clamp(arm_targets, self._arm_lower, self._arm_upper)
        self._joint_targets[:, self._arm_dof_idx] = arm_targets
        self._joint_targets[:, self._grip_dof_idx[0]] = self.cfg.gripper_closed_target

        self.robot_incise.set_joint_position_target(self._joint_targets)

    def _apply_action(self):
        pass

    # Waypoint generation ----------------------------------------------------
    def _generate_waypoints(self, env_ids: torch.Tensor):
        """Random STRAIGHT incision line on the skin surface, fixed cut depth.

        Random centre + direction + length (5-12cm); the length shrinks when
        the sampled angle would push an endpoint outside the sampling region.
        """
        n_env = env_ids.shape[0]
        n_pts = self.cfg.num_waypoints
        x_lo, x_hi = self.cfg.incision_x_range
        y_lo, y_hi = self.cfg.incision_y_range
        l_lo, l_hi = self.cfg.line_length_range

        # Travel direction restricted to [60deg, 150deg): the arm cuts ACROSS its
        # own front. 0 deg points from the knife base toward the holder, so this
        # band is the sideways sweep; cutting toward/away from itself needs a
        # wrist roll the DOFBOT does not have.
        #
        # Completion by heading, measured on the 08-01 policy (64 envs):
        #   60-90   10/10      120-150   8/12
        #   90-120  12/12      330-60    8/30  <- wrist runs out of roll
        # A 180deg window would be needed to cut every line orientation from a
        # fixed patch, but only ~90deg of headings are actually feasible, so the
        # rig covers the rest instead: the square patch drops into the jig at 0
        # or 90 deg, and any orientation lands inside this band from one of them.
        # Training the full 180deg spent half of every batch on headings that can
        # never complete, which is what left 120-150 at 8/12.
        # [60, 105) deg. The upper half is obstructed by the HOLDER, not by the
        # arm's kinematics -- parking the holder 5 m away lifts it from 7/34 to
        # 23/34 completions. Three attempts to pose the holder out of the way
        # all landed on 55-56% against 72% with it removed: a FIXED retune, then
        # per-episode selection by fingertip distance from the line, then by the
        # arm's whole footprint. They fail for the same reason -- the holder's
        # base is at x=0.370 and the patch ends at x=0.265, so its arm must
        # cross x 0.24-0.37 whatever the joint angles, and that is the band the
        # knife needs for the upper headings. Only moving the base would fix it,
        # which is a physical change to the rig.
        #
        # So: keep the holder, cut only where it does not interfere (user,
        # 2026-08-31). Measured in this band with the blade-dive fix applied:
        # [60,75) 9/10, [75,90) 9/10, [90,105) 10/10, 97-100% cut.
        #
        # DEPLOYMENT: 45 deg wide, so the patch must be laid down with the
        # intended line inside the band. It cannot be turned 90 deg to cover the
        # rest (240 mm long side vs a 220 mm gap between the support boxes).
        #
        # Original basis for the earlier 90 deg width, still valid:
        # with the holder in place and the blade-dive fix applied, that the two
        # halves behave completely differently:
        #     [60,105)  28/30 completions, 97-100% cut
        #     [105,150) 7/34 completions, 28-30% cut
        # Removing the holder lifts the upper band to 23/34, so that band is
        # obstructed rather than hard, and the decision (user, 2026-08-31) is to
        # keep the holder and only cut where it does not interfere. Training the
        # full range spends half of every batch on headings the rig cannot do --
        # the same mistake that held completion at 59% until the window was cut
        # from 180 to 90 deg back in July.
        #
        # DEPLOYMENT: this is a 45 deg band, so the patch has to be laid down
        # with the intended line inside it. The patch cannot be turned 90 deg to
        # cover the rest (240 mm long side vs a 220 mm gap between the support
        # boxes), so line orientation has to be controlled when placing it.
        h_lo, h_hi = self.cfg.incision_heading_range
        theta = torch.deg2rad(
            torch.tensor(h_lo, device=self.device)
        ) + torch.rand(n_env, device=self.device) * torch.deg2rad(
            torch.tensor(h_hi - h_lo, device=self.device)
        )
        length = torch.rand(n_env, device=self.device) * (l_hi - l_lo) + l_lo
        # Fit the line inside the region at this angle
        eps = 1e-6
        max_len = torch.minimum(
            (x_hi - x_lo) / torch.clamp(torch.abs(torch.cos(theta)), min=eps),
            (y_hi - y_lo) / torch.clamp(torch.abs(torch.sin(theta)), min=eps),
        )
        length = torch.minimum(length, max_len)
        dx = 0.5 * length * torch.cos(theta)
        dy = 0.5 * length * torch.sin(theta)
        # Feasible centre box given the half-extents, then sample the centre
        cx = torch.rand(n_env, device=self.device) * ((x_hi - dx.abs()) - (x_lo + dx.abs())) + (x_lo + dx.abs())
        cy = torch.rand(n_env, device=self.device) * ((y_hi - dy.abs()) - (y_lo + dy.abs())) + (y_lo + dy.abs())
        start = torch.stack((cx - dx, cy - dy), dim=-1)
        end = torch.stack((cx + dx, cy + dy), dim=-1)

        t = torch.linspace(0.0, 1.0, n_pts, device=self.device).view(1, n_pts, 1)
        xy = start.unsqueeze(1) * (1.0 - t) + end.unsqueeze(1) * t
        z = torch.full(
            (n_env, n_pts, 1),
            self.cfg.skin_top_z - self.cfg.incision_depth,
            device=self.device,
        )
        self._waypoints_local[env_ids] = torch.cat((xy, z), dim=-1)

        d = end - start
        seg_len = torch.linalg.norm(d, dim=-1)
        self._line_dir[env_ids] = d / seg_len.clamp(min=1e-6).unsqueeze(-1)
        self._line_start[env_ids] = start
        self._line_len[env_ids] = seg_len
        self._cut_s[env_ids] = 0.0

    def _point_on_line_w(self, s: torch.Tensor) -> torch.Tensor:
        """World position of the cut point at arc-length `s` along the line."""
        s = s.clamp(min=0.0).minimum(self._line_len).unsqueeze(-1)
        xy = self._line_start + s * self._line_dir
        z = torch.full_like(s, self.cfg.skin_top_z - self.cfg.incision_depth)
        return torch.cat((xy, z), dim=-1) + self.scene.env_origins

    def _tip_line_coords(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Blade tip in line coordinates: (distance along, distance from).

        `lateral` is what decides whether the blade is ON the line, and it is
        deliberately not the distance to a target point -- while stroking, the
        tip runs ahead of where the cut currently ends, and that is correct.
        """
        rel = self._blade_tip_pos_w()[:, :2] - self.scene.env_origins[:, :2] - self._line_start
        s = torch.sum(rel * self._line_dir, dim=-1)
        lateral = torch.linalg.norm(rel - s.unsqueeze(-1) * self._line_dir, dim=-1)
        return s, lateral

    def _current_target_pos_w(self) -> torch.Tensor:
        """Where the cut currently ends -- the point the blade must be at to
        carry on. After a break this is the break point, so the arm has to come
        back to it rather than skip ahead."""
        return self._point_on_line_w(self._cut_s)

    def _next_target_pos_w(self) -> torch.Tensor:
        return self._point_on_line_w(self._cut_s + self.cfg.cut_lookahead)

    def _blade_tip_pos_w(self) -> torch.Tensor:
        """World position of the BLADE TIP, not the gripper.

        The real rig clamps a blade in the gripper, so the cutting point sits
        `blade_length` beyond the fingers along the tool approach axis. Tracking
        the finger link itself drove it to the waypoint depth, i.e. 5mm inside
        the 10mm-thick patch — the gripper visibly passed through the skin.
        With the offset the fingers ride clear above the surface and only the
        blade reaches cut depth, which is also what sim2real needs: the policy's
        target is the blade tip the camera pipeline will hand it.
        """
        grip_pos = self.robot_incise.data.body_pos_w[:, self._ee_body_idx[0]]
        wrist_quat = self.robot_incise.data.body_quat_w[:, self._wrist_body_idx[0]]
        approach_world = matrix_from_quat(wrist_quat)[:, :, 2]
        return grip_pos + self.cfg.blade_length * approach_world

    def _finger_tip_z(self) -> torch.Tensor:
        """Height of both FINGERTIPS above the work surface, (envs, 2).

        The fingertip is `finger_length` along the link's local +x, which is the
        tool axis. Guarding link origins instead understates the depth by
        50.5 mm, which is the whole length of the finger.
        """
        idx = self._finger_body_idx
        pos = self.robot_incise.data.body_pos_w[:, idx]
        quat = self.robot_incise.data.body_quat_w[:, idx]
        rot = matrix_from_quat(quat.reshape(-1, 4)).reshape(quat.shape[0], quat.shape[1], 3, 3)
        off = torch.tensor(
            [self.cfg.finger_length, 0.0, 0.0], device=self.device
        ).view(1, 1, 3, 1)
        tips = pos + (rot @ off).squeeze(-1)
        return tips[:, :, 2] - self.scene.env_origins[:, 2].unsqueeze(-1)

    def _blade_tip_z(self) -> torch.Tensor:
        """Blade tip height above the work surface."""
        return self._blade_tip_pos_w()[:, 2] - self.scene.env_origins[:, 2]

    def _blade_dive(self) -> torch.Tensor:
        """How far the blade tip is below the patch bottom, in metres.

        Zero while the blade is in the material. The tip is a COMPUTED point,
        not a rigid body, so nothing physical stops it: the floor guard watches
        the fingers and deliberately exempts the blade, but that left the blade
        with no lower bound at all. Measured at headings 135-150: failing
        rollouts drove the tip to -15..-23 mm against a -5 mm target, i.e.
        5-13 mm INSIDE the surgical table, and depth is exactly what separated
        them from the successful ones (-3..-11 mm). On hardware the blade would
        have hit the table.
        """
        return (self.cfg.blade_floor_z - self._blade_tip_z()).clamp(
            min=0.0, max=self.cfg.blade_dive_cap
        )

    def _min_guarded_link_z(self) -> torch.Tensor:
        """Height of the lowest guarded point above the work surface.

        Link origins PLUS the two fingertips -- the fingers are the part that
        actually reaches the skin, and only the blade beyond them is allowed
        below the surface.
        """
        z = (
            self.robot_incise.data.body_pos_w[:, self._guard_body_idx, 2]
            - self.scene.env_origins[:, 2].unsqueeze(-1)
        )
        return torch.cat((z, self._finger_tip_z()), dim=1).min(dim=1).values

    def _floor_violation(self) -> torch.Tensor:
        """How far the lowest guarded link is below the work surface, in metres.

        Zero whenever the arm is clear. The blade tip is excluded on purpose —
        it is supposed to go below the skin; everything else is not.
        """
        return (self.cfg.workspace_floor_z - self._min_guarded_link_z()).clamp(
            min=0.0, max=self.cfg.floor_penalty_cap
        )

    def _tool_downward_cos(self) -> torch.Tensor:
        """Alignment of the tool approach axis (wrist local +z) with world -z."""
        wrist_quat = self.robot_incise.data.body_quat_w[:, self._wrist_body_idx[0]]
        approach_world = matrix_from_quat(wrist_quat)[:, :, 2]
        return -approach_world[:, 2]

    def _blade_align_signed(self) -> torch.Tensor:
        """Signed alignment of the blade edge with the travel direction.

        Wrist local +x is the blade edge: measured against the gripper, the jaw
        axis is wrist local y (|cos| = 1.000) and the edge is perpendicular to
        it, so +x is correct. Kept signed for the observation, which needs the
        sign to know which way to roll.
        """
        wrist_quat = self.robot_incise.data.body_quat_w[:, self._wrist_body_idx[0]]
        blade_world = matrix_from_quat(wrist_quat)[:, :, 0]
        b = blade_world[:, :2]
        b = b / torch.linalg.norm(b, dim=-1, keepdim=True).clamp(min=1e-6)
        return torch.sum(b * self._line_dir, dim=-1)

    def _blade_align_cos(self) -> torch.Tensor:
        """How well the blade can cut along the line, in [0, 1].

        UNSIGNED on purpose: the cutting edge is a line segment, not an arrow.
        Rolling the knife 180 deg about the vertical gives an identical cut, so
        both orientations must score the same. Scoring the signed cosine (as
        this did until 2026-08-20) picked one of the two equivalent poses as the
        only correct one and paid the minimum for the other, which doubled the
        wrist roll the arm had to produce and is part of what was measured as a
        kinematic "dead zone" around headings 240-300 (peak signed blade -0.25
        there — that is |cos| = 0.25 at worst, and better where the roll went
        further past perpendicular).
        """
        return self._blade_align_signed().abs()

    # Observations ----------------------------------------------------------
    def _get_observations(self) -> dict:
        ee_pos_w = self._blade_tip_pos_w()
        current_target_w = self._current_target_pos_w()
        next_target_w = self._next_target_pos_w()
        progress = (self._cut_s / self._line_len.clamp(min=1e-6)).clamp(0.0, 1.0).unsqueeze(-1)
        downward = self._tool_downward_cos().unsqueeze(-1)
        blade_align = self._blade_align_signed().unsqueeze(-1)

        obs = torch.cat(
            (
                self.joint_pos[:, self._arm_dof_idx].reshape(self.num_envs, -1),  # 5
                self.joint_vel[:, self._arm_dof_idx].reshape(self.num_envs, -1),  # 5
                self.actions,  # 5 (previous applied action)
                ee_pos_w,  # 3
                current_target_w - ee_pos_w,  # 3
                next_target_w - ee_pos_w,  # 3
                progress,  # 1
                downward,  # 1
                self._line_dir,  # 2 (travel direction the blade must face)
                blade_align,  # 1
            ),
            dim=-1,
        )
        self._update_visualization()
        return {"policy": obs}

    def _update_visualization(self):
        all_waypoints_w = (
            self._waypoints_local + self.scene.env_origins.unsqueeze(1)
        ).reshape(-1, 3)
        self._waypoint_markers.visualize(translations=all_waypoints_w)
        self._blade_tip_marker.visualize(translations=self._blade_tip_pos_w())

    # Rewards -----------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        ee_pos_w = self._blade_tip_pos_w()
        current_target_w = self._current_target_pos_w()
        delta = ee_pos_w - current_target_w
        xy_err = torch.linalg.norm(delta[:, :2], dim=-1)
        z_err = torch.abs(delta[:, 2])

        downward = self._tool_downward_cos()
        vertical_ok = downward > self.cfg.vertical_cos_tol
        blade_align = self._blade_align_cos()
        blade_ok = blade_align > self.cfg.blade_align_tol

        s, lateral = self._tip_line_coords()

        reward = torch.zeros(self.num_envs, device=self.device)

        # Posture terms are APPROACH AIDS, not income. They used to pay
        # 5.0/step unconditionally -- 4490 over an episode for holding a pose in
        # mid-air, against ~330 actually earned by cutting. That is why the
        # policy hovered: being near the line paid ~25x better than cutting it.
        # Now they are weighted by how close the blade is to the LINE, so
        # posture only pays where posture matters, and their scales are cut.
        near = 1.0 - torch.tanh(lateral / self.cfg.posture_gate_eps)

        # Keep the tool vertical (dense; (d+1)/2 keeps gradient even when the
        # tool points up, unlike clamping at zero)
        reward += self.cfg.rew_vertical_scale * 0.5 * (downward + 1.0) * near

        # Blade edge must lie along the line (dense). blade_align is already
        # |cos| in [0, 1], so no remapping: 0 at perpendicular (cannot cut),
        # 1 when the edge lies along the travel direction either way round.
        reward += self.cfg.rew_blade_align_scale * blade_align * near

        # (1) align the tool over the cut point: coarse kernel pulls from far,
        #     sharp kernel shapes the last centimetre
        far_prox = 1.0 - torch.tanh(xy_err / self.cfg.incise_far_eps)
        reward += self.cfg.rew_incise_far_scale * far_prox
        xy_prox = 1.0 - torch.tanh(xy_err / self.cfg.incise_xy_eps)
        reward += self.cfg.rew_incise_xy_scale * xy_prox
        # (2) once aligned, plunge straight down to cut depth. TWO-SCALE, for
        #     the same reason xy is: an 8 mm kernel alone is fully saturated at
        #     the 35 mm standoff the arm starts from (1 - tanh(35/8) = 0.0003),
        #     so it supplies no gradient to descend and the policy just hovers
        #     in xy. Measured on the 08-20 run: xy error 0.0 mm, depth gate open
        #     0-11 steps of 898. The coarse kernel is worth 0.40 at that same
        #     standoff, which is a slope the policy can actually follow down.
        xy_aligned = xy_err < self.cfg.incise_xy_align_tol
        depth_far = 1.0 - torch.tanh(z_err / self.cfg.incise_depth_far_eps)
        reward += self.cfg.rew_incise_depth_far_scale * depth_far * xy_aligned.float()
        depth_prox = 1.0 - torch.tanh(z_err / self.cfg.incise_depth_eps)
        reward += self.cfg.rew_incise_depth_scale * depth_prox * xy_aligned.float()

        # THE CUT. Not "did the tip touch point k" -- that let the policy stab
        # each waypoint in turn and lift in between, which is poking, not
        # incising. Here the blade is cutting whenever it is on the line, at
        # depth, vertical and squared up, and the cut only EXTENDS while that
        # holds. Lift the knife and _cut_s simply stops where it broke.
        cutting = (
            (lateral < self.cfg.cut_lateral_tol)
            & (z_err < self.cfg.waypoint_z_tol)
            & vertical_ok
            & blade_ok
        )
        # Contiguity in SPACE: the tip may only extend the cut from where it
        # already ends. Without this the policy could satisfy the conditions at
        # the far end of the line and claim the whole cut.
        #
        # Contiguity in TIME (`_was_cutting`): the blade must have been cutting
        # on the previous step too. Spatial contiguity alone still allowed a
        # dotted line -- touch, lift, move a gap, touch again -- which is the
        # same poking behaviour at finer granularity. A real incision is plunge
        # once at the start, then drag at constant depth, so an isolated touch
        # must not count.
        can_extend = (
            cutting
            & self._was_cutting
            & (s > self._cut_s)
            & (s <= self._cut_s + self.cfg.cut_max_gap)
        )
        new_s = torch.where(can_extend, s.minimum(self._line_len), self._cut_s)
        gain = (new_s - self._cut_s).clamp(min=0.0)
        self._cut_s = new_s

        # Breaking contact mid-stroke is a failed cut, not merely a pause: pay
        # for it, or "lift and re-approach" stays free and the policy keeps
        # stitching instead of dragging.
        started = self._cut_s > 0.0
        lifted = self._was_cutting & (~cutting) & started & (~self._completed)
        reward -= self.cfg.rew_cut_lift_penalty * lifted.float()
        self._was_cutting = cutting
        # Paid per millimetre actually cut, normalised so a full stroke is worth
        # rew_cut_progress_scale regardless of how long the sampled line was.
        reward += self.cfg.rew_cut_progress_scale * gain / self._line_len.clamp(min=1e-6)

        just_completed = (
            (self._cut_s >= self._line_len - self.cfg.cut_finish_eps) & (~self._completed)
        )
        reward += self.cfg.rew_completion_bonus * just_completed.float()
        self._completed |= just_completed

        # Kept only so the waypoint markers light up in order during rendering.
        frac = (self._cut_s / self._line_len.clamp(min=1e-6)).clamp(0.0, 1.0)
        self._target_idx = (frac * (self.cfg.num_waypoints - 1)).long()

        # Stay above the work surface. Zero while the arm is clear, so it acts
        # as a constraint rather than a shaping gradient the policy has to fight.
        # Joint-limit saturation. Sim clamps a joint at its limit for free, so the
        # policy happily parks against one: the first exported trajectory held
        # arm_joint3 at its stop for 315 of 420 frames, the whole cut included.
        # A real serial servo commanded to the end of its range leans on the
        # mechanical stop, stalls and heats, and has no torque feedback to stop
        # it. Penalise the last `joint_limit_margin` of travel at either end.
        lo, hi = self._arm_lower, self._arm_upper
        span = (hi - lo).clamp(min=1e-6)
        q = self.joint_pos[:, self._arm_dof_idx]
        frac = ((q - lo) / span).clamp(0.0, 1.0)
        m = self.cfg.joint_limit_margin
        near = (
            (m - frac).clamp(min=0.0) + (frac - (1.0 - m)).clamp(min=0.0)
        ) / m
        limit_pen = near.sum(dim=-1)
        reward -= self.cfg.rew_joint_limit_penalty * limit_pen
        limit_frac = (near > 0).any(dim=-1).float().mean()

        floor_violation = self._floor_violation()
        reward -= self.cfg.rew_floor_penalty_scale * floor_violation
        # The blade may enter the skin, but not the table under it.
        blade_dive = self._blade_dive()
        reward -= self.cfg.rew_blade_dive_penalty_scale * blade_dive
        # Surfaced in tensorboard so the retrain can be judged on whether the
        # arm actually stopped diving, not on total reward — which stayed at an
        # all-time high through a run that completed only 31% of lines.
        self.extras["log"] = {
            "floor_violation_m": floor_violation.mean(),
            "floor_violation_frac": (floor_violation > 0).float().mean(),
            # Fraction of the line actually cut. Replaces the old waypoint index:
            # judge runs by this, and by the per-gate breakdown in the eval.
            "cut_fraction": frac.mean(),
            "cut_fraction_max": frac.max(),
            "cutting_frac": cutting.float().mean(),
            "lift_frac": lifted.float().mean(),
            "blade_dive_m": blade_dive.mean(),
            "blade_dive_frac": (blade_dive > 0).float().mean(),
            "joint_limit_frac": limit_frac,
            "waypoint_progress": frac.mean(),  # alias, keeps older tooling working
        }

        # Smoothness: action-rate penalty + small magnitude penalty
        reward -= self.cfg.rew_action_rate_penalty * self._action_rate_sq
        reward -= self.cfg.rew_action_penalty_scale * torch.sum(torch.abs(self.actions), dim=-1)

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # Driving a link under the table is a crash on hardware, so end the
        # episode there. Note this is the opposite case to completion below:
        # terminating on a BAD event teaches avoidance, which is what we want;
        # terminating on a GOOD one taught the policy to avoid finishing.
        crashed = (self._min_guarded_link_z() < self.cfg.floor_crash_z) | (
            self._blade_tip_z() < self.cfg.blade_crash_z
        )
        # Do NOT terminate on completion. With positive dense shaping, ending
        # the episode early on success forfeits all remaining dense reward, so
        # a completion-terminated env teaches the policy to AVOID finishing
        # (observed in run 2: mean episode length climbed back toward the cap
        # while min stayed ~67 — the skill existed but was being suppressed).
        # Completion still pays its one-time bonus; episodes run to timeout,
        # with the arm holding the final waypoint afterwards.
        return crashed, time_out

    # Reset ---------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot_incise._ALL_INDICES
        super()._reset_idx(env_ids)

        env_tensor = self._to_env_tensor(env_ids)

        self._target_idx[env_tensor] = 0
        self._cut_s[env_tensor] = 0.0
        self._was_cutting[env_tensor] = False
        self._completed[env_tensor] = False
        self.actions[env_tensor] = 0.0
        self._prev_actions[env_tensor] = 0.0
        self._generate_waypoints(env_tensor)

        # Knife arm: small randomization around the neutral pose, on the ARM
        # joints only. The five passive finger joints form a closed linkage that
        # has to track grip_joint with +/-1 multipliers (the URDF mimic tags are
        # dropped by the USD importer, so the pose is set explicitly). Jittering
        # each of them independently by +/-0.05 rad makes the linkage internally
        # inconsistent, and since these values are then copied into
        # `_joint_targets`, the PD spends the whole episode holding an impossible
        # pose -- which is the gripper looking broken.
        joint_pos = self.robot_incise.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot_incise.data.default_joint_vel[env_ids].clone()
        noise = 0.05 * (2.0 * torch.rand_like(joint_pos) - 1.0)
        joint_pos[:, self._arm_dof_idx] += noise[:, self._arm_dof_idx]

        default_root = self.robot_incise.data.default_root_state[env_ids].clone()
        default_root[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel
        self.robot_incise.write_root_pose_to_sim(default_root[:, :7], env_ids)
        self.robot_incise.write_root_velocity_to_sim(default_root[:, 7:], env_ids)
        self.robot_incise.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # Position targets start exactly at the (randomized) reset pose so the
        # PD exerts no initial pull.
        self._joint_targets[env_tensor] = joint_pos
        # Grip and the passive fingers are commanded at exactly their closed
        # pose, never at a randomized one.
        self._joint_targets[env_tensor, self._grip_dof_idx[0]] = self.cfg.gripper_closed_target
        self._joint_targets[env_tensor[:, None], self._finger_dof_idx] = (
            self.robot_incise.data.default_joint_pos[env_ids][:, self._finger_dof_idx]
        )
        self.robot_incise.set_joint_position_target(self._joint_targets[env_tensor], env_ids=env_ids)

        # Holder arm: one fixed pose, pressing the patch's far edge. See the
        # cfg for why per-episode selection was dropped.
        hold_pos = self.robot_retract.data.default_joint_pos[env_ids].clone()
        hold_vel = torch.zeros_like(hold_pos)
        hold_root = self.robot_retract.data.default_root_state[env_ids].clone()
        hold_root[:, :3] += self.scene.env_origins[env_ids]
        self.robot_retract.write_root_pose_to_sim(hold_root[:, :7], env_ids)
        self.robot_retract.write_root_velocity_to_sim(hold_root[:, 7:], env_ids)
        self.robot_retract.write_joint_state_to_sim(hold_pos, hold_vel, None, env_ids)
        self.robot_retract.set_joint_position_target(hold_pos, env_ids=env_ids)

    # Helpers -------------------------------------------------------------
    def _to_env_tensor(self, env_ids):
        if isinstance(env_ids, slice):
            start = env_ids.start or 0
            stop = env_ids.stop or self.num_envs
            step = env_ids.step or 1
            return torch.arange(start, stop, step, device=self.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
