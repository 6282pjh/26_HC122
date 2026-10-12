"""
Opening (개복) V0: both arms follow the taught opening motion and hold it.

관측 (43D):
    팔마다 20: q(4), dq(4), prev_action(4), target_rel(3), lookahead_rel(3),
               progress(1), tool_downward_cos(1)
    공통 3:    spread_dir(2), hold_fraction(1)

액션 (8D):
    - 팔마다 관절 1~4 의 위치 타깃 델타. 손목 롤(5번)과 그리퍼는 교시값 고정.

index 0 = right arm (I2C bus 8, base at the origin), 1 = left arm (bus 1).
"""

from __future__ import annotations

import json
import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import SPHERE_MARKER_CFG
from isaaclab.utils.math import matrix_from_quat

from .dofbot_opening_env_cfg import DofbotOpeningEnvCfg

ARM_KEYS = ("right", "left")


class DofbotOpeningEnv(DirectRLEnv):
    """Two arms insert at the incision and spread it, tracking a taught path."""

    cfg: DofbotOpeningEnvCfg

    def __init__(self, cfg: DofbotOpeningEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.robots = (self.robot_incise, self.robot_retract)
        ref = self.robot_incise
        self._arm_dof_idx, _ = ref.find_joints(self.cfg.opening_arm_dof_names, preserve_order=True)
        self._arm5_dof_idx, _ = ref.find_joints(self.cfg.arm_dof_names, preserve_order=True)
        self._j5_dof_idx = self._arm5_dof_idx[4]
        self._finger_dof_idx, _ = ref.find_joints(["llink_joint[1-3]", "rlink_joint[2-3]"])
        self._grip_dof_idx, _ = ref.find_joints(self.cfg.gripper_dof_name, preserve_order=True)
        self._ee_body_idx, _ = ref.find_bodies(self.cfg.ee_link_name)
        self._wrist_body_idx, _ = ref.find_bodies(self.cfg.wrist_link_name)
        self._guard_body_idx = [
            i for i, name in enumerate(ref.body_names)
            if name != "base_link" and not name.startswith(("rlink", "llink"))
        ]

        # Same limit fix-up as incision V3: the fingers first (the USD leaves
        # them at +/-FLT_MAX and PhysX rejects the whole batch), then the wrist
        # roll, which the taught pose (-136 deg) needs widened past the USD's -90.
        def _set_limits(joint_names, lo, hi):
            idx, _ = ref.find_joints(joint_names)
            lim = torch.tensor([[lo, hi]], device=self.device)
            lim = lim.unsqueeze(0).expand(self.num_envs, len(idx), 2).clone()
            for robot in self.robots:
                robot.write_joint_position_limit_to_sim(lim, joint_ids=idx)

        if self.cfg.finger_limit_override is not None:
            _set_limits(["llink_joint[1-3]", "rlink_joint[2-3]"], *self.cfg.finger_limit_override)
        if self.cfg.j5_limit_override is not None:
            _set_limits(["arm_joint5"], *self.cfg.j5_limit_override)

        limits = ref.data.soft_joint_pos_limits
        self._arm_lower = limits[:, self._arm_dof_idx, 0]
        self._arm_upper = limits[:, self._arm_dof_idx, 1]

        self._surface_z = self.cfg.skin_top_z + self.cfg.rig_z_offset
        self._tip_floor_z = self._surface_z - self.cfg.patch_thickness

        # Reference motion: servo degrees -> sim radians -> control-point path.
        # The servos move joint-by-joint between waypoints, so the tip travels
        # an ARC, not the straight line between the taught points (on the first
        # leg the two differ by more than 100 mm). The path is therefore the
        # joint-space interpolation, sampled `path_substeps` times per leg.
        self._ref_names, self._way_q = self._load_motion()          # (2, W, 5)
        self._sub = self.cfg.path_substeps
        a = torch.linspace(0.0, 1.0, self._sub + 1, device=self.device)[:-1].view(1, 1, -1, 1)
        dense = self._way_q[:, :-1].unsqueeze(2) * (1 - a) + self._way_q[:, 1:].unsqueeze(2) * a
        self._ref_q = torch.cat((dense.flatten(1, 2), self._way_q[:, -1:]), dim=1)   # (2, P, 5)
        self._num_pts = self._ref_q.shape[1]
        self._tight_from = self.cfg.tight_from_leg * self._sub
        self._anchor_from = self.cfg.anchor_from_index * self._sub
        self._tool_off = torch.zeros(2, device=self.device)
        self._ref_pts = self._fk_reference()                         # (2, P, 3)
        self._check_reference()

        w = torch.tensor(self.cfg.leg_weights, device=self.device, dtype=torch.float32)
        if w.numel() != self._way_q.shape[1] - 1:
            raise ValueError(
                f"leg_weights has {w.numel()} entries but the motion has "
                f"{self._way_q.shape[1] - 1} legs"
            )
        w = w.repeat_interleave(self._sub)
        self._leg_w = w / w.sum()
        self._leg_wcum = torch.cat((torch.zeros(1, device=self.device), self._leg_w.cumsum(0)))

        n, p = self.num_envs, self._num_pts
        self._path = torch.zeros((n, 2, p, 3), device=self.device)       # env-local
        self._seg_len = torch.ones((n, 2, p - 1), device=self.device)
        self._cum_len = torch.zeros((n, 2, p), device=self.device)
        self._spread_dir = torch.zeros((n, 2), device=self.device)
        # Where this episode's incision is, relative to the taught one:
        # shift x, shift y (m) and turn (rad).
        self._placement = torch.zeros((n, 3), device=self.device)
        # How far along its path each arm has actually travelled. Like the
        # incision's _cut_s: it only extends from where it already is.
        self._s = torch.zeros((n, 2), device=self.device)
        self._was_on = torch.zeros((n, 2), dtype=torch.bool, device=self.device)
        self._hold = torch.zeros(n, dtype=torch.long, device=self.device)
        self._completed = torch.zeros(n, dtype=torch.bool, device=self.device)

        self._joint_targets = torch.stack(
            [r.data.default_joint_pos.clone() for r in self.robots], dim=1
        )                                                                   # (n, 2, J)
        self.actions = torch.zeros((n, self.cfg.action_space), device=self.device)
        self._prev_actions = torch.zeros_like(self.actions)
        self._action_rate_sq = torch.zeros(n, device=self.device)

        marker_cfg = SPHERE_MARKER_CFG.copy()
        marker_cfg.markers["sphere"].radius = 0.005
        marker_cfg.prim_path = "/Visuals/OpeningPath"
        self._path_markers = VisualizationMarkers(marker_cfg)
        tip_cfg = SPHERE_MARKER_CFG.copy()
        tip_cfg.markers["sphere"].radius = 0.004
        tip_cfg.prim_path = "/Visuals/OpeningTips"
        self._tip_markers = VisualizationMarkers(tip_cfg)

        print("[Opening-V0] ========================================")
        print(f"[Opening-V0] Reference motion: {self.cfg.motion_json}")
        print(f"[Opening-V0] Number of environments: {self.num_envs}")
        print(f"[Opening-V0] Control point: wrist + {self._tool_off[0]*1000:.1f} / "
              f"{self._tool_off[1]*1000:.1f} mm along the tool axis (right / left)")
        print(f"[Opening-V0] Work surface z = {self._surface_z*1000:+.0f} mm (rig_z_offset)")
        for k, name in enumerate(self._ref_names):
            r, l = self._ref_pts[0, k * self._sub] * 1000, self._ref_pts[1, k * self._sub] * 1000
            print(f"[Opening-V0]   {name:<16} right ({r[0]:+5.0f},{r[1]:+5.0f},{r[2]:+5.0f})  "
                  f"left ({l[0]:+5.0f},{l[1]:+5.0f},{l[2]:+5.0f})  "
                  f"apart {torch.linalg.norm(r - l):4.0f} mm")
        print("[Opening-V0] ========================================")

    # Reference motion ------------------------------------------------------
    def _load_motion(self) -> tuple[list[str], torch.Tensor]:
        with open(self.cfg.motion_json) as f:
            motion = json.load(f)
        names, rows = [], []
        for wp in motion["waypoints"]:
            names.append(wp["name"])
            per_arm = []
            for key in ARM_KEYS:
                servo = wp[key]["joints"]
                inv = self.cfg.servo_inverted[key]
                q = []
                for i in range(4):
                    deg = servo[i] - 90.0
                    q.append(math.radians(-deg if inv[i] else deg))
                # servo = (q + 90) + offset, the mapping replay_on_agx.py sends
                j5 = servo[4] - 90.0 - self.cfg.j5_servo_offset_deg
                q.append(math.radians(-j5 if inv[4] else j5))
                per_arm.append(q)
            rows.append(per_arm)
        q = torch.tensor(rows, device=self.device, dtype=torch.float32)   # (P, 2, 5)
        return names, q.permute(1, 0, 2).contiguous()

    def _fk_reference(self) -> torch.Tensor:
        """Control-point position at every taught waypoint, env-local.

        Done in the simulator itself rather than from stored numbers, so the
        path is always consistent with the JSON, the servo mapping and the
        control-point definition used for the reward.
        """
        pts = torch.zeros((2, self._num_pts, 3), device=self.device)
        org = self.scene.env_origins[0]
        for k in range(self._num_pts):
            for a, robot in enumerate(self.robots):
                q = robot.data.default_joint_pos.clone()
                q[:, self._arm5_dof_idx] = self._ref_q[a, k]
                # Closed gripper, as it is held for the whole episode.
                q[:, self._grip_dof_idx[0]] = self.cfg.gripper_closed_target
                q[:, self._finger_dof_idx] = self.robot_incise.data.default_joint_pos[
                    :, self._finger_dof_idx
                ]
                robot.write_joint_state_to_sim(q, torch.zeros_like(q))
            self.sim.forward()
            self.scene.update(1e-5)
            for a, robot in enumerate(self.robots):
                wrist = robot.data.body_pos_w[0, self._wrist_body_idx[0]]
                axis = matrix_from_quat(robot.data.body_quat_w[0:1, self._wrist_body_idx[0]])[0, :, 2]
                if k == 0:
                    # Fingertip plane measured from the wrist, along the tool axis.
                    ee = robot.data.body_pos_w[0, self._ee_body_idx[0]]
                    self._tool_off[a] = (
                        torch.dot(ee - wrist, axis)
                        + self.cfg.finger_length
                        + self.cfg.opening_tool_length
                    )
                pts[a, k] = wrist + self._tool_off[a] * axis - org
        return pts

    def _check_reference(self):
        lo = self.robot_incise.data.joint_pos_limits[0, self._arm5_dof_idx, 0]
        hi = self.robot_incise.data.joint_pos_limits[0, self._arm5_dof_idx, 1]
        bad = (self._ref_q < lo - 1e-3) | (self._ref_q > hi + 1e-3)
        if bad.any():
            a, k, j = (int(v) for v in bad.nonzero()[0])
            raise ValueError(
                f"{self.cfg.motion_json}: {ARM_KEYS[a]} arm, near waypoint "
                f"{self._ref_names[k // self._sub]}, "
                f"arm_joint{j + 1} = {math.degrees(float(self._ref_q[a, k, j])):+.1f} deg is "
                f"outside the simulated joint range. Check servo_inverted / j5_servo_offset_deg."
            )
        low = float(self._ref_pts[:, :, 2].min())
        if low < self._tip_floor_z:
            raise ValueError(
                f"The taught motion reaches z = {low*1000:+.0f} mm but the table top is at "
                f"{self._tip_floor_z*1000:+.0f} mm. Lower rig_z_offset."
            )

    # Setup ---------------------------------------------------------------
    def _setup_scene(self):
        self.robot_incise = Articulation(self.cfg.robot_incise_cfg)
        self.robot_retract = Articulation(self.cfg.robot_retract_cfg)
        self.skin = RigidObject(self.cfg.skin_cfg)
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
        self._action_rate_sq = torch.sum((actions - self._prev_actions) ** 2, dim=-1)
        self._prev_actions = self.actions.clone()
        self.actions = actions.clone()

        delta = actions.view(self.num_envs, 2, 4) * self.cfg.joint_delta_scale
        for a, robot in enumerate(self.robots):
            arm = self._joint_targets[:, a, self._arm_dof_idx] + delta[:, a]
            arm = torch.clamp(arm, self._arm_lower, self._arm_upper)
            self._joint_targets[:, a, self._arm_dof_idx] = arm
            self._joint_targets[:, a, self._j5_dof_idx] = self._ref_q[a, -1, 4]
            self._joint_targets[:, a, self._grip_dof_idx[0]] = self.cfg.gripper_closed_target
            robot.set_joint_position_target(self._joint_targets[:, a])

    def _apply_action(self):
        pass

    # Path ------------------------------------------------------------------
    def _generate_path(self, env_ids: torch.Tensor):
        """Place the taught path for this episode's incision.

        Everything from waypoint `anchor_from_index` on is defined relative to
        the incision and moves rigidly with it. The approach before it is
        stretched to meet it, so the path still starts at the READY pose.
        """
        m = len(env_ids)
        shift, yaw = self._sample_placement(m)
        cos, sin = torch.cos(yaw), torch.sin(yaw)
        self._placement[env_ids] = torch.cat((shift, yaw.unsqueeze(-1)), dim=-1)

        k0 = self._anchor_from
        path = self._ref_pts.unsqueeze(0).repeat(m, 1, 1, 1)            # (m, 2, P, 3)
        centre = self._ref_pts[:, k0, :2].mean(dim=0)                   # between the two tips
        rel = path[:, :, k0:, :2] - centre
        rot = torch.stack(
            (
                cos.view(m, 1, 1) * rel[..., 0] - sin.view(m, 1, 1) * rel[..., 1],
                sin.view(m, 1, 1) * rel[..., 0] + cos.view(m, 1, 1) * rel[..., 1],
            ),
            dim=-1,
        )
        moved = centre + rot + shift.view(m, 1, 1, 2)
        offset = moved[:, :, 0] - path[:, :, k0, :2]                     # (m, 2, 2)
        path[:, :, k0:, :2] = moved
        blend = torch.linspace(0.0, 1.0, k0 + 1, device=self.device)[:-1]
        path[:, :, :k0, :2] += blend.view(1, 1, k0, 1) * offset.unsqueeze(2)

        seg = torch.linalg.norm(path[:, :, 1:] - path[:, :, :-1], dim=-1).clamp(min=1e-6)
        self._path[env_ids] = path
        self._seg_len[env_ids] = seg
        self._cum_len[env_ids] = torch.cat(
            (torch.zeros((m, 2, 1), device=self.device), seg.cumsum(dim=-1)), dim=-1
        )
        d = path[:, 1, -1, :2] - path[:, 0, -1, :2]                     # right tip -> left tip
        self._spread_dir[env_ids] = d / torch.linalg.norm(d, dim=-1, keepdim=True).clamp(min=1e-6)

    def _sample_placement(self, m: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Incision placement for `m` episodes: shift (m, 2) in metres, turn (m,) in rad."""
        cfg = self.cfg

        def _uniform(rng):
            return rng[0] + (rng[1] - rng[0]) * torch.rand(m, device=self.device)

        shift = torch.stack((_uniform(cfg.line_shift_x_range), _uniform(cfg.line_shift_y_range)), dim=-1)
        return shift, torch.deg2rad(_uniform(cfg.line_yaw_range_deg))

    def _path_total(self) -> torch.Tensor:
        return self._cum_len[:, :, -1]

    def _path_at(self, s: torch.Tensor):
        """Point, leg index and weighted progress at arc length `s` (n, 2, K)."""
        s = s.clamp(min=0.0).minimum(self._path_total().unsqueeze(-1))
        inner = self._cum_len[:, :, 1:-1].unsqueeze(2)                   # (n, 2, 1, P-2)
        leg = (s.unsqueeze(-1) >= inner).sum(dim=-1)                     # (n, 2, K)
        start = torch.gather(self._cum_len, 2, leg)
        length = torch.gather(self._seg_len, 2, leg)
        t = ((s - start) / length).clamp(0.0, 1.0)
        idx = leg.unsqueeze(-1).expand(-1, -1, -1, 3)
        a = torch.gather(self._path, 2, idx)
        b = torch.gather(self._path, 2, idx + 1)
        point = a + t.unsqueeze(-1) * (b - a)
        progress = self._leg_wcum[leg] + t * self._leg_w[leg]
        return point, leg, progress

    def _progress(self) -> torch.Tensor:
        """Weighted path progress per arm, in [0, 1]."""
        return self._path_at(self._s.unsqueeze(-1))[2].squeeze(-1)

    def _max_gap(self) -> torch.Tensor:
        leg = self._path_at(self._s.unsqueeze(-1))[1].squeeze(-1)
        return torch.where(
            leg >= self._tight_from,
            torch.full_like(self._s, self.cfg.tight_max_gap),
            torch.full_like(self._s, self.cfg.free_max_gap),
        )

    # Kinematics --------------------------------------------------------------
    def _tips(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Control point of each arm (env-local) and how vertical its tool is."""
        pos, down = [], []
        for a, robot in enumerate(self.robots):
            wrist = robot.data.body_pos_w[:, self._wrist_body_idx[0]]
            axis = matrix_from_quat(robot.data.body_quat_w[:, self._wrist_body_idx[0]])[:, :, 2]
            pos.append(wrist + self._tool_off[a] * axis - self.scene.env_origins)
            down.append(-axis[:, 2])
        return torch.stack(pos, dim=1), torch.stack(down, dim=1)

    def _min_link_z(self) -> torch.Tensor:
        """Lowest arm-link origin of either arm above the env origin."""
        z = torch.cat(
            [r.data.body_pos_w[:, self._guard_body_idx, 2] for r in self.robots], dim=1
        )
        return z.min(dim=1).values - self.scene.env_origins[:, 2]

    # Observations ----------------------------------------------------------
    def _get_observations(self) -> dict:
        tips, down = self._tips()
        gap = self._max_gap()
        target = self._path_at(self._s.unsqueeze(-1))[0].squeeze(2)
        ahead = self._path_at((self._s + gap).unsqueeze(-1))[0].squeeze(2)
        progress = self._progress()
        prev = self.actions.view(self.num_envs, 2, 4)

        parts = []
        for a, robot in enumerate(self.robots):
            parts += [
                robot.data.joint_pos[:, self._arm_dof_idx],
                robot.data.joint_vel[:, self._arm_dof_idx],
                prev[:, a],
                target[:, a] - tips[:, a],
                ahead[:, a] - tips[:, a],
                progress[:, a : a + 1],
                down[:, a : a + 1],
            ]
        parts += [
            self._spread_dir,
            (self._hold.float() / self.cfg.hold_steps).clamp(max=1.0).unsqueeze(-1),
        ]

        self._tip_markers.visualize((tips + self.scene.env_origins.unsqueeze(1)).reshape(-1, 3))
        return {"policy": torch.cat(parts, dim=-1)}

    # Rewards ---------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        tips, down = self._tips()
        total = self._path_total()
        progress_before = self._progress()

        # ---- Path progress (the task) -------------------------------------
        # Sample the stretch of path the arm could legitimately reach this step
        # and take the furthest sample the tip is actually on. Progress can only
        # extend from where it already is, at a bounded rate.
        gap = self._max_gap()
        ks = torch.linspace(0.0, 1.0, cfg.path_samples + 1, device=self.device)
        s_k = self._s.unsqueeze(-1) + ks * gap.unsqueeze(-1)              # (n, 2, K+1)
        point, leg, _ = self._path_at(s_k)
        dist = torch.linalg.norm(tips.unsqueeze(2) - point, dim=-1)
        tight = leg >= self._tight_from
        corridor = torch.where(
            tight,
            torch.full_like(dist, cfg.tight_corridor),
            torch.full_like(dist, cfg.free_corridor),
        )
        vertical_ok = (down > cfg.opening_vertical_cos_tol).unsqueeze(-1)
        on_k = (dist < corridor) & (~tight | vertical_ok)
        on = on_k.any(dim=-1)

        # Inserting and spreading are continuous in TIME as well: the tip must
        # have been on the path the step before. Without this, dipping in,
        # pulling out and dipping in further along would still count.
        may_k = on_k & (~tight | self._was_on.unsqueeze(-1))
        # The two arms move together: neither may run ahead of the other.
        lead = progress_before - progress_before.flip(dims=(1,))
        may_k = may_k & (lead < cfg.sync_max_lead).unsqueeze(-1)
        order = torch.arange(cfg.path_samples + 1, device=self.device)
        best = (may_k.long() * order).max(dim=-1).values                  # furthest valid
        new_s = torch.gather(s_k, 2, best.unsqueeze(-1)).squeeze(-1).minimum(total)
        new_s = torch.maximum(new_s, self._s)

        in_tight = self._path_at(self._s.unsqueeze(-1))[1].squeeze(-1) >= self._tight_from
        finished = self._s >= total - cfg.path_finish_eps
        broke = self._was_on & (~on) & in_tight & (~finished)
        self._was_on = on
        self._s = new_s
        progress = self._progress()
        gain = (progress - progress_before).clamp(min=0.0)

        reward = cfg.rew_path_progress_scale * gain.mean(dim=-1)
        reward -= cfg.rew_offpath_penalty * broke.float().sum(dim=-1)
        reward -= cfg.rew_sync_penalty * (progress[:, 0] - progress[:, 1]).abs()

        # ---- Dense shaping (kept small on purpose) --------------------------
        # Aim at the point just ahead on the path, which is where the tip has
        # to be for progress to extend.
        gap = self._max_gap()
        ahead = self._path_at((self._s + gap).unsqueeze(-1))[0].squeeze(2)
        d_ahead = torch.linalg.norm(tips - ahead, dim=-1)
        reward += cfg.rew_track_far_scale * (1.0 - torch.tanh(d_ahead / cfg.track_far_eps)).mean(dim=-1)
        near = 1.0 - torch.tanh(d_ahead / cfg.track_fine_eps)
        reward += cfg.rew_track_fine_scale * near.mean(dim=-1)
        # Tool vertical, paid only next to the path and only once the arm is on
        # the leg that comes down over the incision.
        leg_now = self._path_at(self._s.unsqueeze(-1))[1].squeeze(-1)
        over = (leg_now >= self._tight_from - self._sub).float()
        reward += cfg.rew_open_vertical_scale * (down.clamp(min=0.0) * near * over).mean(dim=-1)

        # ---- Holding the opening ---------------------------------------------
        finished = self._s >= total - cfg.path_finish_eps
        d_end = torch.linalg.norm(tips - self._path[:, :, -1], dim=-1)
        holding = finished.all(dim=-1) & (d_end < cfg.hold_tol).all(dim=-1)
        self._hold = torch.where(holding, self._hold + 1, torch.zeros_like(self._hold))
        reward += cfg.rew_hold_scale * holding.float()
        just_completed = (self._hold >= cfg.hold_steps) & (~self._completed)
        reward += cfg.rew_open_complete_bonus * just_completed.float()
        self._completed |= just_completed

        # ---- Safety ------------------------------------------------------------
        tip_dive = (self._tip_floor_z - tips[:, :, 2]).clamp(min=0.0, max=cfg.tip_dive_cap).sum(dim=-1)
        reward -= cfg.rew_tip_dive_penalty_scale * tip_dive
        link_low = (self._surface_z - self._min_link_z()).clamp(min=0.0, max=cfg.link_floor_cap)
        reward -= cfg.rew_link_floor_penalty_scale * link_low
        tip_sep = torch.linalg.norm(tips[:, 0] - tips[:, 1], dim=-1)
        too_close = (cfg.min_tip_separation - tip_sep).clamp(min=0.0) / cfg.min_tip_separation
        reward -= cfg.rew_tip_collision_penalty * too_close

        reward -= cfg.rew_action_rate_penalty * self._action_rate_sq
        reward -= cfg.rew_action_penalty_scale * torch.sum(torch.abs(self.actions), dim=-1)

        width = torch.sum((tips[:, 1, :2] - tips[:, 0, :2]) * self._spread_dir, dim=-1)
        self.extras["log"] = {
            "open_progress": progress.mean(),
            "open_progress_right": progress[:, 0].mean(),
            "open_progress_left": progress[:, 1].mean(),
            "inserted_frac": (leg_now >= self._num_pts - 1 - self._sub).all(dim=-1).float().mean(),
            "finished_frac": finished.all(dim=-1).float().mean(),
            "holding_frac": holding.float().mean(),
            "completed_frac": self._completed.float().mean(),
            "opening_width_m": width.mean(),
            "on_path_frac": on.float().mean(),
            "offpath_break_frac": broke.any(dim=-1).float().mean(),
            "sync_gap": (progress[:, 0] - progress[:, 1]).abs().mean(),
            "track_error_m": d_ahead.mean(),
            "tip_dive_frac": (tip_dive > 0).float().mean(),
            "link_floor_frac": (link_low > 0).float().mean(),
        }
        return reward

    # Dones -----------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        tips, _ = self._tips()
        crashed = (tips[:, :, 2].min(dim=-1).values < self._tip_floor_z - self.cfg.crash_margin) | (
            self._min_link_z() < self._surface_z - self.cfg.crash_margin
        )
        # Wandering far from the path is ended early: nothing useful is learned
        # from the rest of such an episode, and near the path every step pays,
        # so losing the remainder is itself the penalty.
        here = self._path_at(self._s.unsqueeze(-1))[0].squeeze(2)
        strayed = (torch.linalg.norm(tips - here, dim=-1) > self.cfg.stray_limit).any(dim=-1)
        crashed = crashed | strayed
        # Not terminated on completion, for the reason given in incision V3:
        # ending on success forfeits the remaining reward and teaches the policy
        # to avoid finishing. Here it also has to keep holding.
        return crashed, time_out

    # Reset ---------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot_incise._ALL_INDICES
        super()._reset_idx(env_ids)
        env_tensor = self._to_env_tensor(env_ids)

        self._s[env_tensor] = 0.0
        self._was_on[env_tensor] = False
        self._hold[env_tensor] = 0
        self._completed[env_tensor] = False
        self.actions[env_tensor] = 0.0
        self._prev_actions[env_tensor] = 0.0
        self._generate_path(env_tensor)

        # Reference-state initialisation: some episodes start part-way along
        # the taught motion (both arms at the same point of it) instead of at
        # READY. Starting only at READY, a 26-iteration probe never got past 3%
        # of the path -- the arms random-walked out of the corridor in the first
        # seconds and the insert and spread legs were never visited at all.
        m = len(env_tensor)
        start = torch.randint(0, self._num_pts - 1, (m,), device=self.device)
        start = torch.where(
            torch.rand(m, device=self.device) < self.cfg.rsi_prob, start, torch.zeros_like(start)
        )
        self._s[env_tensor] = torch.gather(
            self._cum_len[env_tensor], 2, start.view(m, 1, 1).expand(m, 2, 1)
        ).squeeze(-1)

        for a, robot in enumerate(self.robots):
            # Noise on the four driven joints only; the finger linkage is set
            # consistently, never jittered.
            joint_pos = robot.data.default_joint_pos[env_ids].clone()
            joint_vel = torch.zeros_like(joint_pos)
            joint_pos[:, self._arm5_dof_idx] = self._ref_q[a][start]
            noise = self.cfg.reset_joint_noise * (
                2.0 * torch.rand((len(env_tensor), 4), device=self.device) - 1.0
            )
            joint_pos[:, self._arm_dof_idx] += noise
            joint_pos[:, self._arm_dof_idx] = torch.clamp(
                joint_pos[:, self._arm_dof_idx],
                self._arm_lower[env_tensor],
                self._arm_upper[env_tensor],
            )
            joint_pos[:, self._j5_dof_idx] = self._ref_q[a, -1, 4]
            joint_pos[:, self._grip_dof_idx[0]] = self.cfg.gripper_closed_target
            joint_pos[:, self._finger_dof_idx] = self.robot_incise.data.default_joint_pos[env_ids][
                :, self._finger_dof_idx
            ]

            root = robot.data.default_root_state[env_ids].clone()
            root[:, :3] += self.scene.env_origins[env_ids]
            robot.write_root_pose_to_sim(root[:, :7], env_ids)
            robot.write_root_velocity_to_sim(root[:, 7:], env_ids)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
            self._joint_targets[env_tensor, a] = joint_pos
            robot.set_joint_position_target(joint_pos, env_ids=env_ids)

        self._path_markers.visualize(
            (self._path + self.scene.env_origins.view(-1, 1, 1, 3)).reshape(-1, 3)
        )

    # Helpers -------------------------------------------------------------
    def _to_env_tensor(self, env_ids):
        if isinstance(env_ids, slice):
            start = env_ids.start or 0
            stop = env_ids.stop or self.num_envs
            step = env_ids.step or 1
            return torch.arange(start, stop, step, device=self.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
