from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from unitree_h12_sim2sim.rma_modules.env_factor_spec import DEFAULT_ET_SPEC, LEG_JOINT_NAMES

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def _ensure_buffer(env: ManagerBasedRLEnv, name: str, dim: int) -> torch.Tensor:
    buf = getattr(env, name, None)
    if buf is None or not isinstance(buf, torch.Tensor) or buf.shape != (env.num_envs, dim):
        buf = torch.zeros((env.num_envs, dim), device=env.device, dtype=torch.float)
        setattr(env, name, buf)
    return buf


def _resolve_asset(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    # SceneEntityCfg in IsaacLab typically uses the attribute name as the key in env.scene.
    return env.scene[asset_cfg.name]


def _get_body_view(asset):
    view = None
    for candidate in ("body_physx_view", "link_physx_view", "_body_physx_view", "_link_physx_view"):
        if hasattr(asset, candidate):
            view = getattr(asset, candidate)
            break
    return view


def _read_body_masses_coms(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, env_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    try:
        asset = _resolve_asset(env, asset_cfg)
    except Exception:
        return None, None

    body_ids = getattr(asset_cfg, "body_ids", None)
    if body_ids is None:
        return None, None
    if isinstance(body_ids, int):
        body_ids = [body_ids]

    view = _get_body_view(asset)
    if view is None:
        return None, None

    try:
        masses = view.get_masses()[env_ids][:, body_ids]
        coms = view.get_coms()[env_ids][:, body_ids, :]
        return masses, coms
    except Exception:
        return None, None


def _read_leg_effort_limits(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, env_ids: torch.Tensor, leg_joint_names: Sequence[str]
) -> tuple[torch.Tensor, list[int]] | tuple[None, None]:
    try:
        asset = _resolve_asset(env, asset_cfg)
    except Exception:
        return None, None

    joint_ids = None
    try:
        if hasattr(asset, "find_joints"):
            joint_ids, _ = asset.find_joints(list(leg_joint_names))
        elif hasattr(asset, "find_joints_by_name"):
            joint_ids = asset.find_joints_by_name(list(leg_joint_names))
    except Exception:
        joint_ids = None

    if joint_ids is None:
        return None, None

    data = getattr(asset, "data", None)
    if data is None or not hasattr(data, "joint_effort_limits"):
        return None, None

    try:
        limits = data.joint_effort_limits[env_ids][:, joint_ids]
        return limits, list(joint_ids)
    except Exception:
        return None, None


def _read_ground_friction(env: ManagerBasedRLEnv) -> float | None:
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None:
        try:
            terrain = env.scene["terrain"]
        except Exception:
            terrain = None
    if terrain is None:
        return None

    for obj in (getattr(terrain, "cfg", None), terrain, getattr(terrain, "physics_material", None)):
        if obj is None:
            continue
        mat = getattr(obj, "physics_material", None)
        if mat is None:
            mat = obj if hasattr(obj, "static_friction") and hasattr(obj, "dynamic_friction") else None
        if mat is None:
            continue
        try:
            return float(getattr(mat, "static_friction"))
        except Exception:
            continue
    return None


def _terrain_encoding_from_env(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> torch.Tensor:
    """Best-effort terrain encoding for bumps-only rough terrain.

    Returns per-env tensor (N, 3):
      [0] amplitude (noise_range max, scaled by vertical_scale if available)
      [1] roughness step (noise_step, scaled by vertical_scale if available)
      [2] curriculum difficulty in [0, 1]
    """

    device = env.device
    num = env_ids.numel()
    enc = torch.zeros((num, DEFAULT_ET_SPEC.terrain_dim), device=device)

    # try to access terrain generator config
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None:
        return enc

    cfg = getattr(terrain, "cfg", None)
    terrain_gen = getattr(cfg, "terrain_generator", None) if cfg is not None else None
    if terrain_gen is None:
        return enc

    # use bumps-only random rough config when available
    sub_terrains = getattr(terrain_gen, "sub_terrains", None)
    random_rough = None
    if isinstance(sub_terrains, dict) and "random_rough" in sub_terrains:
        random_rough = sub_terrains["random_rough"]

    # difficulty based on terrain levels (row index) if available
    level_norm = torch.zeros((num,), device=device)
    levels = getattr(terrain, "terrain_levels", None)
    num_rows = getattr(terrain_gen, "num_rows", None)
    if levels is not None and num_rows is not None and num_rows > 1:
        try:
            level_vals = levels[env_ids].float().to(device=device)
            level_norm = torch.clamp(level_vals / float(num_rows - 1), 0.0, 1.0)
        except Exception:
            pass

    # map difficulty range if provided
    difficulty_range = getattr(terrain_gen, "difficulty_range", None)
    if difficulty_range is not None and len(difficulty_range) == 2:
        try:
            d0, d1 = float(difficulty_range[0]), float(difficulty_range[1])
            level_norm = d0 + (d1 - d0) * level_norm
        except Exception:
            pass

    # amplitude + roughness step (best-effort)
    if random_rough is not None:
        noise_range = getattr(random_rough, "noise_range", None)
        noise_step = getattr(random_rough, "noise_step", None)
        vertical_scale = getattr(terrain_gen, "vertical_scale", 1.0)

        try:
            if noise_range is not None and len(noise_range) >= 2:
                amp = float(noise_range[1]) * float(vertical_scale)
                enc[:, 0] = amp
        except Exception:
            pass

        try:
            if noise_step is not None:
                enc[:, 1] = float(noise_step) * float(vertical_scale)
        except Exception:
            pass

    # difficulty encoding
    enc[:, 2] = level_norm
    return enc


def _maybe_cache_baselines(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, env_ids: torch.Tensor) -> None:
    if not hasattr(env, "_rma_baseline_masses") or not hasattr(env, "_rma_baseline_coms"):
        masses, coms = _read_body_masses_coms(env, asset_cfg, env_ids)
        if masses is not None and coms is not None:
            env._rma_baseline_masses = masses.clone()
            env._rma_baseline_coms = coms.clone()

    if not hasattr(env, "_rma_baseline_effort_limits"):
        limits, joint_ids = _read_leg_effort_limits(env, asset_cfg, env_ids, LEG_JOINT_NAMES)
        if limits is not None and joint_ids is not None:
            env._rma_baseline_effort_limits = limits.clone()
            env._rma_baseline_effort_joint_ids = joint_ids

    if not hasattr(env, "_rma_baseline_friction"):
        mu = _read_ground_friction(env)
        if mu is not None:
            env._rma_baseline_friction = mu


def _try_set_body_mass_and_com(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    env_ids: torch.Tensor,
    mass_add_kg: torch.Tensor,
    com_xy_m: torch.Tensor,
) -> None:
    """Best-effort: apply mass + COM offset to the specified bodies.

    This uses PhysX view methods when available. If APIs differ across IsaacLab versions, we silently no-op.
    """

    try:
        asset = _resolve_asset(env, asset_cfg)
    except Exception:
        return

    body_ids = getattr(asset_cfg, "body_ids", None)
    if body_ids is None:
        return

    # We expect torso_link only (1 body id), but support multiple.
    if isinstance(body_ids, int):
        body_ids = [body_ids]

    view = _get_body_view(asset)
    if view is None:
        return

    # Apply mass
    try:
        masses = view.get_masses()
        # masses: (num_envs, num_bodies)
        masses_env = masses[env_ids][:, body_ids]
        masses_env = masses_env + mass_add_kg.unsqueeze(-1)
        masses[env_ids[:, None], torch.tensor(body_ids, device=env.device)] = masses_env
        # best-effort setter
        if hasattr(view, "set_masses"):
            view.set_masses(masses)
    except Exception:
        pass

    # Apply COM offset in x/y (keep z unchanged)
    try:
        coms = view.get_coms()
        coms_env = coms[env_ids][:, body_ids, :]
        coms_env[..., 0:2] = coms_env[..., 0:2] + com_xy_m.unsqueeze(1)
        coms[env_ids[:, None], torch.tensor(body_ids, device=env.device)] = coms_env
        if hasattr(view, "set_coms"):
            view.set_coms(coms)
    except Exception:
        pass


def _try_set_leg_effort_limits(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    env_ids: torch.Tensor,
    leg_strength_scale: torch.Tensor,
    leg_joint_names: Sequence[str],
) -> None:
    """Best-effort: scale per-joint effort/torque limits for leg joints.

    Different IsaacLab versions expose different access patterns; we try a few common ones.
    If we cannot apply, we silently no-op (but still store e_t).
    """

    try:
        asset = _resolve_asset(env, asset_cfg)
    except Exception:
        return

    # Resolve joint ids
    joint_ids = None
    try:
        if hasattr(asset, "find_joints"):
            joint_ids, _ = asset.find_joints(list(leg_joint_names))
        elif hasattr(asset, "find_joints_by_name"):
            joint_ids = asset.find_joints_by_name(list(leg_joint_names))
    except Exception:
        joint_ids = None

    if joint_ids is None:
        return

    # Try to access joint effort limits
    try:
        data = getattr(asset, "data", None)
        if data is not None and hasattr(data, "joint_effort_limits"):
            limits = data.joint_effort_limits
            limits_env = limits[env_ids][:, joint_ids]
            limits_env = limits_env * leg_strength_scale
            limits[env_ids[:, None], torch.tensor(joint_ids, device=env.device)] = limits_env
            if hasattr(asset, "write_joint_effort_limits_to_sim"):
                asset.write_joint_effort_limits_to_sim(limits)
            return
    except Exception:
        pass


def _try_set_ground_friction(env: "ManagerBasedRLEnv", friction_coeff: torch.Tensor) -> None:
    """Best-effort: apply friction to the ground material.

    Many IsaacLab setups use a single shared ground physics material across all envs. We therefore apply the mean
    friction across the just-reset env_ids (still provides episode-to-episode variation).
    """

    try:
        friction_val = float(torch.mean(friction_coeff).item())
    except Exception:
        return

    # Try scene.terrain first
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None:
        try:
            terrain = env.scene["terrain"]
        except Exception:
            terrain = None

    if terrain is None:
        return

    # Common locations for the physics material
    for obj in (getattr(terrain, "cfg", None), terrain, getattr(terrain, "physics_material", None)):
        if obj is None:
            continue
        mat = getattr(obj, "physics_material", None)
        if mat is None:
            mat = obj if hasattr(obj, "static_friction") and hasattr(obj, "dynamic_friction") else None
        if mat is None:
            continue
        try:
            mat.static_friction = friction_val
            mat.dynamic_friction = friction_val
            return
        except Exception:
            continue


def sample_rma_env_factors(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    *,
    payload_mass_range_kg: tuple[float, float] = (0, 3.0),
    payload_com_range_m: float = 0.03,
    leg_strength_range: tuple[float, float] = (0.9, 1.1),
    friction_range: tuple[float, float] = (0.9, 1.1),
    apply_to_sim: bool = True,
) -> None:
    """Sample + store privileged e_t, and best-effort apply to the simulator.

    Stores into `env.rma_env_factors_buf` with the ordering defined by DEFAULT_ET_SPEC.

    Args:
        env_ids: environments being reset.
        asset_cfg: Scene entity (robot) used for applying payload/strength.
        apply_to_sim: if True, attempts to apply mass/COM and leg effort limit scaling.
    """

    device = env.device
    env_ids = env_ids.to(device=device)
    num = env_ids.numel()

    et = _ensure_buffer(env, "rma_env_factors_buf", DEFAULT_ET_SPEC.dim)

    # --- sample
    payload_mass_add = torch.empty((num,), device=device).uniform_(*payload_mass_range_kg)
    payload_com_xy = torch.empty((num, 2), device=device).uniform_(-payload_com_range_m, payload_com_range_m)

    leg_strength = torch.empty((num, DEFAULT_ET_SPEC.leg_strength_dim), device=device).uniform_(*leg_strength_range)

    # friction is a single scalar factor; we store it but applying is terrain/material dependent.
    friction = torch.empty((num,), device=device).uniform_(*friction_range)

    # Coarse terrain descriptors: bumps-only encoding when available.
    terrain = _terrain_encoding_from_env(env, env_ids)

    # --- pack
    et_env = et[env_ids]
    et_env[:, 0] = payload_mass_add
    et_env[:, 1:3] = payload_com_xy
    et_env[:, DEFAULT_ET_SPEC.leg_strength_slice] = leg_strength
    et_env[:, DEFAULT_ET_SPEC.friction_slice] = friction.unsqueeze(-1)
    et_env[:, DEFAULT_ET_SPEC.terrain_slice] = terrain
    et[env_ids] = et_env

    # Cache baselines before any modifications (best-effort)
    env._rma_asset_cfg = asset_cfg
    _maybe_cache_baselines(env, asset_cfg, env_ids)

    # --- apply (best-effort)
    if apply_to_sim:
        _try_set_body_mass_and_com(env, asset_cfg, env_ids, payload_mass_add, payload_com_xy)
        _try_set_leg_effort_limits(env, asset_cfg, env_ids, leg_strength, LEG_JOINT_NAMES)
        _try_set_ground_friction(env, friction)

    # Keep a copy for debugging
    env.rma_payload_mass_add_kg = payload_mass_add
    env.rma_payload_com_xy_m = payload_com_xy
    env.rma_leg_strength_scale = leg_strength
    env.rma_ground_friction_coeff = friction


def verify_rma_env_factors(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None,
    max_envs: int,
    prefix: str,
) -> None:
    """Verify whether sampled e_t factors appear applied in the simulator (best-effort)."""

    def _emit(msg: str) -> None:
        print(msg, flush=True)
        try:
            import omni.log  # type: ignore

            omni.log.info(msg)
        except Exception:
            pass

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    else:
        env_ids = env_ids.to(device=env.device)
    if env_ids.numel() == 0:
        return

    n_show = int(min(max_envs, env_ids.numel()))
    env_ids_show = env_ids[:n_show]

    # Sampled values (from e_t)
    et = getattr(env, "rma_env_factors_buf", None)
    if et is None or not isinstance(et, torch.Tensor):
        _emit(f"{prefix} e_t buffer missing; cannot verify applied values.")
        return

    et_sel = et[env_ids_show]
    payload = et_sel[:, DEFAULT_ET_SPEC.payload_slice]
    leg_strength = et_sel[:, DEFAULT_ET_SPEC.leg_strength_slice]
    friction = et_sel[:, DEFAULT_ET_SPEC.friction_slice].squeeze(-1)

    def _stats(x: torch.Tensor) -> tuple[float, float, float]:
        return float(x.mean().item()), float(x.min().item()), float(x.max().item())

    m_mass_add, mi_mass_add, ma_mass_add = _stats(payload[:, 0])
    m_cx_add, mi_cx_add, ma_cx_add = _stats(payload[:, 1])
    m_cy_add, mi_cy_add, ma_cy_add = _stats(payload[:, 2])
    m_ls, mi_ls, ma_ls = _stats(leg_strength)
    m_mu, mi_mu, ma_mu = _stats(friction)

    _emit(
        f"{prefix} sampled e_t | mass_add={m_mass_add:.3f}/{mi_mass_add:.3f}/{ma_mass_add:.3f} kg | "
        f"com_x_add={m_cx_add:.3f}/{mi_cx_add:.3f}/{ma_cx_add:.3f} m | "
        f"com_y_add={m_cy_add:.3f}/{mi_cy_add:.3f}/{ma_cy_add:.3f} m | "
        f"leg_strength={m_ls:.3f}/{mi_ls:.3f}/{ma_ls:.3f} | friction={m_mu:.3f}/{mi_mu:.3f}/{ma_mu:.3f}"
    )

    # Readbacks (best-effort)
    asset_cfg = getattr(env, "_rma_asset_cfg", None)
    if asset_cfg is None:
        _emit(f"{prefix} no asset_cfg cached; cannot read back masses/effort limits.")
        return

    masses, coms = _read_body_masses_coms(env, asset_cfg, env_ids_show)
    if masses is not None and coms is not None:
        m_mass, mi_mass, ma_mass = _stats(masses.reshape(-1))
        m_cx, mi_cx, ma_cx = _stats(coms[..., 0].reshape(-1))
        m_cy, mi_cy, ma_cy = _stats(coms[..., 1].reshape(-1))
        _emit(
            f"{prefix} readback torso mass/com (absolute) | mass={m_mass:.3f}/{mi_mass:.3f}/{ma_mass:.3f} kg | "
            f"com_x={m_cx:.3f}/{mi_cx:.3f}/{ma_cx:.3f} m | com_y={m_cy:.3f}/{mi_cy:.3f}/{ma_cy:.3f} m"
        )

        if hasattr(env, "_rma_baseline_masses"):
            baseline = env._rma_baseline_masses
            if baseline is not None and isinstance(baseline, torch.Tensor) and baseline.shape == masses.shape:
                expected = baseline + payload[:, 0:1]
                err = masses - expected
                m_e, mi_e, ma_e = _stats(err.reshape(-1))
                _emit(f"{prefix} mass delta error (actual - expected) mean/min/max={m_e:.3f}/{mi_e:.3f}/{ma_e:.3f} kg")

        if hasattr(env, "_rma_baseline_coms"):
            baseline_c = env._rma_baseline_coms
            if baseline_c is not None and isinstance(baseline_c, torch.Tensor) and baseline_c.shape == coms.shape:
                expected_c = baseline_c.clone()
                expected_c[..., 0:2] = expected_c[..., 0:2] + payload[:, 1:3].unsqueeze(1)
                err = coms - expected_c
                m_e, mi_e, ma_e = _stats(err[..., 0:2].reshape(-1))
                _emit(f"{prefix} com_xy delta error (actual - expected) mean/min/max={m_e:.4f}/{mi_e:.4f}/{ma_e:.4f} m")
    else:
        _emit(f"{prefix} could not read back torso mass/COM.")

    limits, _ = _read_leg_effort_limits(env, asset_cfg, env_ids_show, LEG_JOINT_NAMES)
    if limits is not None:
        m_l, mi_l, ma_l = _stats(limits.reshape(-1))
        _emit(
            f"{prefix} readback leg effort limits (absolute) mean/min/max={m_l:.3f}/{mi_l:.3f}/{ma_l:.3f}"
        )

        if hasattr(env, "_rma_baseline_effort_limits"):
            baseline_l = env._rma_baseline_effort_limits
            if baseline_l is not None and isinstance(baseline_l, torch.Tensor) and baseline_l.shape == limits.shape:
                expected_l = baseline_l * leg_strength
                err = limits - expected_l
                m_e, mi_e, ma_e = _stats(err.reshape(-1))
                _emit(f"{prefix} effort limit delta error (actual - expected) mean/min/max={m_e:.3f}/{mi_e:.3f}/{ma_e:.3f}")
    else:
        _emit(f"{prefix} could not read back leg effort limits.")

    mu = _read_ground_friction(env)
    if mu is not None:
        _emit(f"{prefix} readback ground friction (absolute)={mu:.3f}")
        if hasattr(env, "_rma_baseline_friction"):
            baseline_mu = env._rma_baseline_friction
            if baseline_mu is not None:
                expected_mu = float(torch.mean(friction).item())
                _emit(
                    f"{prefix} friction delta (actual - expected mean)={mu - expected_mu:+.4f} (baseline={baseline_mu:.3f})"
                )
    else:
        _emit(f"{prefix} could not read back ground friction.")


def verify_rma_env_factors_once(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None,
    max_envs: int,
    prefix: str,
) -> None:
    """Run verification only once before training progresses."""

    if getattr(env, "_rma_verify_once_done", False):
        return
    setattr(env, "_rma_verify_once_done", True)
    verify_rma_env_factors(env, env_ids, max_envs, prefix)


def print_rma_env_factors(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None,
    max_envs: int,
    prefix: str,
) -> None:
    """Periodic debug printer for the privileged e_t buffer.

    Intended to be used with an IsaacLab EventTerm in `mode="interval"`.
    """

    def _emit(msg: str) -> None:
        """Emit to terminal stdout (and also to Isaac Sim log when available)."""

        print(msg, flush=True)
        try:
            import omni.log  # type: ignore

            omni.log.info(msg)
        except Exception:
            pass

    et = getattr(env, "rma_env_factors_buf", None)
    if et is None or not isinstance(et, torch.Tensor):
        _emit(f"{prefix} e_t buffer not found yet (env.rma_env_factors_buf is missing).")
        return

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    else:
        env_ids = env_ids.to(device=env.device)

    if env_ids.numel() == 0:
        return

    # Aggregate over the env_ids provided by the event manager.
    et_sel = et[env_ids]
    payload = et_sel[:, DEFAULT_ET_SPEC.payload_slice]
    leg_strength = et_sel[:, DEFAULT_ET_SPEC.leg_strength_slice]
    friction = et_sel[:, DEFAULT_ET_SPEC.friction_slice]

    # A couple of env samples for quick sanity checking.
    n_show = int(min(max_envs, env_ids.numel()))
    env_ids_show = env_ids[:n_show].tolist()
    samples_show = et_sel[:n_show].detach().cpu().numpy()

    def _stats(x: torch.Tensor) -> tuple[float, float, float]:
        return float(x.mean().item()), float(x.min().item()), float(x.max().item())

    m_mass, mi_mass, ma_mass = _stats(payload[:, 0])
    m_cx, mi_cx, ma_cx = _stats(payload[:, 1])
    m_cy, mi_cy, ma_cy = _stats(payload[:, 2])
    m_ls, mi_ls, ma_ls = _stats(leg_strength)
    m_mu, mi_mu, ma_mu = _stats(friction)

    # Step counter (best-effort): print only every 100 steps.
    step_val = (
        getattr(env, "common_step_counter", None)
        or getattr(env, "_step_count", None)
        or getattr(env, "step_count", None)
    )
    step_int: int | None = None
    try:
        if isinstance(step_val, torch.Tensor):
            step_int = int(step_val.item())
        elif step_val is not None:
            step_int = int(step_val)
    except Exception:
        step_int = None

    if step_int is None:
        step_int = int(getattr(env, "_rma_debug_step_counter", 0)) + 1
        setattr(env, "_rma_debug_step_counter", step_int)

    # Don't spam on step 0, and only print every 100 steps.
    if step_int == 0 or (step_int % 100) != 0:
        return

    step_str = f" step={step_int}"

    _emit(
        f"{prefix}{step_str} e_t stats over {env_ids.numel()} envs | "
        f"mass(mean/min/max)={m_mass:.3f}/{mi_mass:.3f}/{ma_mass:.3f} kg | "
        f"com_x={m_cx:.3f}/{mi_cx:.3f}/{ma_cx:.3f} m | "
        f"com_y={m_cy:.3f}/{mi_cy:.3f}/{ma_cy:.3f} m | "
        f"leg_strength={m_ls:.3f}/{mi_ls:.3f}/{ma_ls:.3f} | "
        f"friction={m_mu:.3f}/{mi_mu:.3f}/{ma_mu:.3f}"
    )
    _emit(f"{prefix} sample env_ids={env_ids_show} e_t[:{n_show}]={samples_show}")


def rma_change_terrain_interval(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None,
    step_interval: int = 100,
    change_prob: float = 0.5,
    do_reset: bool = True,
) -> None:
    """Best-effort terrain change and e_t terrain-slice refresh on an interval.

    This mirrors the visualization behavior but runs inside the training pipeline.
    """

    if step_interval <= 0:
        return

    step_val = (
        getattr(env, "common_step_counter", None)
        or getattr(env, "_step_count", None)
        or getattr(env, "step_count", None)
    )
    step_int: int | None = None
    try:
        if isinstance(step_val, torch.Tensor):
            step_int = int(step_val.item())
        elif step_val is not None:
            step_int = int(step_val)
    except Exception:
        step_int = None

    if step_int is None:
        step_int = int(getattr(env, "_rma_terrain_step_counter", 0)) + 1
        setattr(env, "_rma_terrain_step_counter", step_int)

    if step_int == 0 or (step_int % step_interval) != 0:
        return

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    else:
        env_ids = env_ids.to(device=env.device)
    if env_ids.numel() == 0:
        return

    terrain = getattr(env.scene, "terrain", None)
    if terrain is None or not hasattr(terrain, "update_env_origins"):
        return

    move_up = torch.rand((env_ids.numel(),), device=env.device) < float(change_prob)
    move_down = torch.zeros_like(move_up, dtype=torch.bool)
    try:
        terrain.update_env_origins(env_ids, move_up, move_down)
    except Exception:
        return

    if do_reset:
        try:
            env.reset(env_ids)
        except Exception:
            pass

    # Refresh terrain encoding slice in e_t (if buffer exists).
    et = getattr(env, "rma_env_factors_buf", None)
    if et is None or not isinstance(et, torch.Tensor):
        return

    enc = _terrain_encoding_from_env(env, env_ids)
    if enc is not None:
        et[env_ids, DEFAULT_ET_SPEC.terrain_slice] = enc


def print_rma_env_factors_once(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None,
    max_envs: int,
    prefix: str,
) -> None:
    """Print e_t once (first time invoked), then stay silent.

    Useful during early training when episodes are short, so interval-based prints may never fire.
    """

    if getattr(env, "_rma_debug_print_once_done", False):
        return
    setattr(env, "_rma_debug_print_once_done", True)
    print_rma_env_factors(env, env_ids, max_envs, prefix)
