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

    # Try common PhysX view handles
    view = None
    for candidate in ("body_physx_view", "link_physx_view", "_body_physx_view", "_link_physx_view"):
        if hasattr(asset, candidate):
            view = getattr(asset, candidate)
            break
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
    payload_mass_range_kg: tuple[float, float] = (-1.0, 5.0),
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

    # Coarse terrain descriptors: placeholder zeros for now.
    terrain = torch.zeros((num, DEFAULT_ET_SPEC.terrain_dim), device=device)

    # --- pack
    et_env = et[env_ids]
    et_env[:, 0] = payload_mass_add
    et_env[:, 1:3] = payload_com_xy
    et_env[:, DEFAULT_ET_SPEC.leg_strength_slice] = leg_strength
    et_env[:, DEFAULT_ET_SPEC.friction_slice] = friction.unsqueeze(-1)
    et_env[:, DEFAULT_ET_SPEC.terrain_slice] = terrain
    et[env_ids] = et_env

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
        """Emit log message in Isaac Sim (preferred), otherwise stdout."""

        try:
            import omni.log  # type: ignore

            omni.log.info(msg)
        except Exception:
            print(msg, flush=True)

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

    # Step counter (best-effort)
    step = (
        getattr(env, "common_step_counter", None)
        or getattr(env, "_step_count", None)
        or getattr(env, "step_count", None)
    )
    step_str = f" step={step}" if step is not None else ""

    _emit(
        f"{prefix}{step_str} e_t stats over {env_ids.numel()} envs | "
        f"mass(mean/min/max)={m_mass:.3f}/{mi_mass:.3f}/{ma_mass:.3f} kg | "
        f"com_x={m_cx:.3f}/{mi_cx:.3f}/{ma_cx:.3f} m | "
        f"com_y={m_cy:.3f}/{mi_cy:.3f}/{ma_cy:.3f} m | "
        f"leg_strength={m_ls:.3f}/{mi_ls:.3f}/{ma_ls:.3f} | "
        f"friction={m_mu:.3f}/{mi_mu:.3f}/{ma_mu:.3f}"
    )
    _emit(f"{prefix} sample env_ids={env_ids_show} e_t[:{n_show}]={samples_show}")
