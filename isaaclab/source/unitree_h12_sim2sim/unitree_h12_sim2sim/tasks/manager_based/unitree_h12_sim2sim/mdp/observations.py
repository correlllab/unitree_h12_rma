from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from unitree_h12_sim2sim.rma_modules.env_factor_spec import DEFAULT_ET_SPEC

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def _get_or_create_env_buffer(env: ManagerBasedRLEnv, attr_name: str, dim: int) -> torch.Tensor:
    """Returns a per-env buffer stored on the env instance.

    We use this to plumb RMA tensors through IsaacLab's observation terms without tightly coupling to any specific
    runner/policy implementation. The buffers can be populated by future event terms, wrappers, or custom policies.
    """

    buf = getattr(env, attr_name, None)
    if buf is None or not isinstance(buf, torch.Tensor) or buf.shape != (env.num_envs, dim):
        buf = torch.zeros((env.num_envs, dim), device=env.device, dtype=torch.float)
        setattr(env, attr_name, buf)
    return buf


def rma_env_factors(env: ManagerBasedRLEnv, dim: int = DEFAULT_ET_SPEC.dim) -> torch.Tensor:
    """Privileged environment factors e_t (simulation-only, 18 dims for Unitree-H12).

    Default intended ordering:
    [
       payload_downward_force_N (1D),
       leg_strength_scale (12D: multiplicative scale on effort limits for each leg joint, range 0.9–1.1),
       ground_friction_coeff (1D),
       terrain_params (4D: amplitude_m, lengthscale_m, noise_step_m, friction_coeff)
    ]

    This function currently returns a per-env buffer that can be populated by RMA event terms (sample_rma_env_factors).
    """

    return _get_or_create_env_buffer(env, "rma_env_factors_buf", dim)


def rma_extrinsics(env: ManagerBasedRLEnv, dim: int = 8) -> torch.Tensor:
    """Extrinsics latent z_t appended to the policy observation.

    In phase-1, z_t will typically be produced by an env-factor encoder μ(e_t). This observation term provides a
    stable hook for that latent, but defaults to zeros unless populated elsewhere.
    """

    return _get_or_create_env_buffer(env, "rma_extrinsics_buf", dim)