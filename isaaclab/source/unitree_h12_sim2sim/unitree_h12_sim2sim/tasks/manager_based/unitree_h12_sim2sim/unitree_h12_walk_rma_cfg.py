# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RMA-ready variant of the Unitree H12 walking task.

Phase-1 (privileged) training intent:
- The policy consumes an *extrinsics* vector z_t (latent) appended to the usual proprioceptive observation.
- The environment can expose a separate *environment factor* vector e_t (privileged, sim-only) that will later be
  encoded into z_t by the env-factor encoder (implemented under unitree_h12_sim2sim.rma_modules).

This file only changes observation wiring and keeps dynamics identical to Unitree-H12-Walk-v0.
"""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from .unitree_h12_sim2sim_walk_cfg import (
    EventCfg as _BaseEventCfg,
    H12LocomotionFullBodyEnvCfg,
    ObservationsCfg as _BaseObservationsCfg,
)

from unitree_h12_sim2sim.rma_modules.env_factor_spec import DEFAULT_ET_SPEC


RMA_Z_DIM = 8
RMA_ET_DIM = DEFAULT_ET_SPEC.dim


@configclass
class ObservationsRmaCfg(_BaseObservationsCfg):
    """Observation specification extended with RMA extrinsics / env factors."""

    @configclass
    class PolicyCfg(_BaseObservationsCfg.PolicyCfg):
        """Observations for policy group."""

        # RMA latent (z_t) appended to the end of the policy observation.
        # NOTE: For now the env provides a placeholder buffer (zeros) that can be populated later by the policy wrapper.
        rma_extrinsics = ObsTerm(func=mdp.rma_extrinsics, params={"dim": RMA_Z_DIM})

        def __post_init__(self):
            super().__post_init__()
            # keep the same behavior as base policy group
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(_BaseObservationsCfg.CriticCfg):
        """Observations for critic group."""

        # Privileged environment factors (e_t). This is not used by the actor in the default IsaacLab PPO wiring,
        # but is useful to keep available for phase-1 training experiments/logging.
        # Leg-only e_t (strength scaling is 12 dims for the 12 leg joints).
        rma_env_factors = ObsTerm(func=mdp.rma_env_factors, params={"dim": RMA_ET_DIM})

        def __post_init__(self):
            super().__post_init__()

    critic: CriticCfg = CriticCfg()


@configclass
class EventRmaCfg(_BaseEventCfg):
    """Events for RMA phase-1 training.

    Adds an explicit reset-time sampler that fills `env.rma_env_factors_buf`.
    """

    # Disable the base task's startup randomizations for mass/friction so they don't desync from e_t.
    # RMA handles these per-episode at reset via `rma_env_factors`.
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (0.0, 0.0),
            "operation": "add",
        },
    )

    rma_env_factors = EventTerm(
        func=mdp.sample_rma_env_factors,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "payload_mass_range_kg": (-1.0, 5.0),
            "payload_com_range_m": 0.03,
            "leg_strength_range": (0.9, 1.1),
            "friction_range": (0.9, 1.1),
            "apply_to_sim": True,
        },
    )

    # Print once at the first reset (early training episodes can be <5s, so interval prints may not fire).
    rma_debug_print_once = EventTerm(
        func=mdp.print_rma_env_factors_once,
        mode="reset",
        params={
            "max_envs": 2,
            "prefix": "[RMA:e_t][once]",
        },
    )

    # Verify that sampled values are applied in the simulator (best-effort readbacks), once.
    rma_verify_apply_once = EventTerm(
        func=mdp.verify_rma_env_factors_once,
        mode="reset",
        params={
            "max_envs": 2,
            "prefix": "[RMA:verify]",
        },
    )

    rma_debug_print = EventTerm(
        func=mdp.print_rma_env_factors,
        mode="interval",
        # Run every sim step; the function itself gates printing to once every 100 steps.
        interval_range_s=(0.02, 0.02),
        params={
            "max_envs": 2,
            "prefix": "[RMA:e_t]",
        },
    )



@configclass
class CurriculumRmaCfg:
    """Curriculum config for RMA training.

    Enable terrain curriculum so terrain changes are driven by curriculum.
    Keep command curriculum enabled.
    """

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)


@configclass
class H12LocomotionFullBodyRmaEnvCfg(H12LocomotionFullBodyEnvCfg):
    """Configuration for H12 full-body locomotion task with RMA observation extensions."""

    observations: ObservationsRmaCfg = ObservationsRmaCfg()
    events: EventRmaCfg = EventRmaCfg()
    curriculum: CurriculumRmaCfg = CurriculumRmaCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # Ensure terrain generator curriculum is enabled for RMA training.
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = True
