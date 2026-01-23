# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Visualize and verify RMA e_t parameters before training.

Run with IsaacLab launcher, e.g.:
  /home/niraj/isaac_projects/IsaacLab/isaaclab.sh -p wbc_agile_utils/visualize_rma_params.py \
    --task Unitree-H12-Walk-RMA-v0 --num_envs 1
"""

# flake8: noqa

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Visualize RMA env-factor effects in simulation.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--steps_per_setting", type=int, default=300, help="Steps to run per setting.")
parser.add_argument("--print_interval", type=int, default=50, help="Steps between terrain encoding prints.")
parser.add_argument(
    "--terrain_change_interval",
    type=int,
    default=100,
    help="Steps between forced terrain changes (0 disables).",
)
parser.add_argument(
    "--terrain_change_prob",
    type=float,
    default=0.5,
    help="Probability of moving to a harder vs easier terrain when changing.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.utils import parse_env_cfg

import unitree_h12_sim2sim  # noqa: F401

from unitree_h12_sim2sim.rma_modules.env_factor_spec import DEFAULT_ET_SPEC
from unitree_h12_sim2sim.tasks.manager_based.unitree_h12_sim2sim import mdp
from unitree_h12_sim2sim.tasks.manager_based.unitree_h12_sim2sim.mdp import rma as rma_mdp


def _apply_manual_et(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    *,
    mass_add_kg: float,
    com_xy_m: tuple[float, float],
    leg_strength_scale: float,
    friction: float,
) -> None:
    """Apply a deterministic e_t to the simulator (best-effort) and store it in the buffer."""

    env_ids = env_ids.to(device=env.device)
    num = env_ids.numel()

    # store buffer
    et = rma_mdp._ensure_buffer(env, "rma_env_factors_buf", DEFAULT_ET_SPEC.dim)
    et_env = et[env_ids]
    et_env[:, 0] = mass_add_kg
    et_env[:, 1] = com_xy_m[0]
    et_env[:, 2] = com_xy_m[1]
    et_env[:, DEFAULT_ET_SPEC.leg_strength_slice] = leg_strength_scale
    et_env[:, DEFAULT_ET_SPEC.friction_slice] = friction
    # encode terrain (amp, roughness, difficulty)
    terrain_enc = rma_mdp._terrain_encoding_from_env(env, env_ids)
    et_env[:, DEFAULT_ET_SPEC.terrain_slice] = terrain_enc
    et[env_ids] = et_env

    # cache asset_cfg and baselines
    env._rma_asset_cfg = asset_cfg
    rma_mdp._maybe_cache_baselines(env, asset_cfg, env_ids)

    # apply best-effort
    mass_add = torch.full((num,), mass_add_kg, device=env.device)
    com_xy = torch.tensor([com_xy_m], device=env.device).repeat(num, 1)
    leg_scale = torch.full((num, DEFAULT_ET_SPEC.leg_strength_dim), leg_strength_scale, device=env.device)
    fric = torch.full((num,), friction, device=env.device)

    rma_mdp._try_set_body_mass_and_com(env, asset_cfg, env_ids, mass_add, com_xy)
    rma_mdp._try_set_leg_effort_limits(env, asset_cfg, env_ids, leg_scale, rma_mdp.LEG_JOINT_NAMES)
    rma_mdp._try_set_ground_friction(env, fric)


# parse configuration
env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
# disable automatic RMA sampling/printing events for this visualization
if hasattr(env_cfg, "events") and env_cfg.events is not None:
    for name in ("rma_env_factors", "rma_debug_print", "rma_debug_print_once", "rma_verify_apply_once"):
        if hasattr(env_cfg.events, name):
            delattr(env_cfg.events, name)
# enable terrain debug visualization for this run
if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "terrain"):
    env_cfg.scene.terrain.debug_vis = True
# create isaac environment
env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
env_unwrapped = env.unwrapped

# resolve robot asset
asset_cfg = SceneEntityCfg("robot", body_names="torso_link")
asset_cfg.resolve(env_unwrapped.scene)

# prepare actions
action_dim = int(getattr(env_unwrapped.action_manager, "total_action_dim", env.action_space.shape[0]))
zero_actions = torch.zeros((env_unwrapped.num_envs, action_dim), device=env_unwrapped.device)

# simple sweep of settings (env 0)
settings = [
    {"name": "baseline", "mass_add_kg": 0.0, "com_xy_m": (0.0, 0.0), "leg_strength_scale": 1.0, "friction": 1.0},
    {"name": "heavy_load", "mass_add_kg": 4.0, "com_xy_m": (0.02, -0.02), "leg_strength_scale": 1.0, "friction": 1.0},
    {"name": "weak_legs", "mass_add_kg": 0.0, "com_xy_m": (0.0, 0.0), "leg_strength_scale": 0.9, "friction": 1.0},
    {"name": "slippery", "mass_add_kg": 0.0, "com_xy_m": (0.0, 0.0), "leg_strength_scale": 1.0, "friction": 0.9},
]

# reset env
env.reset()

# run sweep
for setting in settings:
    print(f"\n[RMA:viz] Applying setting: {setting['name']}")
    env_ids = torch.tensor([0], device=env_unwrapped.device)
    _apply_manual_et(
        env_unwrapped,
        env_ids,
        asset_cfg,
        mass_add_kg=setting["mass_add_kg"],
        com_xy_m=setting["com_xy_m"],
        leg_strength_scale=setting["leg_strength_scale"],
        friction=setting["friction"],
    )
    mdp.verify_rma_env_factors(env_unwrapped, env_ids, max_envs=1, prefix="[RMA:verify]")
    # print terrain encoding for this env
    terrain_enc = rma_mdp._terrain_encoding_from_env(env_unwrapped, env_ids)
    print(f"[RMA:terrain] enc={terrain_enc.detach().cpu().numpy()}")

    for step_idx in range(args_cli.steps_per_setting):
        if not simulation_app.is_running():
            break
        step_out = env.step(zero_actions)
        # handle both gymnasium (5-tuple) and older gym (4-tuple)
        if isinstance(step_out, tuple) and len(step_out) == 5:
            _, _, terminated, truncated, _ = step_out
            done = terminated | truncated
        elif isinstance(step_out, tuple) and len(step_out) == 4:
            _, _, done, _ = step_out
        else:
            done = None

        if done is not None:
            done_tensor = torch.as_tensor(done, device=env_unwrapped.device)
            if done_tensor.any():
                reset_ids = torch.nonzero(done_tensor).flatten()
                _apply_manual_et(
                    env_unwrapped,
                    reset_ids,
                    asset_cfg,
                    mass_add_kg=setting["mass_add_kg"],
                    com_xy_m=setting["com_xy_m"],
                    leg_strength_scale=setting["leg_strength_scale"],
                    friction=setting["friction"],
                )

        # Force terrain changes over time (best-effort): update terrain origins then reset.
        if args_cli.terrain_change_interval > 0 and (step_idx % args_cli.terrain_change_interval) == 0:
            terrain = getattr(env_unwrapped.scene, "terrain", None)
            if terrain is not None and hasattr(terrain, "update_env_origins"):
                move_up = torch.rand_like(env_ids.float()) < float(args_cli.terrain_change_prob)
                move_down = ~move_up
                try:
                    terrain.update_env_origins(env_ids, move_up, move_down)
                    env.reset()
                    _apply_manual_et(
                        env_unwrapped,
                        env_ids,
                        asset_cfg,
                        mass_add_kg=setting["mass_add_kg"],
                        com_xy_m=setting["com_xy_m"],
                        leg_strength_scale=setting["leg_strength_scale"],
                        friction=setting["friction"],
                    )
                except Exception:
                    pass

        # Show terrain changes over time
        if args_cli.print_interval > 0 and (step_idx % args_cli.print_interval) == 0:
            enc = rma_mdp._terrain_encoding_from_env(env_unwrapped, env_ids)
            # update terrain slice in e_t so debug prints reflect changes
            et = rma_mdp._ensure_buffer(env_unwrapped, "rma_env_factors_buf", DEFAULT_ET_SPEC.dim)
            et_env = et[env_ids]
            et_env[:, DEFAULT_ET_SPEC.terrain_slice] = enc
            et[env_ids] = et_env
            print(f"[RMA:terrain] step={step_idx} enc={enc.detach().cpu().numpy()}")

# close the simulator
env.close()
simulation_app.close()
