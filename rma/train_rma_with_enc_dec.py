# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Script to train RMA policy with encoder and decoder jointly."""

"""Launch Isaac Sim Simulator first."""

# flake8: noqa

import argparse
import sys
from pathlib import Path


# Add isaaclab scripts directory to path for cli_args import
sys.path.insert(0, str(Path(__file__).parent.parent / "isaaclab" / "scripts" / "rsl_rl"))

# Ensure environment registration is executed before training
import isaaclab.source.unitree_h12_sim2sim.unitree_h12_sim2sim.tasks.manager_based.unitree_h12_sim2sim  # noqa: F401

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train RMA RL agent with encoder and decoder.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--video_interval_iter",
    type=int,
    default=100,
    help="Interval between video recordings (in training iterations).",
)
parser.add_argument(
    "--video_interval_steps",
    type=int,
    default=1000,
    help="Interval between video recordings (in environment steps). Overrides --video_interval_iter if > 0.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--encoder_weight", type=float, default=0.01, help="Weight for encoder loss in joint loss.")
parser.add_argument("--decoder_weight", type=float, default=0.1, help="Weight for decoder loss in joint loss.")
parser.add_argument("--checkpoint_interval", type=int, default=100, help="Save encoder/decoder checkpoints every N iterations.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime
from pathlib import Path

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import agile.isaaclab_extras.monkey_patches


# Ensure rma/ is in sys.path for rma_modules import
sys.path.insert(0, str(Path(__file__).parent))
from rma_modules import (
    EnvFactorEncoder,
    EnvFactorEncoderCfg,
    EnvFactorDecoder,
    EnvFactorDecoderCfg,
)

from agile.rl_env.rsl_rl import (  # isort: skip
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


class EnvFactorNormalizer:
    """Normalizes environment factors to [0, 1] range (force, leg_strength, friction only, 14 dims)."""
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.mins = torch.tensor([
            0.0,      # payload force
            *([0.9] * 12),  # leg strengths
            0.0,      # friction
        ], device=device, dtype=torch.float32)
        self.maxs = torch.tensor([
            50.0,     # payload force
            *([1.1] * 12),  # leg strengths
            1.0,      # friction
        ], device=device, dtype=torch.float32)
        self.ranges = self.maxs - self.mins
    def normalize(self, e_t: torch.Tensor) -> torch.Tensor:
        return (e_t - self.mins) / (self.ranges + 1e-8)
    def denormalize(self, e_t_normalized: torch.Tensor) -> torch.Tensor:
        ranges = self.ranges.to(e_t_normalized.device)
        mins = self.mins.to(e_t_normalized.device)
        return e_t_normalized * ranges + mins


class RMATrainerJoint:
    """Joint training of policy, encoder, and decoder."""
    
    def __init__(
        self,
        runner: OnPolicyRunner,
        encoder: EnvFactorEncoder,
        decoder: EnvFactorDecoder,
        log_dir: str,
        device: str,
        encoder_weight: float = 0.01,
        decoder_weight: float = 0.1,
        checkpoint_interval: int = 100,
    ):
        """Initialize trainer.
        
        Args:
            runner: RSL-RL OnPolicyRunner
            encoder: Environment factor encoder
            decoder: Environment factor decoder
            log_dir: Logging directory
            device: Torch device
            encoder_weight: Weight for encoder loss in total loss
            decoder_weight: Weight for decoder loss in total loss
            checkpoint_interval: Save checkpoints every N policy iterations
        """
        self.runner = runner
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.log_dir = log_dir
        self.device = device
        self.normalizer = EnvFactorNormalizer(device=device)
        
        self.encoder_weight = encoder_weight
        self.decoder_weight = decoder_weight
        self.checkpoint_interval = checkpoint_interval
        
        # Optimizers
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-4)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=1e-3)
        
        # Create checkpoint directories
        encoder_dir = Path(log_dir) / "checkpoints" / "encoder"
        decoder_dir = Path(log_dir) / "checkpoints" / "decoder"
        encoder_dir.mkdir(parents=True, exist_ok=True)
        decoder_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.encoder_losses = []
        self.decoder_losses = []
    
    def extract_env_factors(self) -> torch.Tensor:
        """Extract environment factors from environment.
        Returns:
            Tensor of shape (num_envs, 14), or None if unable to extract.
        """
        env = self.runner.env
        env_to_check = env
        
        # Unwrap RslRlVecEnvWrapper
        if hasattr(env_to_check, "env"):
            env_to_check = env_to_check.env
        
        # Unwrap gym wrappers
        while hasattr(env_to_check, "env"):
            env_to_check = env_to_check.env
        
        # Check for rma_env_factors in scene
        if hasattr(env_to_check, "scene") and hasattr(env_to_check.scene, "rma_env_factors"):
            e_t = env_to_check.scene.rma_env_factors
            if isinstance(e_t, torch.Tensor) and e_t.shape[0] > 0:
                return e_t.clone().to(self.device)
        
        # Check attributes for rma_env_factors
        for attr_name in ["state", "_state", "unwrapped"]:
            if hasattr(env_to_check, attr_name):
                attr = getattr(env_to_check, attr_name)
                if hasattr(attr, "rma_env_factors"):
                    e_t = attr.rma_env_factors
                    if isinstance(e_t, torch.Tensor) and e_t.shape[0] > 0:
                        return e_t.clone().to(self.device)
        
        return None
    
    def train_step(self, e_t: torch.Tensor):
        """Perform one encoder/decoder training step.
        Args:
            e_t: Environment factors from environment (num_envs, 14)
        """
        if e_t is None or e_t.numel() == 0:
            return
        
        # Normalize environment factors to [0, 1]
        e_t_normalized = self.normalizer.normalize(e_t)
        
        # Forward pass: e_t → encoder → z_t → decoder → e_t_recon
        z_t = self.encoder(e_t_normalized)
        e_t_recon = self.decoder(z_t, apply_scaling=False)
        
        # Decoder loss (reconstruction in normalized space)
        decoder_loss = torch.nn.functional.mse_loss(e_t_recon, e_t_normalized)
        
        # Encoder auxiliary loss (optional: encourage informative latent)
        encoder_loss = torch.nn.functional.mse_loss(z_t, torch.zeros_like(z_t)) * 0.001
        
        # Total auxiliary loss
        total_aux_loss = (
            self.encoder_weight * encoder_loss + 
            self.decoder_weight * decoder_loss
        )
        
        # Backprop encoder/decoder
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        total_aux_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        
        # Track losses
        self.encoder_losses.append(encoder_loss.item())
        self.decoder_losses.append(decoder_loss.item())
    
    def save_checkpoint(self, iteration: int):
        """Save encoder and decoder checkpoints.
        
        Args:
            iteration: Current policy iteration
        """
        encoder_path = Path(self.log_dir) / "checkpoints" / "encoder" / f"encoder_iter_{iteration}.pt"
        decoder_path = Path(self.log_dir) / "checkpoints" / "decoder" / f"decoder_iter_{iteration}.pt"
        
        torch.save(
            {"model_state_dict": self.encoder.state_dict(), "iteration": iteration},
            encoder_path
        )
        torch.save(
            {"model_state_dict": self.decoder.state_dict(), "iteration": iteration},
            decoder_path
        )
        
        # Also save as _latest for easy access
        torch.save(
            {"model_state_dict": self.encoder.state_dict(), "iteration": iteration},
            Path(self.log_dir) / "checkpoints" / "encoder" / "encoder_latest.pt"
        )
        torch.save(
            {"model_state_dict": self.decoder.state_dict(), "iteration": iteration},
            Path(self.log_dir) / "checkpoints" / "decoder" / "decoder_latest.pt"
        )
        
        print(f"[INFO] Saved encoder/decoder checkpoints at policy iteration {iteration}")
    
    def log_losses(self, policy_iter: int):
        """Print average losses.
        
        Args:
            policy_iter: Current policy iteration
        """
        if len(self.encoder_losses) == 0:
            return
        
        avg_encoder_loss = sum(self.encoder_losses[-100:]) / min(100, len(self.encoder_losses))
        avg_decoder_loss = sum(self.decoder_losses[-100:]) / min(100, len(self.decoder_losses))
        
        print(
            f"[Policy Iter {policy_iter:5d}] "
            f"Encoder Loss: {avg_encoder_loss:.6f} | "
            f"Decoder Loss: {avg_decoder_loss:.6f}"
        )


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlOnPolicyRunnerCfg,
):
    """Train with RSL-RL agent + encoder/decoder."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # update frequency in case it is overridden
    if any("physics_freq" in arg or "controller_freq" in arg for arg in hydra_args):
        env_cfg.decimation = int(env_cfg.physics_freq / env_cfg.controller_freq)
        env_cfg.sim.dt = 1.0 / env_cfg.physics_freq
        env_cfg.sim.render_interval = env_cfg.decimation
        for attr_name in dir(env_cfg.scene):
            attr = getattr(env_cfg.scene, attr_name)
            if hasattr(attr, "update_period"):
                prev_update_period = attr.update_period
                if hasattr(attr, "force_threshold"):
                    attr.update_period = 1.0 / env_cfg.physics_freq
                else:
                    attr.update_period = 1.0 / env_cfg.controller_freq
                print(f"{attr_name} update period changed from {prev_update_period} to {attr.update_period}")

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    
    # specify directory for logging runs
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # # wrap for video recording
    # if args_cli.video:
    #     if args_cli.video_interval_steps and args_cli.video_interval_steps > 0:
    #         video_interval_steps = args_cli.video_interval_steps
    #     else:
    #         video_interval_steps = args_cli.video_interval_iter * agent_cfg.num_steps_per_env
    #     video_kwargs = {
    #         "video_folder": os.path.join(log_dir, "videos", "train"),
    #         "step_trigger": lambda step: step % video_interval_steps == 0,
    #         "video_length": args_cli.video_length,
    #         "disable_logger": True,
    #     }
    #     print("[INFO] Recording videos during training.")
    #     print_dict(video_kwargs, nesting=4)
    #     env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=log_dir,
        device=agent_cfg.device,
    )
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    
    # load the checkpoint
    if agent_cfg.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path, load_optimizer=agent_cfg.load_optimizer)

    # Create encoder and decoder
    encoder_cfg = EnvFactorEncoderCfg(in_dim=14, latent_dim=8, hidden_dims=(256, 128))
    encoder = EnvFactorEncoder(cfg=encoder_cfg)

    decoder_cfg = EnvFactorDecoderCfg()
    decoder = EnvFactorDecoder(cfg=decoder_cfg)

    # Create joint trainer
    trainer = RMATrainerJoint(
        runner=runner,
        encoder=encoder,
        decoder=decoder,
        log_dir=log_dir,
        device=agent_cfg.device,
        encoder_weight=args_cli.encoder_weight,
        decoder_weight=args_cli.decoder_weight,
        checkpoint_interval=args_cli.checkpoint_interval,
    )

    print("\n[INFO] Starting joint policy + encoder/decoder training")
    print(f"[INFO] Policy iterations: {agent_cfg.max_iterations}")
    print(f"[INFO] Encoder/decoder trained alongside policy each iteration\n")

    # Joint training loop
    for policy_iter in range(agent_cfg.max_iterations):
        # Policy learning step
        runner.learn(num_learning_iterations=1)
        
        # Extract environment factors
        e_t = trainer.extract_env_factors()
        
        # Train encoder/decoder on current environment factors
        if e_t is not None:
            trainer.train_step(e_t)
        
        # Log losses
        if (policy_iter + 1) % 10 == 0:
            trainer.log_losses(policy_iter + 1)
        
        # Save checkpoints
        if (policy_iter + 1) % args_cli.checkpoint_interval == 0:
            trainer.save_checkpoint(policy_iter + 1)
    
    # Final checkpoint
    trainer.save_checkpoint(agent_cfg.max_iterations)
    
    print("\n[INFO] Training completed!")

    # close the simulator
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
