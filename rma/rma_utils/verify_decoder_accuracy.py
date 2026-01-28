#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0

"""Script to verify decoder reconstruction accuracy on trained models."""

import argparse
import os
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

from unitree_h12_sim2sim.rma_modules import (
    EnvFactorEncoder,
    EnvFactorEncoderCfg,
    EnvFactorDecoder,
    EnvFactorDecoderCfg,
)


def find_latest_checkpoint(log_dir: str, model_type: str = "encoder") -> str:
    """Find the latest checkpoint of a specific type.
    
    Args:
        log_dir: Path to log directory (training run directory)
        model_type: "encoder" or "decoder"
    
    Returns:
        Path to latest checkpoint
    """
    checkpoint_dir = Path(log_dir) / "checkpoints" / model_type
    
    if not checkpoint_dir.exists():
        return None
    
    # Find latest checkpoint
    checkpoints = list(checkpoint_dir.glob(f"{model_type}_iter_*.pt"))
    if not checkpoints:
        # Try latest
        latest = checkpoint_dir / f"{model_type}_latest.pt"
        if latest.exists():
            return str(latest)
        return None
    
    # Sort by iteration number and return latest
    checkpoints.sort(key=lambda x: int(x.stem.split("_")[-1]))
    return str(checkpoints[-1])


def load_models(
    encoder_path: str,
    decoder_path: str,
    device: str = "cuda"
) -> tuple:
    """Load encoder and decoder models.
    
    Args:
        encoder_path: Path to encoder checkpoint
        decoder_path: Path to decoder checkpoint
        device: Device to load to
    
    Returns:
        Tuple of (encoder, decoder, encoder_cfg, decoder_cfg)
    """
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder checkpoint not found: {encoder_path}")
    if not os.path.exists(decoder_path):
        raise FileNotFoundError(f"Decoder checkpoint not found: {decoder_path}")
    
    # Load encoder
    encoder_ckpt = torch.load(encoder_path, map_location=device)
    encoder_cfg = EnvFactorEncoderCfg(in_dim=17, latent_dim=8, hidden_dims=(256, 128))
    encoder = EnvFactorEncoder(cfg=encoder_cfg).to(device)
    encoder.load_state_dict(encoder_ckpt["model_state_dict"])
    encoder.eval()
    
    # Load decoder
    decoder_ckpt = torch.load(decoder_path, map_location=device)
    decoder_cfg = EnvFactorDecoderCfg()
    decoder = EnvFactorDecoder(cfg=decoder_cfg).to(device)
    decoder.load_state_dict(decoder_ckpt["model_state_dict"])
    decoder.eval()
    
    return encoder, decoder, encoder_cfg, decoder_cfg


def compute_reconstruction_stats(
    encoder: EnvFactorEncoder,
    decoder: EnvFactorDecoder,
    e_t: torch.Tensor,
    device: str = "cuda"
) -> dict:
    """Compute reconstruction statistics for environment factors.
    
    Args:
        encoder: Encoder model
        decoder: Decoder model
        e_t: Ground truth environment factors (N, 17)
        device: Device to compute on
    
    Returns:
        Dictionary with reconstruction statistics
    """
    e_t = e_t.to(device)
    
    with torch.no_grad():
        # Encode
        z_t = encoder(e_t)
        
        # Decode
        e_t_recon = decoder(z_t, apply_scaling=True)
    
    # Compute errors
    mse = torch.nn.functional.mse_loss(e_t, e_t_recon, reduction="none")
    mae = torch.nn.functional.l1_loss(e_t, e_t_recon, reduction="none")
    
    # Per-factor statistics
    factor_names = [
        "Payload Force (N)",
        *[f"Leg Strength {i}" for i in range(12)],
        "Friction",
        "Terrain Amplitude (m)",
        "Terrain Lengthscale (m)",
        "Terrain Noise Step",
    ]
    
    stats = {
        "total_mse": mse.mean().item(),
        "total_mae": mae.mean().item(),
        "total_rmse": mse.mean().sqrt().item(),
        "per_factor": {}
    }
    
    for i, name in enumerate(factor_names):
        stats["per_factor"][name] = {
            "mse": mse[:, i].mean().item(),
            "mae": mae[:, i].mean().item(),
            "rmse": mse[:, i].mean().sqrt().item(),
            "min": e_t[:, i].min().item(),
            "max": e_t[:, i].max().item(),
            "mean": e_t[:, i].mean().item(),
            "std": e_t[:, i].std().item(),
        }
    
    return stats


def main():
    """Main verification function."""
    parser = argparse.ArgumentParser(description="Verify decoder reconstruction accuracy")
    parser.add_argument("--log-dir", type=str, default=None, help="Path to training log directory")
    parser.add_argument("--encoder", type=str, default=None, help="Path to encoder checkpoint")
    parser.add_argument("--decoder", type=str, default=None, help="Path to decoder checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--num-samples", type=int, default=1024, help="Number of test samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Find checkpoints
    encoder_path = args.encoder
    decoder_path = args.decoder
    
    if args.log_dir:
        print(f"[INFO] Using log directory: {args.log_dir}")
        if not encoder_path:
            encoder_path = find_latest_checkpoint(args.log_dir, "encoder")
        if not decoder_path:
            decoder_path = find_latest_checkpoint(args.log_dir, "decoder")
    
    if not encoder_path or not decoder_path:
        # Try to find automatically
        print("[INFO] Searching for latest training run...")
        log_root = Path("/home/niraj/isaac_projects/unitree_h12_rma/wbc_agile_utils/logs/rsl_rl/unitree_h12_walk_rma")
        if log_root.exists():
            runs = sorted(log_root.glob("2026-*"))
            if runs:
                latest_run = runs[-1]
                print(f"[INFO] Found latest run: {latest_run.name}")
                if not encoder_path:
                    encoder_path = find_latest_checkpoint(str(latest_run), "encoder")
                if not decoder_path:
                    decoder_path = find_latest_checkpoint(str(latest_run), "decoder")
    
    if not encoder_path or not decoder_path:
        print("[ERROR] Could not find encoder/decoder checkpoints")
        print("[ERROR] Try specifying --log-dir or --encoder/--decoder paths")
        return
    
    print(f"[INFO] Encoder checkpoint: {encoder_path}")
    print(f"[INFO] Decoder checkpoint: {decoder_path}")
    
    # Load models
    print("\n[INFO] Loading models...")
    try:
        encoder, decoder, encoder_cfg, decoder_cfg = load_models(
            encoder_path, decoder_path, device=args.device
        )
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        return
    
    print(f"[INFO] Encoder: in_dim={encoder_cfg.in_dim}, latent_dim={encoder_cfg.latent_dim}")
    print(f"[INFO] Decoder: output_dim=17")
    
    # Generate test samples
    print(f"\n[INFO] Generating {args.num_samples} test samples...")
    e_t_test = torch.randn(args.num_samples, 17)
    
    # Scale to realistic ranges
    e_t_test[:, 0] = torch.clamp(e_t_test[:, 0] * 12.5 + 25, 0, 50)      # Force: 0-50 N
    e_t_test[:, 1:13] = torch.clamp(e_t_test[:, 1:13] * 0.05 + 1.0, 0.9, 1.1)  # Leg strength: 0.9-1.1
    e_t_test[:, 13] = torch.clamp(e_t_test[:, 13] * 0.3 + 0.5, 0, 1)     # Friction: 0-1
    e_t_test[:, 14] = torch.clamp(e_t_test[:, 14] * 0.5e-3, 0, 2e-3)     # Terrain amplitude
    e_t_test[:, 15] = torch.clamp(e_t_test[:, 15] * 0.05 + 0.1, 0, 0.2)  # Terrain lengthscale
    e_t_test[:, 16] = torch.clamp(e_t_test[:, 16] * 0.025 + 0.05, 0, 0.1)  # Terrain noise step
    
    # Compute reconstruction statistics
    print("\n[INFO] Computing reconstruction statistics...")
    stats = compute_reconstruction_stats(encoder, decoder, e_t_test, device=args.device)
    
    # Print results
    print("\n" + "="*100)
    print("DECODER RECONSTRUCTION ACCURACY REPORT")
    print("="*100)
    print(f"Test samples: {args.num_samples}")
    print(f"Device: {args.device}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("-"*100)
    
    print(f"\nOverall Reconstruction Error:")
    print(f"  Mean Squared Error (MSE):      {stats['total_mse']:.6f}")
    print(f"  Mean Absolute Error (MAE):    {stats['total_mae']:.6f}")
    print(f"  Root Mean Squared Error (RMSE): {stats['total_rmse']:.6f}")
    
    print(f"\nPer-Factor Reconstruction Error:")
    print("-"*100)
    print(f"{'Factor':<30} {'MSE':<12} {'MAE':<12} {'RMSE':<12} {'Range':<20}")
    print("-"*100)
    
    for factor_name, factor_stats in stats["per_factor"].items():
        range_str = f"[{factor_stats['min']:.4f}, {factor_stats['max']:.4f}]"
        print(
            f"{factor_name:<30} "
            f"{factor_stats['mse']:<12.6f} "
            f"{factor_stats['mae']:<12.6f} "
            f"{factor_stats['rmse']:<12.6f} "
            f"{range_str:<20}"
        )
    
    print("\n" + "="*100)
    
    # Interpretation
    print("\nInterpretation Guide:")
    print("  MSE < 0.001: Excellent reconstruction (< 0.1% error)")
    print("  MSE < 0.01:  Good reconstruction (< 1% error)")
    print("  MSE < 0.1:   Acceptable reconstruction (< 10% error)")
    print("  MSE > 0.1:   Poor reconstruction (> 10% error)")
    
    # Quality assessment
    print("\nQuality Assessment:")
    if stats['total_mse'] < 0.001:
        print("  ✅ EXCELLENT: Decoder provides excellent reconstruction")
    elif stats['total_mse'] < 0.01:
        print("  ✅ GOOD: Decoder provides good reconstruction")
    elif stats['total_mse'] < 0.1:
        print("  ⚠️  ACCEPTABLE: Decoder reconstruction is acceptable but could improve")
    else:
        print("  ❌ POOR: Decoder reconstruction needs improvement")
    
    print("\n" + "="*100)


if __name__ == "__main__":
    main()
