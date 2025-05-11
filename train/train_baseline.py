#!/usr/bin/env python
"""
Baseline 3-D UNet denoiser training driven from a YAML config.

Usage:
    python -m train.train_baseline \
        --config experiments/baseline_full/config.yaml
"""
import argparse
import yaml
import time
import pathlib

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional import (
    peak_signal_noise_ratio as psnr,
    structural_similarity_index_measure as ssim,
)

from kim_dataset.sampler import TimePatchSampler
from models.unet3d import UNet3D

def flatten_time(x: torch.Tensor) -> torch.Tensor:
    b, c, t, d, h, w = x.shape
    return x.permute(0, 2, 1, 3, 4, 5).reshape(b * t, c, d, h, w)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config", type=pathlib.Path,
        default=pathlib.Path("experiments/baseline_full/config.yaml"),
        help="Path to YAML experiment config"
    )
    return p.parse_args()

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    losses, psns, ssms = [], [], []
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            noisy = batch["noisy"].to(device)                       # [B,1,T,D,H,W]
            clean = batch["clean"].to(device).permute(0,2,1,3,4,5)   # [B,T,1,D,H,W]→[B,1,T,D,H,W] then permute

            pred  = model(noisy)                                     # [B,16,32,32,32] etc
            # ensure pred has same layout as clean:
            pred  = pred.permute(0,2,1,3,4,5)                         # if needed

            loss  = F.mse_loss(pred, clean)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # flatten time for metrics
            pf = flatten_time(pred)
            cf = flatten_time(clean)

            losses.append(loss.item())
            psns .append(psnr (pf, cf).item())
            ssms .append(ssim (pf, cf).item())

    return (
        sum(losses) / len(losses),
        sum(psns ) / len(psns ),
        sum(ssms ) / len(ssms ),
    )

def main():
    args   = parse_args()
    cfg    = load_config(args.config)

    # ———— Build datasets & loaders ————————————— #
    train_ds = TimePatchSampler(
        cfg["dataset"]["manifest_csv"],
        cfg["dataset"]["zarr_root"],
        patch_size=tuple(cfg["dataset"]["patch_size"]),
        window=cfg["dataset"]["window"],
        patches_per_run=cfg["dataset"]["patches_per_run_train"],
    )
    val_ds = TimePatchSampler(
        cfg["dataset"]["manifest_csv"],
        cfg["dataset"]["zarr_root"],
        patch_size=tuple(cfg["dataset"]["patch_size"]),
        window=cfg["dataset"]["window"],
        patches_per_run=cfg["dataset"]["patches_per_run_val"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"].get("num_workers", 2),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"].get("num_workers", 2),
    )

    # ———— Model, optimizer, device ——————————————— #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mcfg   = cfg["model"]
    model  = UNet3D(
        in_ch   = mcfg["in_ch"],
        out_ch  = mcfg["out_ch"],
        features= mcfg["features"],
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])

    # ———— Logging & checkpointing ———————————————— #
    run_dir = pathlib.Path(cfg["logging"]["logdir"]) / cfg["logging"]["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    tb = SummaryWriter(run_dir)

    best_val = float("inf")
    stale    = 0

    E = cfg["train"]["epochs"]
    P = cfg["train"]["patience"]

    for epoch in range(E):
        tr_loss, tr_psnr, tr_ssim = run_epoch(model, train_loader, device, optimizer)
        vl_loss, vl_psnr, vl_ssim = run_epoch(model, val_loader,   device, None)

        tb.add_scalars("loss", {"train": tr_loss, "val": vl_loss}, epoch)
        tb.add_scalars("psnr", {"train": tr_psnr, "val": vl_psnr}, epoch)
        tb.add_scalars("ssim", {"train": tr_ssim, "val": vl_ssim}, epoch)

        print(
            f"Epoch {epoch:02d} | "
            f"train {tr_loss:.4f}/{tr_psnr:.2f} PSNR/{tr_ssim:.3f} SSIM  ||  "
            f"val   {vl_loss:.4f}/{vl_psnr:.2f} PSNR/{vl_ssim:.3f} SSIM"
        )

        # early‐stopping + best model save
        if vl_loss < best_val:
            best_val = vl_loss
            stale    = 0
            torch.save(
                model.state_dict(),
                cfg["logging"]["checkpoint_path"],
            )
        else:
            stale += 1
            if stale >= P:
                print(f"Early stopping after {epoch:02d} (no val‐improve ≥ {P})")
                break

    tb.close()
    print(f"Training finished.  Best validation loss: {best_val:.6f}")

if __name__ == "__main__":
    main()
