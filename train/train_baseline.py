#!/usr/bin/env python
"""
Baseline 3D-UNet denoiser training script
----------------------------------------

    python -m train.train_baseline \
        --epochs 30 --batch 4 --patience 6 --run_name baseline_full
"""

import argparse, time, pathlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional import (
    peak_signal_noise_ratio as psnr,
    structural_similarity_index_measure as ssim,
)

from kim_dataset.sampler import TimePatchSampler
from models.unet3d import UNet3D


# --------------------------------------------------------------------------- #
#  CLI arguments
# --------------------------------------------------------------------------- #
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--patience", type=int, default=4,
                   help="early-stopping patience (epochs)")
    p.add_argument("--run_name", type=str,
                   default=time.strftime("baseline_%Y%m%d_%H%M"))
    return p.parse_args()


# --------------------------------------------------------------------------- #
#  Single epoch helpers
# --------------------------------------------------------------------------- #
def run_epoch(model, loader, device, optim=None):
    """Run **one** epoch. If `optim` is None → evaluation mode."""
    train = optim is not None
    model.train() if train else model.eval()

    losses, pss, sss = [], [], []
    with (torch.enable_grad() if train else torch.no_grad()):
        for batch in loader:
            noisy = batch["noisy"].to(device)
            clean = batch["clean"].to(device).permute(0, 2, 1, 3, 4, 5)  # [B,16,1...]

            pred = model(noisy).permute(0, 2, 1, 3, 4, 5)               # [B,1,16...]

            loss = nn.functional.mse_loss(pred, clean)
            if train:
                optim.zero_grad()
                loss.backward()
                optim.step()

            losses.append(loss.item())
            pss.append(psnr(pred, clean).item())
            sss.append(ssim(pred, clean).item())

    return (
        sum(losses) / len(losses),
        sum(pss) / len(pss),
        sum(sss) / len(sss),
    )


# --------------------------------------------------------------------------- #
#  Main training routine
# --------------------------------------------------------------------------- #
def main():
    args = get_args()

    # ------------------------- datasets & loaders -------------------------- #
    train_ds = TimePatchSampler(
        "data/manifests/all_runs.csv",
        "data/processed/data.zarr",
        patch_size=(32, 32, 32),
        window=16,
        patches_per_run=64,
    )
    val_ds = TimePatchSampler(
        "data/manifests/all_runs.csv",
        "data/processed/data.zarr",
        patch_size=(32, 32, 32),
        window=16,
        patches_per_run=16,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch, num_workers=2)

    # ------------------------- model / optim ------------------------------- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(in_ch=16, out_ch=16, features=16).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # ------------------------- logging ------------------------------------- #
    log_dir = pathlib.Path("runs") / args.run_name
    tb = SummaryWriter(log_dir)

    best_val, stale = float("inf"), 0
    for epoch in range(args.epochs):
        # ---- training ----
        train_loss, train_psnr, train_ssim = run_epoch(
            model, train_loader, device, optimizer
        )

        # ---- validation ----
        val_loss, val_psnr, val_ssim = run_epoch(model, val_loader, device)

        # ---- log & print ----
        tb.add_scalars("loss",   {"train": train_loss, "val": val_loss}, epoch)
        tb.add_scalars("psnr",   {"train": train_psnr, "val": val_psnr}, epoch)
        tb.add_scalars("ssim",   {"train": train_ssim, "val": val_ssim}, epoch)

        print(
            f"Epoch {epoch:02d} | "
            f"train {train_loss:.4f} / {train_psnr:.2f} PSNR / {train_ssim:.3f} SSIM  ||  "
            f"val {val_loss:.4f} / {val_psnr:.2f} PSNR / {val_ssim:.3f} SSIM"
        )

        # ---- early-stopping ----
        if val_loss < best_val:
            best_val, stale = val_loss, 0
            torch.save(model.state_dict(), "models/unet_baseline_best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping (no val-improve ≥ {args.patience})")
                break

    tb.close()
    print("Training done. Best val loss:", best_val)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
