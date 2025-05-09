#!/usr/bin/env python
"""
Baseline 3-D UNet denoiser training
----------------------------------

Example:
    python -m train.train_baseline \
        --epochs 30 --batch 4 --patience 6 --run_name baseline_full
"""

# --------------------------------------------------------------------------- #
#  Imports
# --------------------------------------------------------------------------- #
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
#  Helpers
# --------------------------------------------------------------------------- #
def flatten_time(x: torch.Tensor) -> torch.Tensor:
    """
    Collapse the time dimension (T) so torchmetrics sees
    [B', C, D, H, W] where B' = B*T.
    Input  shape: [B, C, T, D, H, W]
    Output shape: [B*T, C, D, H, W]
    """
    b, c, t, d, h, w = x.shape
    return x.permute(0, 2, 1, 3, 4, 5).reshape(b * t, c, d, h, w)


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",   type=int, default=10)
    p.add_argument("--batch",    type=int, default=2)
    p.add_argument("--patience", type=int, default=4,
                   help="early-stopping patience (epochs)")
    p.add_argument("--run_name", type=str,
                   default=time.strftime("baseline_%Y%m%d_%H%M"))
    return p.parse_args()


def run_epoch(model, loader, device, optimiser=None):
    """Run one epoch; if optimiser is None → eval mode."""
    training = optimiser is not None
    model.train() if training else model.eval()

    losses, pss, sss = [], [], []
    ctxt = torch.enable_grad() if training else torch.no_grad()
    with ctxt:
        for batch in loader:
            noisy  = batch["noisy"].to(device)                               # [B,1,T,D,H,W]
            clean  = batch["clean"].to(device).permute(0, 2, 1, 3, 4, 5)     # [B,16,1,D,H,W]

            pred   = model(noisy).permute(0, 2, 1, 3, 4, 5)                  # [B,16,1→1?,D,H,W]
            loss   = nn.functional.mse_loss(pred, clean)

            if training:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            # metrics (flatten time axis)
            pred_f  = flatten_time(pred)
            clean_f = flatten_time(clean)

            losses.append(loss.item())
            pss.append(psnr(pred_f,  clean_f).item())
            sss.append(ssim(pred_f,  clean_f).item())

    return (
        sum(losses) / len(losses),
        sum(pss)   / len(pss),
        sum(sss)   / len(sss),
    )


# --------------------------------------------------------------------------- #
#  Main training routine
# --------------------------------------------------------------------------- #
def main():
    args = get_args()

    # ------------------------- datasets & loaders ------------------------- #
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
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, num_workers=2)

    # ------------------------- model / optimiser -------------------------- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = UNet3D(in_ch=16, out_ch=16, features=16).to(device)
    opt    = optim.AdamW(model.parameters(), lr=1e-4)

    # ------------------------- logging ------------------------------------ #
    run_dir = pathlib.Path("runs") / args.run_name
    tb      = SummaryWriter(run_dir)

    best_val, stale = float("inf"), 0
    for epoch in range(args.epochs):
        tr_loss, tr_psnr, tr_ssim = run_epoch(model, train_loader, device, opt)
        vl_loss, vl_psnr, vl_ssim = run_epoch(model, val_loader,   device)

        tb.add_scalars("loss", {"train": tr_loss, "val": vl_loss}, epoch)
        tb.add_scalars("psnr", {"train": tr_psnr, "val": vl_psnr}, epoch)
        tb.add_scalars("ssim", {"train": tr_ssim, "val": vl_ssim}, epoch)

        print(
            f"Epoch {epoch:02d} | "
            f"train {tr_loss:.4f}/{tr_psnr:.2f} PSNR/{tr_ssim:.3f} SSIM  ||  "
            f"val {vl_loss:.4f}/{vl_psnr:.2f} PSNR/{vl_ssim:.3f} SSIM"
        )

        # --------------------- early stopping ----------------------------- #
        if vl_loss < best_val:
            best_val, stale = vl_loss, 0
            torch.save(model.state_dict(), "models/unet_baseline_best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stop (no val-improve ≥ {args.patience})")
                break

    tb.close()
    print("Training finished.  Best val loss:", best_val)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
