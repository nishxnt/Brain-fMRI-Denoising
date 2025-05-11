#!/usr/bin/env python
"""
Baseline 3D-UNet denoiser training with YAML config

Usage:
  python -m train.train_baseline --config path/to/config.yaml
"""

import argparse, time, pathlib, yaml
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional import (
    peak_signal_noise_ratio as psnr,
    structural_similarity_index_measure as ssim,
)
from kim_dataset.sampler import TimePatchSampler
from models.unet3d import UNet3D

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config", type=pathlib.Path,
        required=True,
        help="Path to YAML experiment config"
    )
    return p.parse_args()

def flatten_time(x: torch.Tensor) -> torch.Tensor:
    b,c,t,d,h,w = x.shape
    return x.permute(0,2,1,3,4,5).reshape(b*t, c, d, h, w)

def run_epoch(model, loader, device, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    losses, pss, sss = [], [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            noisy = batch["noisy"].to(device)
            clean = batch["clean"].to(device).permute(0,2,1,3,4,5)
            pred  = model(noisy).permute(0,2,1,3,4,5)
            loss  = nn.functional.mse_loss(pred, clean)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
            pss.append(psnr(flatten_time(pred), flatten_time(clean)).item())
            sss.append(ssim(flatten_time(pred), flatten_time(clean)).item())
    return sum(losses)/len(losses), sum(pss)/len(pss), sum(sss)/len(sss)

def main():
    args = get_args()
    cfg = yaml.safe_load(args.config.read_text())

    # --- Datasets & loaders from YAML ---
    ds_cfg = cfg["dataset"]
    train_ds = TimePatchSampler(
        ds_cfg["manifest_csv"],
        ds_cfg["zarr_root"],
        tuple(ds_cfg["patch_size"]),
        ds_cfg["window"],
        ds_cfg["patches_per_run"]
    )
    val_ds   = TimePatchSampler(
        ds_cfg["manifest_csv"],
        ds_cfg["zarr_root"],
        tuple(ds_cfg["patch_size"]),
        ds_cfg["window"],
        ds_cfg["patches_per_run_val"]
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["train"]["batch_size"], num_workers=2)

    # --- Model, optimizer, loss ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = cfg["model"]
    model = UNet3D(
        in_ch   = m["in_ch"],
        out_ch  = m["out_ch"],
        features= m["features"]
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    patience  = cfg["train"]["patience"]
    epochs    = cfg["train"]["epochs"]

    # --- TensorBoard / checkpoint dirs ---
    run_cfg = cfg["logging"]
    tb = SummaryWriter(pathlib.Path(run_cfg["logdir"]) / run_cfg["run_name"])
    ckpt_path = pathlib.Path(run_cfg["logdir"]) / "checkpoint.pt"

    best_val, stale = float("inf"), 0
    for epoch in range(epochs):
        tr_loss, tr_psnr, tr_ssim = run_epoch(model, train_loader, device, optimizer)
        vl_loss, vl_psnr, vl_ssim = run_epoch(model, val_loader,   device, None)

        # log
        tb.add_scalar("loss/train", tr_loss, epoch)
        tb.add_scalar("loss/val",   vl_loss, epoch)
        tb.add_scalar("psnr/train", tr_psnr, epoch)
        tb.add_scalar("psnr/val",   vl_psnr, epoch)
        tb.add_scalar("ssim/train", tr_ssim, epoch)
        tb.add_scalar("ssim/val",   vl_ssim, epoch)

        print(f"Epoch {epoch:02d} | "
              f"train {tr_loss:.4f}/{tr_psnr:.2f}/{tr_ssim:.3f}  ||  "
              f"val   {vl_loss:.4f}/{vl_psnr:.2f}/{vl_ssim:.3f}")

        # early‐stop + checkpoint
        if vl_loss < best_val:
            best_val, stale = vl_loss, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping@{epoch}, no val-improve ≥ {patience}")
                break

    tb.close()
    print("Done. Best val loss:", best_val)

if __name__ == "__main__":
    main()
