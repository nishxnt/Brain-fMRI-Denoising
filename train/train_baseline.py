#!/usr/bin/env python
import argparse, time, pathlib
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim

from kim_dataset.sampler import TimePatchSampler
from models.unet3d import UNet3D


# ------------------------------------------------------------------
#  Arg-parse
# ------------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument("--epochs",   type=int, default=10)
p.add_argument("--batch",    type=int, default=2)
p.add_argument("--patience", type=int, default=4, help="early-stopping patience (epochs)")
p.add_argument("--run_name", type=str, default=time.strftime("baseline_%Y%m%d_%H%M"))
args = p.parse_args()

# ------------------------------------------------------------------
#  Datasets & loaders
# ------------------------------------------------------------------
train_ds = TimePatchSampler(
    "data/manifests/all_runs.csv",
    "data/processed/data.zarr",
    patch_size=(32, 32, 32), window=16, patches_per_run=64
)
val_ds = TimePatchSampler(
    "data/manifests/all_runs.csv",
    "data/processed/data.zarr",
    patch_size=(32, 32, 32), window=16, patches_per_run=16
)

train_loader = DataLoader(train_ds, batch_size=args.batch, num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=args.batch, num_workers=2)

# ------------------------------------------------------------------
#  Model / opt / loss
# ------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = UNet3D(in_ch=16, out_ch=16, features=16).to(device)
loss_fn   = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# ------------------------------------------------------------------
#  TensorBoard
# ------------------------------------------------------------------
log_dir = pathlib.Path("runs") / args.run_name
tb = SummaryWriter(log_dir)

# ------------------------------------------------------------------
#  Training loop with early-stopping
# ------------------------------------------------------------------
best_val = float("inf")
stale_epochs = 0

for epoch in range(args.epochs):
    # ---- train ----------------------------------------------------
    model.train(); train_losses = []
    for batch in train_loader:
        noisy  = batch["noisy"].to(device)
        clean  = batch["clean"].to(device).permute(0, 2, 1, 3, 4, 5)   # [B,16,1,...]
        pred   = model(noisy).permute(0, 2, 1, 3, 4, 5)                # [B,1,16,...]
        loss   = loss_fn(pred, clean)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        train_losses.append(loss.item())
    avg_train = sum(train_losses) / len(train_losses)

    # ---- validation ----------------------------------------------
    model.eval(); val_losses = []; psnr_vals = []; ssim_vals = []
    with torch.no_grad():
        for batch in val_loader:
            noisy  = batch["noisy"].to(device)
            clean  = batch["clean"].to(device).permute(0, 2, 1, 3, 4, 5)
            pred   = model(noisy).permute(0, 2, 1, 3, 4, 5)
            val_losses.append(loss_fn(pred, clean).item())
            psnr_vals.append(psnr(pred, clean).item())
            ssim_vals.append(ssim(pred, clean).item())
    avg_val  = sum(val_losses) / len(val_losses)
    avg_psnr = sum(psnr_vals) / len(psnr_vals)
    avg_ssim = sum(ssim_vals) / len(ssim_vals)

    # ---- logging --------------------------------------------------
    tb.add_scalars("loss", {"train": avg_train, "val": avg_val}, epoch)
    tb.add_scalars("metrics", {"psnr": avg_psnr, "ssim": avg_ssim}, epoch)
    print(f"Epoch {epoch:02d} | train {avg_train:.4f} | "
          f"val {avg_val:.4f} | PSNR {avg_psnr:.2f} | SSIM {avg_ssim:.3f}")

    # ---- early-stopping ------------------------------------------
    if avg_val < best_val:
        best_val = avg_val
        stale_epochs = 0
        torch.save(model.state_dict(), f"models/unet_baseline_best.pt")
    else:
        stale_epochs += 1
        if stale_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

tb.close()

if __name__=="__main__":
    main()
