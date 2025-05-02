#!/usr/bin/env python
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from kim_dataset.sampler import TimePatchSampler
from models.unet3d import UNet3D
from torch.utils.tensorboard import SummaryWriter

def main():
    # Datasets & DataLoaders
    train_ds = TimePatchSampler(
        "data/manifests/all_runs.csv",
        "data/processed/data.zarr",
        patch_size=(32,32,32), window=16, patches_per_run=64
    )
    train_loader = DataLoader(train_ds, batch_size=2, num_workers=2)

    val_ds = TimePatchSampler(
        "data/manifests/all_runs.csv",
        "data/processed/data.zarr",
        patch_size=(32,32,32), window=16, patches_per_run=16
    )
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=2)

    # Model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D().to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # TensorBoard logger
    tb = SummaryWriter("runs/baseline")

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            noisy = batch["noisy"].to(device)
            clean = batch["clean"].to(device)
            pred  = model(noisy)
            loss  = loss_fn(pred, clean)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_losses.append(loss.item())
        avg_train = sum(train_losses)/len(train_losses)

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                noisy = batch["noisy"].to(device)
                clean = batch["clean"].to(device)
                pred  = model(noisy)
                val_losses.append(loss_fn(pred, clean).item())
        avg_val = sum(val_losses)/len(val_losses)

        # Logging
        tb.add_scalars("Loss", {"train": avg_train, "val": avg_val}, epoch)
        print(f"Epoch {epoch:02d} — train: {avg_train:.4f}, val: {avg_val:.4f}")

    tb.close()

if __name__=="__main__":
    main()
