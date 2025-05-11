"""
Visualise UNet-3D denoising on a single fMRI volume.
Produces a 3-panel PNG in reports/figures/.
"""
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt

from models.unet3d import UNet3D           # PYTHONPATH=. when launched with  -m
from kim_dataset.mask      import compute_mask
from kim_dataset.normalize import zscore

# -----------------------------------------------------------------------------#
# 1) load NIfTI (change the filename if you use a different run)
nii_path = Path("data/raw/sub-01_ses-1_task-motor_run-1_bold.nii.gz")
img      = nib.load(nii_path)                   # keep nibabel image
vol_np   = img.get_fdata(dtype="float32")       # raw data

# 2) make sure volume axes are (X,Y,Z,T)      ───────────────────────────────────
mask_np  = compute_mask(img)                    # (X,Y,Z)
if vol_np.shape[0] == mask_np.shape[0] and vol_np.ndim == 4:
    # volume is (T,X,Y,Z)  →  move T-axis to the end → (X,Y,Z,T)
    vol_np = np.moveaxis(vol_np, 0, -1)

vol_np   = zscore(vol_np, mask_np)              # normalise

# 3) tiny centre crop (32³) just for the demo
xs = ys = zs = slice(40, 72)                    # adjust if needed
crop_np   = vol_np[xs, ys, zs, :]               # (32,32,32,T)

# build tensor  (N,C,D,H,W) = (1,16,32,32,32)
patch = ( torch.from_numpy(crop_np)
            .permute(3,0,1,2)                 # → (T,32,32,32)
            .unsqueeze(0)                     # → (1,T,32,32,32)
            .float() )

# 4) run the network  ───────────────────────────────────────────────────────────
ckpt = Path("content/unet_baseline_best.pt")   # already downloaded earlier
net  = UNet3D(in_ch=16, out_ch=16, features=16).cpu()
net.load_state_dict(torch.load(ckpt, map_location="cpu"))
net.eval()

with torch.no_grad():
    den = net(patch).cpu().squeeze()           # (16,T,32,32,1) → (16,T,32,32)

# 5) pick one slice (t=8, z=16) & plot  ─────────────────────────────────────────
t , z = 8 , 16
noisy = crop_np[:, :, :, t][ :, :, z ]         # (32,32)
den2d = den[ 0, t, :, :, z ].numpy()           # channel-0 output
clean = vol_np[xs, ys, zs, t][ :, :, z ]       # ground truth (same as noisy)

fig, axs = plt.subplots(1, 3, figsize=(9,3))
for ax, img, ttl in zip(axs, [noisy, den2d, clean],
                        ["Noisy", "Denoised", "Clean"]):
    ax.imshow(img, cmap="gray", vmin=-2, vmax= 2)
    ax.set_title(ttl); ax.axis("off")
plt.tight_layout()

out_png = Path("reports/figures/baseline_full_demo.png")
out_png.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_png, dpi=150)
print(f"\n✓  saved → {out_png.resolve()}\n")
