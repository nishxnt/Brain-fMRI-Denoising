#!/usr/bin/env python3
"""
Visualise UNet-3D denoising on a single fMRI volume.

* Loads one run (NIfTI) from data/raw/
* Runs the baseline UNet-3D checkpoint
* Saves a 3-panel PNG (noisy / denoised / “clean”*) in reports/figures/

*the “clean” panel is just the input again, because we do not have ground truth
"""

from pathlib import Path
import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt

from models.unet3d import UNet3D          # PYTHONPATH=.  when launched with -m
from kim_dataset.mask import compute_mask
from kim_dataset.normalize import zscore

# -----------------------------------------------------------------------------
# 1) load NIfTI  --------------------------------------------------------------
# -----------------------------------------------------------------------------
nii_path = Path("data/raw/sub-01_ses-1_task-motor_run-1_bold.nii.gz")
img      = nib.load(nii_path)                   # keep nibabel image (affine, hdr)
vol_np   = img.get_fdata(dtype="float32")       # raw data  (T,X,Y,Z) or (X,Y,Z)

# if the file you downloaded has a different name, change `nii_path` above
# -----------------------------------------------------------------------------
# 2) make sure volume axes are (X,Y,Z,T)  -------------------------------------
# -----------------------------------------------------------------------------
mask_np = compute_mask(img)                     # (X,Y,Z)

# ----- if T is first, move it to the end -------------------------------------
if vol_np.ndim == 4 and vol_np.shape[0] < 10:   # T-axis is first  -> move it
    vol_np = np.moveaxis(vol_np, 0, -1)         # (T,X,Y,Z) -> (X,Y,Z,T)

# ----- recompute mask after possible axis move (keeps affine) ----------------
mask_np = compute_mask(nib.Nifti1Image(vol_np.mean(-1), img.affine))   # (X,Y,Z)

# make mask 4-D so it matches vol_np (X,Y,Z,T)
mask_np = np.broadcast_to(mask_np[..., None], vol_np.shape)           # (X,Y,Z,T)

# ----- z-score normalisation  -------------------------------------------------
vol_np = zscore(vol_np, mask_np)                # (X,Y,Z,T)

# -----------------------------------------------------------------------------
# 3) tiny centre crop (32³) just for the demo  --------------------------------
# -----------------------------------------------------------------------------
xs = ys = zs = slice(40, 72)                    # adjust if needed (>=32³)
crop_np = vol_np[xs, ys, zs, :]                 # (32,32,32,T)

# build tensor  (N,C,D,H,W) = (1,16,32,32,32)  -------------------------------
patch = ( torch.from_numpy(crop_np)             # (32,32,32,16)  X,Y,Z,T
            .permute(3,0,1,2)                   # -> (T,32,32,32)
            .unsqueeze(0).unsqueeze(1)          # -> (1,1,T,32,32,32)
            .float() )

# -----------------------------------------------------------------------------
# 4) run the network  ----------------------------------------------------------
# -----------------------------------------------------------------------------
ckpt = Path("models/unet_baseline_best.pt")
net   = UNet3D(in_ch=16, out_ch=16, features=16).cpu()
net.load_state_dict(torch.load(ckpt, map_location="cpu"))
net.eval()

with torch.no_grad():
    den = net(patch).cpu().squeeze()            # (16,T,32,32) -> (16,32,32)

# -----------------------------------------------------------------------------
# 5) pick one slice (t=8, z=16) & plot  ----------------------------------------
# -----------------------------------------------------------------------------
t, z = 8, 16
noisy  = patch_np[t, :, :, z] .numpy()          # (32,32)
den2d  = den[ 0, t, :, :, z].numpy()            # channel-0 output
clean  = vol_np[xs, ys, zs, t][ :, :, z]        # same as noisy

fig, axs = plt.subplots(1, 3, figsize=(9,3))
for ax, img, title in zip(
        axs, [noisy, den2d, clean], ["Noisy", "Denoised", "Clean"]):
    ax.imshow(img, cmap="gray", vmin=-2, vmax= 2)
    ax.set_title(title); ax.axis("off")
plt.tight_layout()

# -----------------------------------------------------------------------------
# 6) save figure  --------------------------------------------------------------
# -----------------------------------------------------------------------------
out_png = Path("reports/figures/baseline_full_demo.png")
out_png.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_png, dpi=150)
print("✓  saved →", out_png.resolve())

