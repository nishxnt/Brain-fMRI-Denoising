'''
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
'''
#!/usr/bin/env python3
"""
Visualise UNet-3D denoising on a single fMRI volume.
Produces a 3-panel PNG in reports/figures/.
"""
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt

from models.unet3d import UNet3D
from kim_dataset.mask import compute_mask
from kim_dataset.normalize import zscore

def main():
    # ---- 1) Paths ----
    root = Path(__file__).parent.parent
    nii_path = root / "data/raw/sub-01_ses-1_task-motor-run-1_bold.nii.gz"
    ckpt_path = root / "models/unet_baseline_best.pt"
    out_png   = root / "reports/figures/baseline_full_demo.png"

    # ---- 2) Load & preprocess ----
    img = nib.load(str(nii_path))
    vol = img.get_fdata(dtype=np.float32)      # shape: (X,Y,Z) or (X,Y,Z,T)
    if vol.ndim == 3:
        vol = vol[..., None]                   # → (X,Y,Z,1)
    # now vol.shape == (X,Y,Z,T)
    mask = compute_mask(img)                   # (X,Y,Z), uint8
    vol  = zscore(vol, mask)                   # still (X,Y,Z,T)

    # ---- 3) Centre‐crop a 32×32×32 patch ----
    crop = 32
    X,Y,Z,T = vol.shape
    xs = X//2 - crop//2
    ys = Y//2 - crop//2
    zs = Z//2 - crop//2
    patch_np = vol[xs:xs+crop, ys:ys+crop, zs:zs+crop, :]  # (32,32,32,T)

    # ---- 4) Build torch tensor ----
    # → (N=1, C=T, D=32, H=32, W=32)
    patch = (
        torch.from_numpy(patch_np)
             .permute(3,0,1,2)   # (T,32,32,32)
             .unsqueeze(0)       # (1,T,32,32,32)
             .float()
    )

    # ---- 5) Load model & checkpoint ----
    C = patch.shape[1]  # number of channels = time‐frames
    net = UNet3D(in_ch=C, out_ch=C, features=16)
    net.load_state_dict(torch.load(str(ckpt_path), map_location="cpu"))
    net.eval()

    # ---- 6) Inference ----
    with torch.no_grad():
        den = net(patch).squeeze(0).cpu().numpy()  # shape: (C,32,32,32)

    # ---- 7) Pick one slice & time‐point ----
    t0, z0 = C//2, crop//2
    noisy    = patch_np[:, :, z0, t0]   # [32×32]
    denoised = den[t0, :, :, z0]        # [32×32]
    clean    = patch_np[:, :, z0, t0]   # [32×32]  (we only have noisy)

    # ---- 8) Plot & save ----
    fig, axes = plt.subplots(1,3,figsize=(9,3))
    for ax, img_arr, title in zip(axes,
                                  [noisy, denoised, clean],
                                  ["Noisy","Denoised","Clean"]):
        ax.imshow(img_arr.T, cmap="gray", vmin=-2, vmax=2, origin="lower")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_png), dpi=150)
    print(f"✓ saved → {out_png}")

if __name__ == "__main__":
    main()
