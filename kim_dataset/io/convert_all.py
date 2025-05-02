#!/usr/bin/env python
import argparse, os, json, zarr, nibabel as nib, numpy as np, tqdm
from ..mask import compute_mask
from ..normalize import zscore

def convert_run(nifti_path, out_root):
    img  = nib.load(nifti_path)
    data = img.get_fdata(dtype=np.float32)        # (X,Y,Z,T)
    mask = compute_mask(img)                      # (X,Y,Z)
    data = zscore(data, mask)                     # z-scored & zero background

    # Prepare Zarr store & dataset
    key   = os.path.splitext(os.path.basename(nifti_path))[0]
    store = zarr.open(os.path.join(out_root, "data.zarr"), mode="a")
    ds    = store.require_dataset(
                key, shape=data.shape,
                chunks=(32,32,16,16),
                dtype="float32", overwrite=True
            )
    ds[:] = data

    # Save metadata beside it
    meta = {"key": key, "shape": data.shape, "affine": img.affine.tolist()}
    with open(os.path.join(out_root, f"{key}.json"), "w") as f:
        json.dump(meta, f, indent=2)

def main(args):
    os.makedirs(args.out, exist_ok=True)
    runs = sorted(
        os.path.join(dp, f)
        for dp,_,fs in os.walk(args.raw)
        for f in fs if f.endswith(".nii.gz")
    )
    for fn in tqdm.tqdm(runs, desc="Converting runs"):
        convert_run(fn, args.out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-r", "--raw", required=True,
                   help="Path to data/raw with .nii.gz files")
    p.add_argument("-o", "--out", required=True,
                   help="Output folder under data/processed")
    args = p.parse_args()
    main(args)
