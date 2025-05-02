import torch
from torch.utils.data import IterableDataset
import zarr, pandas as pd
import numpy as np

class TimePatchSampler(IterableDataset):
    def __init__(self,
                 manifest_csv: str,
                 zarr_root:     str,
                 patch_size:    tuple=(32,32,32),
                 window:        int=16,
                 patches_per_run:int=64):
        """
        manifest_csv: path to your all_runs.csv
        zarr_root:     path to data/processed/data.zarr
        patch_size:    (dx,dy,dz) spatial patch size
        window:        number of time frames per sample
        patches_per_run: how many patches to draw from each run
        """
        self.df    = pd.read_csv(manifest_csv)
        self.store = zarr.open(zarr_root, mode="r")
        self.patch_size      = patch_size
        self.window          = window
        self.patches_per_run = patches_per_run

    def __iter__(self):
        # Shuffle runs each epoch
        for _, row in self.df.sample(frac=1).iterrows():
            key = row["key"]
            vol = self.store[key]            # shape (X,Y,Z,T)
            X,Y,Z,T = vol.shape
            for _ in range(self.patches_per_run):
                # random spatial start
                x0 = np.random.randint(0, X - self.patch_size[0] + 1)
                y0 = np.random.randint(0, Y - self.patch_size[1] + 1)
                z0 = np.random.randint(0, Z - self.patch_size[2] + 1)
                # random time window
                t0 = np.random.randint(0, T - self.window + 1)
                patch = vol[
                    x0:x0+self.patch_size[0],
                    y0:y0+self.patch_size[1],
                    z0:z0+self.patch_size[2],
                    t0:t0+self.window
                ]  # shape (dx,dy,dz,window)
                # convert to torch: [C=1, T, Z, Y, X]
                arr = torch.from_numpy(patch).float()
                arr = arr.permute(3,2,1,0).unsqueeze(1)
                # For now use same for noisy and clean;
                # adjust if you have two Zarr groups
                yield {"noisy": arr, "clean": arr}

