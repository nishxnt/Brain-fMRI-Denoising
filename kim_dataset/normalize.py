import numpy as np

def zscore(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Z-score each voxel’s time series within the brain mask.
    Background (mask==0) is left at zero.
    """
    # data: (X,Y,Z,T), mask: (X,Y,Z)
    brain_ts = data[mask > 0]                # shape (N_brain_voxels, T)
    mean_ts  = brain_ts.mean(axis=-1, keepdims=True)
    std_ts   = brain_ts.std(axis=-1, keepdims=True) + 1e-6
    # broadcast back to full volume
    data[mask > 0] = (brain_ts - mean_ts) / std_ts
    data[mask == 0] = 0
    return data.astype(np.float32)
