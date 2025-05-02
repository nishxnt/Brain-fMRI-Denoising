import nibabel as nib
import numpy as np

def compute_mask(img: nib.Nifti1Image, thr: float = 0.2) -> np.ndarray:
    """
    Compute a brain mask by thresholding the 95th percentile over time.
    Keeps voxels whose 95th-percentile intensity > thr * global 95th percentile.
    """
    data       = img.get_fdata(dtype=np.float32)       # shape (X,Y,Z,T)
    p95_global = np.percentile(data, 95)
    # percentiles per-voxel over the time axis
    p95_voxels = np.percentile(data, 95, axis=3)       # shape (X,Y,Z)
    mask       = (p95_voxels > thr * p95_global)
    return mask.astype(np.uint8)
