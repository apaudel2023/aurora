# fine_tuning/data_loader/atmos_utils.py

import numpy as np

def standardize_atmospheric_vars(atom_vars, target_levels=50, target_y=None, target_x=None):
    """
    Crop each (2, L, H, W) array in atom_vars to (2, target_levels, target_y, target_x).
    """
    out = {}
    # infer grid dims from surf if not provided
    if target_y is None or target_x is None:
        # pick first var
        sample = next(iter(atom_vars.values()))
        _, _, y0, x0 = sample.shape
        target_y = target_y or y0
        target_x = target_x or x0

    for key, arr in atom_vars.items():
        a = np.asarray(arr, dtype=np.float32)
        # crop levels
        if a.shape[1] != target_levels:
            a = a[:, :target_levels, :, :]
        # crop spatial dims
        if a.shape[2] != target_y or a.shape[3] != target_x:
            a = a[:, :, :target_y, :target_x]
        out[key] = a
    return out
