# fine_tuning/data_loader/pressure_utils.py

import numpy as np

def process_pressure_levels(raw_levels, target=50):
    """
    Collapse a 3D raw_levels array (levels, H, W) into a 1D tuple of `target` integers 
    evenly spaced between min and max.
    """
    arr = np.asarray(raw_levels, dtype=np.float32)
    p_min, p_max = float(arr.min()), float(arr.max())
    spaced = np.linspace(p_min, p_max, num=target)
    ints   = np.round(spaced).astype(int)
    return tuple(ints)
