# fine_tuning/data_loader/coords_utils.py

import numpy as np

def process_lat_lon(lat, lon):
    """
    From collapsed 2D→1D or raw 1D, produce:
      lat1d: (H,) strictly decreasing in [-90,90]
      lon1d: (W,) strictly increasing in [0,360)

    Args:
      lat: 1D or 2D. If 2D, shape (H,W), will take lat[:,0]
      lon: 1D or 2D. If 2D, shape (H,W), will take lon[0,:]

    Returns:
      lat1d, lon1d
    """
    lat = np.asarray(lat, dtype=np.float32)
    lon = np.asarray(lon, dtype=np.float32)

    # collapse 2D→1D if needed
    if lat.ndim == 2:
        lat = lat[:,0]
    elif lat.ndim != 1:
        raise ValueError("lat must be 1D or 2D")
    if lon.ndim == 2:
        lon = lon[0,:]
    elif lon.ndim != 1:
        raise ValueError("lon must be 1D or 2D")

    # clip and wrap
    lat = np.clip(lat, -90.0, 90.0)
    lon = (lon + 360.0) % 360.0

    # enforce strict monotonicity
    if not np.all(lat[1:] < lat[:-1]):
        lat = np.sort(lat)[::-1]
    if not np.all(lon[1:] > lon[:-1]):
        lon = np.sort(lon)

    return lat, lon
