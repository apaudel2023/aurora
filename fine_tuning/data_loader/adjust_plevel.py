from pathlib import Path
from typing import List, Dict
import numpy as np
import xarray as xr
from wrf import interplevel, destagger


def interpolate_wrf3d_to_levels(
    file_paths: List[str],
    target_levels: List[float],
) -> Dict[str, np.ndarray]:
    """
    Concatenate multiple 3D-WRF files along Time and interpolate each variable to
    the specified pressure levels, handling staggered grids and half-levels.

    Parameters
    ----------
    file_paths : List[str]
        Paths to 3D WRF NetCDF files (must contain 'P','Z','TK','U','V','QVAPOR').
    target_levels : List[float]
        Pressure levels in hPa at which to interpolate.

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping {'z','t','u','v','q'} -> arrays shaped
        (ntime, n_levels, south_north, west_east).
    """
    # Load & concatenate
    ds_list = [xr.open_dataset(Path(p), engine="netcdf4", decode_times=False)
               for p in file_paths]
    ds = xr.concat(ds_list, dim="Time")
    for d in ds_list:
        d.close()

    # Map keys to WRF variable names
    wrf_map = {"z": "Z", "t": "TK", "u": "U", "v": "V", "q": "QVAPOR"}
    p_da = ds["P"]  # pressure DataArray in Pa

    # Dimensions
    nt = ds.dims["Time"]
    nl = len(target_levels)
    ny, nx = ds.dims["south_north"], ds.dims["west_east"]

    atom_interp: Dict[str, np.ndarray] = {}

    for key, varname in wrf_map.items():
        da = ds[varname]

        # Z has one extra half-level vertically; slice to match pressure levels
        if varname == "Z":
            da = da.isel(bottom_top=slice(0, p_da.shape[1]))

        # Destagger U and V onto the mass grid
        if varname == "U":
            da = destagger(da, "west_east")
        if varname == "V":
            da = destagger(da, "south_north")

        out = np.empty((nt, nl, ny, nx), dtype=np.float32)

        # Interpolate at each time and target level
        for ti in range(nt):
            da_t = da.isel(Time=ti)
            p_t = p_da.isel(Time=ti)
            for li, hpa in enumerate(target_levels):
                slice2d = interplevel(da_t, p_t, hpa * 100.0)
                out[ti, li, :, :] = slice2d.values.astype(np.float32)

        atom_interp[key] = out

    return atom_interp
