# fine_tuning/data_loader/load_wrf.py

from pathlib import Path
import numpy as np
import xarray as xr


def load_and_combine_wrf_files(
    path_2d_1,
    path_2d_2,
    path_3d_1,
    path_3d_2,
    path_static,
):
    """
    Load two 2D and two 3D WRF files plus one static constants file, and return four dicts:
      meta_data   : {'lat':(H,), 'lon':(W,), 'time':(2,), 'pressure_levels':(levels,H,W)}
      surf_vars   : {'t2','u10','v10','psfc'} each (2, H, W)
      static_vars : {'z','slt','lsm'} each (H, W)
      atom_vars   : {'z','t','u','v','q'} each (2, levels, H, W)
    """
    # Paths
    f2d1, f2d2 = Path(path_2d_1), Path(path_2d_2)
    f3d1, f3d2 = Path(path_3d_1), Path(path_3d_2)
    f_static = Path(path_static)

    # --- 2D datasets ---
    ds2_1 = xr.open_dataset(f2d1, engine="netcdf4", decode_times=False)
    ds2_2 = xr.open_dataset(f2d2, engine="netcdf4", decode_times=False)

    raw_lat = ds2_1["XLAT"].values
    raw_lon = ds2_1["XLONG"].values

    # Reduce to 1D lat/lon
    if raw_lat.ndim == 3:
        lat2d = raw_lat[0]
    elif raw_lat.ndim == 2:
        lat2d = raw_lat
    else:
        lat1d = raw_lat.astype(np.float32)

    if raw_lon.ndim == 3:
        lon2d = raw_lon[0]
    elif raw_lon.ndim == 2:
        lon2d = raw_lon
    else:
        lon1d = raw_lon.astype(np.float32)

    if "lat2d" in locals():
        lat1d = lat2d[:, 0].astype(np.float32)
    if "lon2d" in locals():
        lon1d = lon2d[0, :].astype(np.float32)

    times = np.array([ds2_1["Times"].values[0], ds2_2["Times"].values[0]])
    meta_data = {"lat": lat1d, "lon": lon1d, "time": times}

    # Surface vars
    surf_vars = {}
    for key, var in [("t2", "T2"), ("u10", "U10"), ("v10", "V10"), ("psfc", "PSFC")]:
        a1 = ds2_1[var].values
        a2 = ds2_2[var].values
        if a1.ndim == 2:
            a1 = a1[None]; a2 = a2[None]
        surf_vars[key] = np.concatenate([a1, a2], axis=0).astype(np.float32)

    ds2_1.close(); ds2_2.close()

    # --- Static constants file ---
    ds_stat = xr.open_dataset(f_static, engine="netcdf4", decode_times=False)
    static_mapping = {
        "z": "PHB",
        "slt": "ISLTYP",
        "lsm": "LANDMASK",
    }
    static_vars = {}
    for out_key, var_name in static_mapping.items():
        arr = ds_stat[var_name].values
        # strip any extra dims to get (H, W)
        if arr.ndim == 3:
            arr = arr[0]
        elif arr.ndim == 4:
            arr = arr[0, 0]
        static_vars[out_key] = arr.astype(np.float32)
    ds_stat.close()

    # --- 3D datasets ---
    ds3_1 = xr.open_dataset(f3d1, engine="netcdf4", decode_times=False)
    ds3_2 = xr.open_dataset(f3d2, engine="netcdf4", decode_times=False)

    raw_p = ds3_1["P"].values
    p3d = raw_p[0] if raw_p.ndim == 4 else raw_p
    meta_data["pressure_levels"] = p3d.astype(np.float32)

    atom_vars = {}
    mapping_3d = {"z": "Z", "t": "TK", "u": "U", "v": "V", "q": "QVAPOR"}
    for key, var in mapping_3d.items():
        b1 = ds3_1[var].values; b2 = ds3_2[var].values
        if b1.ndim == 3:
            b1 = b1[None]; b2 = b2[None]
        atom_vars[key] = np.concatenate([b1, b2], axis=0).astype(np.float32)

    ds3_1.close(); ds3_2.close()

    # (Optional) print shapes for verification
    print("static shapes:", {k: v.shape for k, v in static_vars.items()})

    return meta_data, surf_vars, static_vars, atom_vars
