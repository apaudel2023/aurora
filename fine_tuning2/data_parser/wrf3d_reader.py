# fine_tuning2/data_parser/wrf3d_reader.py

import logging
from typing import Dict, List, Optional

import numpy as np
import xarray as xr
from wrf import interplevel, destagger


class WRF3DReader:
    """
    Reader & processor for WRF 3D snapshot files. Extracts configured
    atmospheric fields and interpolates them to the specified pressure
    levels (in hPa), handling half‐level slicing for Z and destaggering
    for U/V.
    """

    def __init__(
        self,
        path: str,
        var_map: Optional[Dict[str, str]] = None,
        engine: str = "netcdf4",
        decode_times: bool = False,
    ):
        """
        var_map: maps output keys → WRF variable names, e.g.
          {
            "z":      "Z",
            "t":      "TK",
            "u":      "U",
            "v":      "V",
            "q":      "QVAPOR",
          }
        """
        self.path         = path
        self.engine       = engine
        self.decode_times = decode_times
        self.logger       = logging.getLogger(self.__class__.__name__)

        # Use provided map or default
        self.var_map = var_map or {
            "z": "Z",
            "t": "TK",
            "u": "U",
            "v": "V",
            "q": "QVAPOR",
        }

    def load_and_adjust(
        self,
        target_levels: List[float]
    ) -> Dict[str, np.ndarray]:
        ds = xr.open_dataset(self.path,
                             engine=self.engine,
                             decode_times=self.decode_times)
        self.logger.info("Opened 3D file %s", self.path)

        # 1) build pressure DataArray on the mass grid
        p_da = ds["P"]
        if "Time" in p_da.dims:
            p_da = p_da.isel(Time=0)
        p_da = p_da.astype(np.float32)

        nt   = ds.sizes.get("Time", 1)
        nlev = len(target_levels)
        ny   = p_da.sizes["south_north"]
        nx   = p_da.sizes["west_east"]

        output: Dict[str, np.ndarray] = {}

        # 2) loop over configured variables
        for out_key, wrf_name in self.var_map.items():
            da = ds[wrf_name]
            # ensure time dimension
            if "Time" not in da.dims:
                da = da.expand_dims("Time")

            # half‐level slicing for geopotential (z)
            if out_key == "z":
                # WRF uses "bottom_top_stag" on Z
                da = da.isel(bottom_top_stag=slice(0, p_da.sizes["bottom_top"]))
                da = da.rename({"bottom_top_stag": "bottom_top"})
                self.logger.debug("Sliced Z → %s", da.shape)

            # destagger for U and V
            if out_key == "u":
                ax = da.get_axis_num("west_east_stag")
                da = destagger(da, ax, meta=True)
                self.logger.debug("Destaggered U → %s", da.shape)
            elif out_key == "v":
                ax = da.get_axis_num("south_north_stag")
                da = destagger(da, ax, meta=True)
                self.logger.debug("Destaggered V → %s", da.shape)

            # 3) interpolate each time & level
            arr = np.empty((nt, nlev, ny, nx), dtype=np.float32)
            for ti in range(da.sizes["Time"]):
                da_t = da.isel(Time=ti)
                for li, hpa in enumerate(target_levels):
                    sl2d = interplevel(da_t, p_da, hpa * 100.0)
                    arr[ti, li, :, :] = sl2d.values.astype(np.float32)

            output[out_key] = arr
            self.logger.info("Processed %-1s → shape %s", out_key, arr.shape)

        ds.close()
        return output
