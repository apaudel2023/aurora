# fine_tuning2/data_parser/wrf2d_reader.py

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import xarray as xr


class WRF2DReader:
    def __init__(
        self,
        path: str,
        var_map: Dict[str, str],
        engine: str = "netcdf4",
        decode_times: bool = False,
    ):
        self.path = Path(path)
        self.var_map = var_map
        self.engine = engine
        self.decode_times = decode_times
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

    def load_metadata(self) -> Dict[str, np.ndarray]:
        """
        Load lat, lon, and time from a single 2D WRF file.

        Returns a dict:
          {
            'lat': 1D array (south_north,),
            'lon': 1D array (west_east,),
            'time': 1D array of length ntime (often 1)
          }
        """
        ds = xr.open_dataset(
            self.path, engine=self.engine, decode_times=self.decode_times
        )

        # --- Latitude → 1D ---
        raw_lat = ds[self.var_map["lat"]].data
        if raw_lat.ndim == 3:
            lat1d = raw_lat[0, :, 0]
        elif raw_lat.ndim == 2:
            lat1d = raw_lat[:, 0]
        else:
            lat1d = raw_lat

        # --- Longitude → 1D ---
        raw_lon = ds[self.var_map["lon"]].data
        if raw_lon.ndim == 3:
            lon1d = raw_lon[0, 0, :]
        elif raw_lon.ndim == 2:
            lon1d = raw_lon[0, :]
        else:
            lon1d = raw_lon

        # --- Time → flatten into 1D array ---
        raw_time = ds[self.var_map["time"]].data
        # Ensure it's at least 1D, then flatten
        time_arr = np.atleast_1d(raw_time).flatten()

        ds.close()

        self.logger.info(
            "Loaded metadata: lat.shape=%s, lon.shape=%s, time.shape=%s",
            lat1d.shape, lon1d.shape, time_arr.shape
        )
        return {
            "lat": lat1d.astype(np.float32),
            "lon": lon1d.astype(np.float32),
            "time": time_arr,           # e.g. dtype '|S19' or datetime64
        }

    def load_surface(self) -> Dict[str, np.ndarray]:
        """
        Load surface variables (e.g. t2, u10, v10, psfc) from the same 2D file.

        Returns a dict mapping each key to an array of shape (ntime, south_north, west_east).
        """
        ds = xr.open_dataset(
            self.path, engine=self.engine, decode_times=self.decode_times
        )
        surf: Dict[str, np.ndarray] = {}

        # iterate over the var_map, skipping lat/lon/time
        for out_key, var_name in self.var_map.items():
            if out_key in ("lat", "lon", "time"):
                continue

            arr = ds[var_name].data
            # if 2D, add time axis
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]        # (1, H, W)
            # if already has time axis first, leave as-is
            surf[out_key] = arr.astype(np.float32)
            self.logger.info(
                "Loaded surface '%s' shape %s", out_key, arr.shape
            )

        ds.close()
        return surf
