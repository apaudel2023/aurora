
from typing import Dict, Optional
import logging

import numpy as np
import xarray as xr


class StaticReader:
    def __init__(
        self,
        path: str,
        engine: str = "netcdf4",
        decode_times: bool = False,
        var_map: Optional[Dict[str, str]] = None,
    ):
        self.path = path
        self.engine = engine
        self.decode_times = decode_times
        self.var_map = var_map or {
            "z": "PHB",
            "slt": "ISLTYP",
            "lsm": "LANDMASK",
        }
        self.logger = logging.getLogger(self.__class__.__name__)

    def load(self) -> Dict[str, np.ndarray]:
        """
        Load WRF static constants file and return a dict with three arrays:
          - 'z'   : geopotential base height (PHB)
          - 'slt' : soil type (ISLTYP)
          - 'lsm' : land-sea mask (LANDMASK)

        Each array is coerced to 2D (south_north, west_east) and dtype float32.
        """
        ds = xr.open_dataset(
            self.path, engine=self.engine, decode_times=self.decode_times
        )
        out: Dict[str, np.ndarray] = {}

        for out_key, var_name in self.var_map.items():
            arr = ds[var_name].data  # could be 2D or 3D (or more)

            # If there are extra leading dims (e.g. bottom_top, Time), slice the 0th index
            while arr.ndim > 2:
                arr = arr[0]

            # Now we must have exactly 2 dims
            if arr.ndim != 2:
                raise ValueError(
                    f"Static var '{var_name}' is {arr.ndim}D with shape {arr.shape}; "
                    "expected 2D (south_north, west_east) after slicing."
                )

            out[out_key] = arr.astype(np.float32)
            self.logger.info(
                "Loaded static '%s' from '%s' with shape %s",
                out_key, var_name, arr.shape
            )

        ds.close()
        return out
