# fine_tuning2/batch/build_batch.py

import torch
from typing import Dict, Any
from aurora import Batch, Metadata
import numpy as np


class BuildBatch:
    """
    Build an Aurora Batch from full‐series data + metadata dict.
    """

    def __init__(self, history: int = 2, start_index: int = 0):
        """
        history: how many consecutive frames to include
        start_index: index of the first frame in that window
        """
        self.history     = history
        self.start_index = start_index

    def make(
        self,
        metadata: Dict[str, Any],
        surf_full: Dict[str, np.ndarray],
        static: Dict[str, np.ndarray],
        atmos_full: Dict[str, np.ndarray],
    ) -> Batch:
        """
        metadata: {
          'lat': 1D array,
          'lon': 1D array,
          'time': tuple length N,
          'atmos_levels': tuple of levels
        }
        surf_full: dict of arrays (N, H, W)
        atmos_full: dict of arrays (N, plev, H, W)
        static: dict of arrays (H, W)
        """
        H = self.history
        S = self.start_index
        start = S
        end   = S + H  # exclusive slice

        # 1) Surf & atmos: slice [start:end], then add batch dim
        surf_tensors = {
            key: torch.from_numpy(val[start:end][None])
            for key, val in surf_full.items()
        }
        atmos_tensors = {
            key: torch.from_numpy(val[start:end][None])
            for key, val in atmos_full.items()
        }

        # 2) Static (constant)
        static_tensors = {
            key: torch.from_numpy(val.astype("float32"))
            for key, val in static.items()
        }

        # 3) Metadata: pick the last time in the window
        time_pt = metadata["time"][end - 1]
        meta = Metadata(
            lat=torch.from_numpy(metadata["lat"]),
            lon=torch.from_numpy(metadata["lon"]),
            time=(time_pt,),
            atmos_levels=metadata["atmos_levels"]
        )

        # 4) Build and return the Batch
        return Batch(
            surf_vars=surf_tensors,
            static_vars=static_tensors,
            atmos_vars=atmos_tensors,
            metadata=meta
        )
