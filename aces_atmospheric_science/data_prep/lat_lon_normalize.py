# fine_tuning2/prep/latlon_normalizer.py

import logging
import numpy as np
from typing import Tuple

class LatLonNormalizer:
    """
    Normalize and reorder latitude/longitude arrays:
      - lon: wrap into [0,360), then ensure strictly increasing
      - lat: clip into [-90,90], then ensure strictly decreasing
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def normalize(
        self,
        lat: np.ndarray,
        lon: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Wrap longitude into [0,360)
        lon360 = np.mod(lon + 360.0, 360.0).astype(np.float32)
        self.logger.info("Wrapped lon into [0,360): min=%.2f, max=%.2f",
                          lon360.min(), lon360.max())

        # Clip latitude into [-90,90]
        lat_clipped = np.clip(lat, -90.0, 90.0).astype(np.float32)
        self.logger.info("Clipped lat into [-90,90]: min=%.2f, max=%.2f",
                          lat_clipped.min(), lat_clipped.max())

        # Enforce strict monotonicity
        if not np.all(lat_clipped[1:] < lat_clipped[:-1]):
            lat_clipped = np.sort(lat_clipped)[::-1]
            self.logger.info("Re-sorted lat to be strictly decreasing.")
        else:
            self.logger.info("Lat already strictly decreasing.")

        if not np.all(lon360[1:] > lon360[:-1]):
            lon360 = np.sort(lon360)
            self.logger.info("Re-sorted lon to be strictly increasing.")
        else:
            self.logger.info("Lon already strictly increasing.")

        return lat_clipped, lon360
