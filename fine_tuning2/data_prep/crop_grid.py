# fine_tuning2/prep/grid_cropper.py

import logging
import numpy as np
from typing import Tuple

class GridCropper:
    """
    Crop spatial grids so both dimensions become divisible by patch_size.
    Trims from the end (highest indices), and can apply the same crop
    to 1D latitude/longitude arrays without re‑passing the removal counts.
    """

    def __init__(self, patch_size: int):
        self.patch       = patch_size
        self.logger      = logging.getLogger(self.__class__.__name__)
        # store last removal
        self.last_removed_h = 0
        self.last_removed_w = 0

    def crop_grid(
        self,
        arr: np.ndarray
    ) -> Tuple[np.ndarray, int, int]:
        """
        Crop the last two dims of `arr` so they’re divisible by patch_size.
        Stores the removed row/col counts internally.

        Returns
        -------
        cropped : same dtype, shape (..., H', W')
        removed_h : number of rows removed
        removed_w : number of cols removed
        """
        *rest, H, W = arr.shape
        H_crop = H - (H % self.patch)
        W_crop = W - (W % self.patch)
        removed_h = H - H_crop
        removed_w = W - W_crop

        # remember for later
        self.last_removed_h = removed_h
        self.last_removed_w = removed_w

        self.logger.debug(
            "Cropping grid from %d×%d to %d×%d (−%d rows, −%d cols)",
            H, W, H_crop, W_crop, removed_h, removed_w
        )

        idx = tuple(slice(None) for _ in rest) + (slice(0, H_crop), slice(0, W_crop))
        cropped = arr[idx]
        return cropped, removed_h, removed_w

    def crop_latlon(
        self,
        lat: np.ndarray,
        lon: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crop the 1D lat/lon according to the most recent 2D crop.
        Uses the internally stored last_removed_h/w.
        """
        rh, rw = self.last_removed_h, self.last_removed_w

        # latitude: drop last rh entries
        if rh > 0:
            lat_c = lat[:-rh].copy()
            self.logger.debug("Cropping lat: %d → %d (−%d)", lat.size, lat_c.size, rh)
        else:
            lat_c = lat.copy()
            self.logger.debug("No lat cropping needed: %d", lat.size)

        # longitude: drop last rw entries
        if rw > 0:
            lon_c = lon[:-rw].copy()
            self.logger.debug("Cropping lon: %d → %d (−%d)", lon.size, lon_c.size, rw)
        else:
            lon_c = lon.copy()
            self.logger.debug("No lon cropping needed: %d", lon.size)

        return lat_c.astype(np.float32), lon_c.astype(np.float32)
