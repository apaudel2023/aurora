"""Utility functions for handling pressure level conversion between Pa and hPa."""

import dataclasses

from aurora.batch import Metadata


def convert_batch_pressure_levels(batch):
    """
    Convert batch pressure levels from Pa to hPa.

    Args:
        batch: Aurora batch with pressure levels in Pa

    Returns:
        Batch with pressure levels in hPa
    """
    # Convert pressure levels to hPa
    hpa_levels = tuple(level / 100.0 for level in batch.metadata.atmos_levels)

    # Create new metadata with converted pressure levels
    new_metadata = Metadata(
        lat=batch.metadata.lat,
        lon=batch.metadata.lon,
        time=batch.metadata.time,
        atmos_levels=hpa_levels,
    )

    # Create new batch with converted pressure levels
    return dataclasses.replace(batch, metadata=new_metadata)
