import logging
from datetime import datetime, timedelta

import torch

from aurora.batch import Batch, Metadata


def create_aurora_batch(grid_height=17, grid_width=32, num_levels=13, num_timesteps=4):
    """
    Create a sample batch for the Aurora model with customizable grid dimensions.

    Args:
        grid_height (int): Height of the grid
        grid_width (int): Width of the grid
        num_levels (int): Number of atmospheric levels
        num_timesteps (int): Number of timesteps to include in the batch

    Returns:
        Batch: An Aurora batch with random data
    """
    # Generate pressure levels between 1000 hPa (surface) and 100 hPa (upper atmosphere)
    # Ensure we have num_levels evenly spaced levels between 100 and 1000
    pressure_levels = tuple(int(level) for level in torch.linspace(100, 1000, num_levels).tolist())
    logging.info(f"Generated {len(pressure_levels)} pressure levels: {pressure_levels}")

    # Generate timesteps
    base_time = datetime(2020, 6, 1, 0, 0)
    time_points = tuple(base_time + timedelta(hours=6 * i) for i in range(num_timesteps))
    logging.info(f"Generated {len(time_points)} timesteps: {time_points}")

    # Create a batch with the specified dimensions and timesteps
    batch = Batch(
        surf_vars={
            k: torch.randn(1, num_timesteps, grid_height, grid_width) for k in ("2t", "10u", "10v")
        },
        static_vars={k: torch.randn(grid_height, grid_width) for k in ("z",)},
        atmos_vars={
            k: torch.randn(1, num_timesteps, num_levels, grid_height, grid_width)
            for k in ("z", "u", "v", "t", "q")
        },
        metadata=Metadata(
            lat=torch.linspace(40, -90, grid_height),
            lon=torch.linspace(0, 260, grid_width + 1)[:-1],
            time=time_points,
            atmos_levels=pressure_levels,
        ),
    )

    # Log batch data shapes
    logging.info("\nBatch Data Shapes:")
    logging.info("Surface Variables:")
    for var_name, tensor in batch.surf_vars.items():
        logging.info(f"  {var_name}: {tensor.shape}")

    logging.info("\nStatic Variables:")
    for var_name, tensor in batch.static_vars.items():
        logging.info(f"  {var_name}: {tensor.shape}")

    logging.info("\nAtmospheric Variables:")
    for var_name, tensor in batch.atmos_vars.items():
        logging.info(f"  {var_name}: {tensor.shape}")

    logging.info("\nMetadata:")
    logging.info(f"  lat: {batch.metadata.lat.shape}")
    logging.info(f"  lon: {batch.metadata.lon.shape}")
    logging.info(f"  time: {len(batch.metadata.time)} timesteps - {batch.metadata.time}")
    logging.info(f"  atmos_levels: {len(batch.metadata.atmos_levels)} levels")

    return batch
