"""
WRF Data Loader for Aurora Fine-tuning

This module provides a data loader for WRF model output that can efficiently
handle large datasets and prepare them for Aurora model fine-tuning.
"""

import glob
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import xarray as xr

# Setup logging with timestamped filename
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Create a timestamp for the log filename
timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
log_file = log_dir / f"wrf_data_loader_{timestamp}.log"

# Configure logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d:%(funcName)s] - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("WrfDataLoader")
logger.info(f"Starting new log session at {timestamp}, writing to {log_file}")


class WrfDataLoader:
    """
    Efficient loader for WRF data that handles large datasets and prepares
    batches for Aurora model fine-tuning.
    """

    def __init__(self, data_2d_path, data_3d_path):
        """
        Initialize the WRF data loader

        Args:
            data_2d_path (str): Path to 2D surface data files
            data_3d_path (str): Path to 3D atmospheric data files
        """
        logger.info(
            f"Initializing WrfDataLoader with 2D path: {data_2d_path}, 3D path: {data_3d_path}"
        )
        self.data_2d_path = data_2d_path
        self.data_3d_path = data_3d_path

        # Variable mapping from WRF to Aurora
        self.wrf_to_aurora = {
            # Static variables
            "Z": "z",
            "LANDMASK": "lsm",  # Optional - may not be present
            "LU_INDEX": "slt",  # Optional - may not be present
            # Surface variables
            "T2": "t2m",
            "U10": "u10",
            "V10": "v10",
            "PSFC": "msl",  # Will use surface pressure if mean sea level not available
            # Atmospheric variables
            "TK": "t",  # Temperature
            "U": "u",  # U wind component
            "V": "v",  # V wind component
            "QVAPOR": "q",  # Specific humidity
        }

        # Alternative variable names in case primary ones aren't found
        self.alternative_var_names = {
            "z": ["HGT", "TERRAIN"],
            "lsm": ["LAND"],
            "slt": ["SOILCAT", "ISLTYP"],
            "msl": ["SLP", "PMSL"],
            "t": ["T"],
            "q": ["Q"],
        }

        # Datasets to keep open during batch creation (for efficiency)
        self.open_datasets = {}

        # Scan files without loading them
        self.available_timesteps = self._scan_available_timesteps()

        # Load a sample 3D file to get the number of vertical levels
        self.pressure_levels = None
        self._get_pressure_levels()

    def _get_pressure_levels(self):
        """
        Get pressure level information from a sample 3D file
        """
        if not self.available_timesteps:
            logger.warning("No timesteps available, cannot determine pressure levels")
            return

        try:
            # Get the first available 3D file
            sample_time = next(iter(self.available_timesteps))
            _, file_3d = self.available_timesteps[sample_time]

            # Load just the dimensions information - disable time decoding to prevent errors
            with xr.open_dataset(file_3d, decode_times=False) as ds:
                # Check if file has explicit pressure levels
                if "P" in ds:
                    logger.info("Using pressure levels from 'P' variable")
                    # In a real implementation, you would extract actual pressure levels here
                    # For now, just create a sequence of levels
                    num_levels = ds.dims.get("bottom_top", 50)
                    self.pressure_levels = tuple(range(1, num_levels + 1))
                    logger.info(f"Found {num_levels} vertical levels")
                else:
                    # Use model levels as indices
                    num_levels = ds.dims.get("bottom_top", 50)
                    logger.info(f"No pressure field found, using {num_levels} model levels")
                    self.pressure_levels = tuple(range(1, num_levels + 1))
        except Exception as e:
            logger.error(f"Error determining pressure levels: {e}")
            # Default to 50 levels if we can't determine
            self.pressure_levels = tuple(range(1, 51))
            logger.warning("Using default 50 vertical levels")

    def __del__(self):
        """Clean up any open datasets when the object is destroyed"""
        logger.info("Cleaning up WrfDataLoader resources")
        self.close_datasets()

    def close_datasets(self):
        """Close all open datasets to free memory"""
        logger.info(f"Closing {len(self.open_datasets)} open datasets")
        for ds in self.open_datasets.values():
            ds.close()
        self.open_datasets = {}
        xr.Dataset.close_all()  # Ensure all datasets are closed

    def _extract_timestamp(self, filename):
        """
        Extract timestamp from WRF filename

        Args:
            filename (str): WRF output filename

        Returns:
            datetime: Extracted timestamp
        """
        try:
            basename = os.path.basename(filename)
            # Handle standard WRF naming format: prefix_domain_YYYY-MM-DD_HH:MM:SS.nc
            parts = basename.split("_")
            date_part = parts[2]
            time_part = parts[3].split(".")[0]
            time_str = f"{date_part}_{time_part}"
            dt = datetime.strptime(time_str, "%Y-%m-%d_%H:%M:%S")
            logger.debug(f"Extracted timestamp {dt} from {basename}")
            return dt
        except (IndexError, ValueError) as e:
            logger.error(f"Failed to extract timestamp from {filename}: {e}")
            return None

    def _scan_available_timesteps(self):
        """
        Scan directories and identify available timesteps that have both 2D and 3D data

        Returns:
            dict: Mapping of timestamps to (2d_file, 3d_file) tuples
        """
        logger.info("Scanning for available timesteps in data directories")

        # Get sorted lists of files
        files_2d = sorted(glob.glob(os.path.join(self.data_2d_path, "*.nc")))
        files_3d = sorted(glob.glob(os.path.join(self.data_3d_path, "*.nc")))

        logger.info(f"Found {len(files_2d)} 2D files and {len(files_3d)} 3D files")

        if not files_2d:
            logger.error(f"No 2D netCDF files found in {self.data_2d_path}")
            raise ValueError(f"No 2D netCDF files found in {self.data_2d_path}")
        if not files_3d:
            logger.error(f"No 3D netCDF files found in {self.data_3d_path}")
            raise ValueError(f"No 3D netCDF files found in {self.data_3d_path}")

        # Map timestamps to files
        timesteps_2d = {}
        for f in files_2d:
            timestamp = self._extract_timestamp(f)
            if timestamp:
                timesteps_2d[timestamp] = f

        timesteps_3d = {}
        for f in files_3d:
            timestamp = self._extract_timestamp(f)
            if timestamp:
                timesteps_3d[timestamp] = f

        logger.info(
            f"Extracted {len(timesteps_2d)} timestamps from 2D files and {len(timesteps_3d)} from 3D files"
        )

        # Find common timesteps
        common_times = sorted(set(timesteps_2d.keys()).intersection(set(timesteps_3d.keys())))

        if not common_times:
            logger.warning("No matching timesteps found between 2D and 3D files")
            return {}

        # Create mapping of timestamp -> (2d_file, 3d_file)
        timestep_files = {t: (timesteps_2d[t], timesteps_3d[t]) for t in common_times}

        logger.info(f"Found {len(timestep_files)} matching timesteps between 2D and 3D files")
        if common_times:
            logger.info(f"Time range: {min(common_times)} to {max(common_times)}")

        return timestep_files

    def get_timestep_list(self):
        """
        Get list of available timesteps

        Returns:
            list: Sorted list of datetime objects representing available timesteps
        """
        timesteps = sorted(self.available_timesteps.keys())
        logger.info(f"Returning list of {len(timesteps)} available timesteps")
        return timesteps

    def load_timestep(self, timestep, cache=True):
        """
        Load data for a specific timestep

        Args:
            timestep (datetime): Datetime object representing the timestep to load
            cache (bool): Whether to cache the dataset in memory

        Returns:
            tuple: (ds_2d, ds_3d) for the requested timestep
        """
        logger.info(f"Loading data for timestep {timestep}, cache={cache}")

        if timestep not in self.available_timesteps:
            logger.error(f"Timestep {timestep} not available in the dataset")
            raise ValueError(f"Timestep {timestep} not available in the dataset")

        # Check if already cached
        cache_key_2d = f"2d_{timestep}"
        cache_key_3d = f"3d_{timestep}"

        if cache and cache_key_2d in self.open_datasets and cache_key_3d in self.open_datasets:
            logger.debug(f"Using cached datasets for timestep {timestep}")
            return self.open_datasets[cache_key_2d], self.open_datasets[cache_key_3d]

        # Load files
        file_2d, file_3d = self.available_timesteps[timestep]
        logger.debug(f"Loading files: {os.path.basename(file_2d)}, {os.path.basename(file_3d)}")

        try:
            # Disable time decoding to prevent errors with WRF time format
            ds_2d = xr.open_dataset(file_2d, decode_times=False)
            ds_3d = xr.open_dataset(file_3d, decode_times=False)

            logger.debug(
                f"Successfully loaded datasets with shapes - 2D: {ds_2d.dims}, 3D: {ds_3d.dims}"
            )

            # Cache if requested
            if cache:
                self.open_datasets[cache_key_2d] = ds_2d
                self.open_datasets[cache_key_3d] = ds_3d

            return ds_2d, ds_3d

        except Exception as e:
            logger.error(f"Error loading data for timestep {timestep}: {e}")
            raise

    def _destagger_variable(self, data, stagger_dim):
        """
        Destagger a variable along the specified dimension

        Args:
            data (np.ndarray): Variable data
            stagger_dim (int): Dimension to destagger (0 for time, 2 for y, 3 for x)

        Returns:
            np.ndarray: Destaggered data
        """
        logger.debug(f"Destaggering variable along dimension {stagger_dim}")

        # Simple averaging method for destaggering
        if data.shape[stagger_dim] <= 1:
            return data

        # Create slices for the staggered dimension
        slice1 = [slice(None)] * data.ndim
        slice2 = [slice(None)] * data.ndim

        slice1[stagger_dim] = slice(0, -1)
        slice2[stagger_dim] = slice(1, None)

        # Average adjacent points to destagger
        return 0.5 * (data[tuple(slice1)] + data[tuple(slice2)])

    def _fix_staggered_grid(self, ds_2d, ds_3d):
        """
        Handle staggered variables (U, V) in WRF output

        Args:
            ds_2d (xarray.Dataset): 2D dataset
            ds_3d (xarray.Dataset): 3D dataset

        Returns:
            tuple: Processed 2D and 3D datasets with destaggered variables
        """
        logger.info("Checking and fixing staggered grid variables")

        # Check for staggered dimensions in U (west_east_stag)
        if "U" in ds_2d and ds_2d["U"].shape[-1] > ds_2d.dims["west_east"]:
            logger.debug(f"Destaggering U in 2D file: shape {ds_2d['U'].shape}")
            # Destagger U in x-dimension
            ds_2d["U"] = xr.DataArray(
                self._destagger_variable(ds_2d["U"].values, -1),
                dims=ds_2d["T2"].dims,
                coords=ds_2d["T2"].coords,
            )

        if "U" in ds_3d and ds_3d["U"].shape[-1] > ds_3d.dims["west_east"]:
            logger.debug(f"Destaggering U in 3D file: shape {ds_3d['U'].shape}")
            # Destagger U in x-dimension
            ds_3d["U"] = xr.DataArray(
                self._destagger_variable(ds_3d["U"].values, -1),
                dims=ds_3d["TK"].dims,
                coords=ds_3d["TK"].coords,
            )

        # Check for staggered dimensions in V (south_north_stag)
        if "V" in ds_2d and ds_2d["V"].shape[-2] > ds_2d.dims["south_north"]:
            logger.debug(f"Destaggering V in 2D file: shape {ds_2d['V'].shape}")
            # Destagger V in y-dimension
            ds_2d["V"] = xr.DataArray(
                self._destagger_variable(ds_2d["V"].values, -2),
                dims=ds_2d["T2"].dims,
                coords=ds_2d["T2"].coords,
            )

        if "V" in ds_3d and ds_3d["V"].shape[-2] > ds_3d.dims["south_north"]:
            logger.debug(f"Destaggering V in 3D file: shape {ds_3d['V'].shape}")
            # Destagger V in y-dimension
            ds_3d["V"] = xr.DataArray(
                self._destagger_variable(ds_3d["V"].values, -2),
                dims=ds_3d["TK"].dims,
                coords=ds_3d["TK"].coords,
            )

        logger.debug("Staggered grid check and fix completed")
        return ds_2d, ds_3d

    def create_batch_iterator(self, batch_size=1, sequence_length=2):
        """
        Create an iterator that yields batches of data for Aurora model

        Args:
            batch_size (int): Number of samples in each batch
            sequence_length (int): Number of consecutive timesteps in each sample

        Yields:
            Batch: An Aurora batch object for each set of timesteps
        """
        try:
            from aurora import Batch, Metadata
        except ImportError:
            logger.error("Aurora package not found. Please install it.")
            raise ImportError("Aurora package is required. Please install it first.")

        # Get all available timesteps
        all_timesteps = sorted(self.available_timesteps.keys())

        if len(all_timesteps) < sequence_length:
            logger.error(
                f"Not enough timesteps available. Need at least {sequence_length}, but only have {len(all_timesteps)}"
            )
            raise ValueError(
                f"Not enough timesteps available. Need at least {sequence_length}, but only have {len(all_timesteps)}"
            )

        logger.info(
            f"Creating batch iterator with batch_size={batch_size}, sequence_length={sequence_length}"
        )
        logger.info(f"Will process {len(all_timesteps) - sequence_length + 1} potential samples")

        # Process data in batches
        for batch_start in range(0, len(all_timesteps) - sequence_length + 1, batch_size):
            logger.info(f"Processing batch starting at index {batch_start}")

            # Prepare data for this batch
            batch_surf_vars = defaultdict(list)
            batch_atmos_vars = defaultdict(list)
            batch_static_vars = {}
            batch_times = []
            batch_lats = None
            batch_lons = None

            # Process each sample in the batch
            for sample_idx in range(batch_size):
                if batch_start + sample_idx >= len(all_timesteps) - sequence_length + 1:
                    logger.debug(f"Not enough data left for sample {sample_idx}")
                    break  # Not enough data left for another complete sample

                # Get timesteps for this sample
                sample_timesteps = all_timesteps[
                    batch_start + sample_idx : batch_start + sample_idx + sequence_length
                ]
                batch_times.append(sample_timesteps[0])  # Use first timestep as reference

                logger.debug(f"Processing sample {sample_idx} with timesteps {sample_timesteps}")

                # Process each timestep in the sequence
                sample_surf_vars = defaultdict(list)
                sample_atmos_vars = defaultdict(list)

                for t_idx, timestep in enumerate(sample_timesteps):
                    # Load data for this timestep
                    ds_2d, ds_3d = self.load_timestep(timestep)

                    # Fix staggered grid variables
                    ds_2d, ds_3d = self._fix_staggered_grid(ds_2d, ds_3d)

                    # Store lat/lon from first timestep of first sample
                    if sample_idx == 0 and t_idx == 0:
                        logger.debug("Creating synthetic latitude and longitude arrays")

                        # Get grid dimensions
                        grid_height = ds_2d.dims["south_north"]
                        grid_width = ds_2d.dims["west_east"]

                        # Create synthetic lat/lon arrays like in the example
                        # Latitude: decreasing from north to south (90 to -90)
                        batch_lats = torch.linspace(90, -90, grid_height)

                        # Longitude: increasing from 0 to 360 (excluding 360)
                        batch_lons = torch.linspace(0, 360, grid_width + 1)[:-1]

                        logger.info(
                            f"Created synthetic lat/lon vectors with shapes: {batch_lats.shape}, {batch_lons.shape}"
                        )
                        logger.debug(
                            f"Latitude range: {batch_lats[0].item()} to {batch_lats[-1].item()}"
                        )
                        logger.debug(
                            f"Longitude range: {batch_lons[0].item()} to {batch_lons[-1].item()}"
                        )

                    # Store static variables from first timestep of first sample
                    if sample_idx == 0 and t_idx == 0:
                        logger.debug("Extracting static variables")
                        # Try primary and alternative variable names
                        for wrf_var, aurora_var in self.wrf_to_aurora.items():
                            if aurora_var in ["z", "lsm", "slt"]:  # Static variables
                                # Try to find in 2D file first
                                if wrf_var in ds_2d:
                                    logger.debug(
                                        f"Found static variable {aurora_var} from {wrf_var}"
                                    )
                                    batch_static_vars[aurora_var] = torch.tensor(
                                        ds_2d[wrf_var].values[0].astype(np.float32)
                                    )
                                    continue

                                # Try alternative names
                                alt_vars = self.alternative_var_names.get(aurora_var, [])
                                for alt_var in alt_vars:
                                    if alt_var in ds_2d:
                                        logger.debug(
                                            f"Found static variable {aurora_var} from alternative {alt_var}"
                                        )
                                        batch_static_vars[aurora_var] = torch.tensor(
                                            ds_2d[alt_var].values[0].astype(np.float32)
                                        )
                                        break

                    # Extract surface variables
                    for wrf_var, aurora_var in self.wrf_to_aurora.items():
                        if aurora_var in ["t2m", "u10", "v10", "msl"]:  # Surface variables
                            if wrf_var in ds_2d:
                                logger.debug(
                                    f"Extracting surface variable {aurora_var} from {wrf_var}"
                                )
                                sample_surf_vars[aurora_var].append(ds_2d[wrf_var].values[0])
                                continue

                            # Try alternative names
                            alt_vars = self.alternative_var_names.get(aurora_var, [])
                            for alt_var in alt_vars:
                                if alt_var in ds_2d:
                                    logger.debug(
                                        f"Extracting surface variable {aurora_var} from alternative {alt_var}"
                                    )
                                    sample_surf_vars[aurora_var].append(ds_2d[alt_var].values[0])
                                    break

                    # Extract atmospheric variables - directly use levels from 3D file
                    for wrf_var, aurora_var in self.wrf_to_aurora.items():
                        if aurora_var in ["t", "u", "v", "q"]:  # Atmospheric variables
                            if wrf_var in ds_3d:
                                logger.debug(
                                    f"Extracting atmospheric variable {aurora_var} from {wrf_var}"
                                )
                                sample_atmos_vars[aurora_var].append(ds_3d[wrf_var].values)
                                continue

                            # Try alternative names
                            alt_vars = self.alternative_var_names.get(aurora_var, [])
                            for alt_var in alt_vars:
                                if alt_var in ds_3d:
                                    logger.debug(
                                        f"Extracting atmospheric variable {aurora_var} from alternative {alt_var}"
                                    )
                                    sample_atmos_vars[aurora_var].append(ds_3d[alt_var].values)
                                    break

                # Stack timesteps for this sample's surface variables
                for var, data_list in sample_surf_vars.items():
                    if len(data_list) == sequence_length:  # Only if we have all timesteps
                        # Shape: [sequence_length, height, width]
                        stacked = np.stack(data_list)
                        # Add to batch list with shape [1, sequence_length, height, width]
                        batch_surf_vars[var].append(stacked[np.newaxis, :])
                        logger.debug(f"Stacked surface variable {var} with shape {stacked.shape}")

                # Stack timesteps for this sample's atmospheric variables
                for var, data_list in sample_atmos_vars.items():
                    if len(data_list) == sequence_length:  # Only if we have all timesteps
                        # Shape: [sequence_length, levels, height, width]
                        stacked = np.stack(data_list)
                        # Add to batch list with shape [1, sequence_length, levels, height, width]
                        batch_atmos_vars[var].append(stacked[np.newaxis, :])
                        logger.debug(
                            f"Stacked atmospheric variable {var} with shape {stacked.shape}"
                        )

            # Skip if no complete samples
            if not batch_surf_vars:
                logger.warning(f"No complete samples created for batch starting at {batch_start}")
                continue

            # Combine samples into batch tensors
            final_surf_vars = {}
            for var, samples in batch_surf_vars.items():
                if samples:
                    # Shape: [batch_size, sequence_length, height, width]
                    final_surf_vars[var] = torch.tensor(np.concatenate(samples).astype(np.float32))
                    logger.debug(
                        f"Created batch tensor for surface variable {var} with shape {final_surf_vars[var].shape}"
                    )

            final_atmos_vars = {}
            for var, samples in batch_atmos_vars.items():
                if samples:
                    # Shape: [batch_size, sequence_length, levels, height, width]
                    final_atmos_vars[var] = torch.tensor(np.concatenate(samples).astype(np.float32))
                    logger.debug(
                        f"Created batch tensor for atmospheric variable {var} with shape {final_atmos_vars[var].shape}"
                    )

            # Create and yield batch
            logger.info(
                f"Creating Aurora batch with {len(final_surf_vars)} surface variables, "
                f"{len(batch_static_vars)} static variables, and {len(final_atmos_vars)} atmospheric variables"
            )

            try:
                # Make sure we have the minimum required variables
                required_vars = ["t2m", "u10", "v10", "msl"]
                missing_vars = [var for var in required_vars if var not in final_surf_vars]
                if missing_vars:
                    logger.warning(f"Missing required surface variables: {missing_vars}")

                # Make sure we have valid lat/lon vectors
                if batch_lats is None or batch_lons is None:
                    logger.error("Missing latitude or longitude data")
                    raise ValueError("Missing latitude or longitude data")

                # Log metadata shapes for debugging
                logger.debug(
                    f"Metadata shapes - lat: {batch_lats.shape}, lon: {batch_lons.shape}, "
                    f"levels: {len(self.pressure_levels)}, times: {len(batch_times)}"
                )

                # Create Aurora batch
                aurora_batch = Batch(
                    surf_vars=final_surf_vars,
                    static_vars=batch_static_vars,
                    atmos_vars=final_atmos_vars,
                    metadata=Metadata(
                        lat=batch_lats,
                        lon=batch_lons,
                        time=tuple(batch_times),
                        atmos_levels=self.pressure_levels,
                    ),
                )

                logger.info(f"Successfully created batch {batch_start//batch_size}")
                yield aurora_batch

            except Exception as e:
                logger.error(f"Error creating Aurora batch: {e}")
                logger.error(
                    f"Batch shapes - surf_vars: {[v.shape for v in final_surf_vars.values()]}, "
                    f"atmos_vars: {[v.shape for v in final_atmos_vars.values()]}"
                )
                raise

            # Keep memory usage in check by periodically closing datasets
            # that are no longer needed for the next batch
            if batch_start % (10 * batch_size) == 0:
                logger.info("Performing periodic dataset cleanup")
                self.close_datasets()

    def create_single_batch(self, start_time=None, sequence_length=2):
        """
        Create a single batch starting at a specific time

        Args:
            start_time (datetime, optional): Starting time for the batch. If None, uses earliest available.
            sequence_length (int): Number of consecutive timesteps in the batch

        Returns:
            Batch: An Aurora batch object
        """
        logger.info(
            f"Creating single batch with sequence_length={sequence_length}, start_time={start_time}"
        )

        # Get all available timesteps
        all_timesteps = sorted(self.available_timesteps.keys())

        if not all_timesteps:
            logger.error("No timesteps available")
            raise ValueError("No timesteps available")

        # Determine start index
        if start_time is None:
            start_idx = 0
            logger.info(f"Using earliest available timestep: {all_timesteps[0]}")
        else:
            # Find closest timestep to requested start_time
            start_idx = 0
            min_diff = abs(all_timesteps[0] - start_time)

            for i, t in enumerate(all_timesteps):
                diff = abs(t - start_time)
                if diff < min_diff:
                    min_diff = diff
                    start_idx = i

            logger.info(
                f"Found closest timestep to {start_time}: {all_timesteps[start_idx]} (index {start_idx})"
            )

        # Check if we have enough timesteps
        if start_idx + sequence_length > len(all_timesteps):
            logger.error(f"Not enough timesteps available starting from {start_time}")
            raise ValueError(f"Not enough timesteps available starting from {start_time}")

        # Create an iterator with batch_size=1 and get the first batch
        logger.info(f"Creating batch iterator to extract batch at index {start_idx}")
        batch_iterator = self.create_batch_iterator(batch_size=1, sequence_length=sequence_length)

        try:
            for i, batch in enumerate(batch_iterator):
                if i == start_idx:
                    logger.info(
                        f"Returning batch for timestep starting at {all_timesteps[start_idx]}"
                    )
                    return batch
                if i > start_idx:
                    # We've gone past the desired batch
                    break
        except Exception as e:
            logger.error(f"Error creating batch: {e}")
            raise

        logger.error(f"Failed to create batch starting at {start_time}")
        raise ValueError(f"Failed to create batch starting at {start_time}")


# Example usage
if __name__ == "__main__":
    try:
        # Example configuration
        logger.info("Starting example usage in __main__")
        data_loader = WrfDataLoader(
            data_2d_path="/home/user/Documents/aurora/data_wrf/2d/",
            data_3d_path="/home/user/Documents/aurora/data_wrf/3d/",
        )

        # Print available timesteps
        timesteps = data_loader.get_timestep_list()
        logger.info(f"Available timesteps: {timesteps}")
        print(f"Available timesteps: {timesteps}")

        # Create a single batch for testing
        if timesteps and len(timesteps) >= 2:
            try:
                logger.info("Attempting to create a single batch")
                batch = data_loader.create_single_batch(start_time=timesteps[0], sequence_length=2)

                print("Created batch with shapes:")
                logger.info("Successfully created batch with shapes:")

                # Log surface variables
                for var_name, var_data in batch.surf_vars.items():
                    print(f"- Surface {var_name}: {var_data.shape}")
                    logger.info(f"- Surface {var_name}: {var_data.shape}")

                # Log static variables
                for var_name, var_data in batch.static_vars.items():
                    print(f"- Static {var_name}: {var_data.shape}")
                    logger.info(f"- Static {var_name}: {var_data.shape}")

                # Log atmospheric variables
                for var_name, var_data in batch.atmos_vars.items():
                    print(f"- Atmospheric {var_name}: {var_data.shape}")
                    logger.info(f"- Atmospheric {var_name}: {var_data.shape}")

            except Exception as e:
                logger.error(f"Error in batch creation: {e}")
                print(f"Error creating batch: {e}")
        else:
            logger.warning("Not enough timesteps available for sequence_length=2")
            print("Not enough timesteps available to create a batch")

    except Exception as e:
        logger.error(f"Error in __main__: {e}")
        print(f"Error: {e}")

    finally:
        # Clean up even if there's an error
        logger.info("Cleaning up resources")
        if "data_loader" in locals():
            data_loader.close_datasets()
