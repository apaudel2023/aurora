# Aurora Model Compatibility Report

## 1. Introduction

This document summarizes our experiences adapting the Aurora weather prediction model to work with our regional WRF data. It covers the model's expectations, our testing approach, encountered issues, and potential solutions.

## 2. Aurora Model Input Requirements

The Aurora model expects specific types and structures of input data:

### Standard Input Variables

**Static Variables** (single time point):
- `z`: Geopotential height/orography - shape: `(height, width)`
- `lsm`: Land-sea mask (1=land, 0=water) - shape: `(height, width)`
- `slt`: Soil type - shape: `(height, width)`

**Surface Variables** (time series):
- `2t`: 2-meter temperature - shape: `(batch, time, height, width)`
- `10u`: 10-meter u-wind component - shape: `(batch, time, height, width)`
- `10v`: 10-meter v-wind component - shape: `(batch, time, height, width)`
- `msl`: Mean sea level pressure - shape: `(batch, time, height, width)`

**Atmospheric Variables** (3D atmospheric structure):
- `z`: Geopotential height - shape: `(batch, time, pressure_levels, height, width)`
- `u`: U-wind component - shape: `(batch, time, pressure_levels, height, width)`
- `v`: V-wind component - shape: `(batch, time, pressure_levels, height, width)`
- `t`: Air temperature - shape: `(batch, time, pressure_levels, height, width)`
- `q`: Specific humidity - shape: `(batch, time, pressure_levels, height, width)`

**Metadata** (required for proper model operation):
- `lat`: Latitude values - shape: `(height,)` or `(height, width)` - must be strictly decreasing
- `lon`: Longitude values - shape: `(width,)` or `(height, width)` - must be strictly increasing
- `time`: Tuple of datetime objects for each time step
- `atmos_levels`: Tuple of pressure levels in hPa

### Critical Constraints

1. **Pressure Levels**: Aurora expects **exactly 13 standard pressure levels**:
   ```python
   standard_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]  # in hPa
   ```

2. **Spatial Dimensions**: Height and width must be multiples of `patch_size` (4)

3. **Normalization**: Model has hardcoded normalization parameters for each variable at each standard level

4. **Architecture Limitations**:
   - Normalization statistics are hardcoded for exactly 13 pressure levels
   - Grid width % patch_size must == 0 (mandatory)
   - Grid height % patch_size can be == 1 (model handles cropping), otherwise needs preprocessing
   - Model requires longitude values in specific range and format

### Normalization Parameters

The Aurora model uses predefined normalization parameters for each variable. For example, from `normalisation.py`:

```python
# Excerpt from Aurora's normalization parameters
locations: dict[str, float] = {
    "z": -1.386496e03,
    "lsm": 0.000000e00,
    "slt": 0.000000e00,
    "2t": 2.785140e02,
    "10u": -5.135059e-02,
    "10v": 1.891580e-01,
    "msl": 1.009578e05,
    "z_50": 1.993730e05,
    "z_100": 1.576421e05,
    "z_150": 1.331414e05,
    # ... (parameters for each standard level)
}

scales: dict[str, float] = {
    "z": 5.884467e04,
    "lsm": 1.000000e00,
    "slt": 7.000000e00,
    "2t": 2.122036e01,
    "10u": 5.547512e00,
    "10v": 4.765339e00,
    "msl": 1.332246e03,
    "z_50": 5.875553e03,
    "z_100": 5.510640e03,
    "z_150": 5.823912e03,
    # ... (scales for each standard level)
}
```

These dictionaries include entries for every variable at every standard pressure level, demonstrating how tightly coupled the model architecture is to the expected levels.

## 3. Testing with Mock Data

We tested various configurations using synthetic data to understand Aurora's flexibility:

### Mock Data Example

```python
# Mock data generation function
def create_aurora_batch(batch_size=1, time_step=4, grid_height=16, grid_width=32, pressure_levels=13):
    # Create random data with specified dimensions
    surf_vars = {
        "2t": torch.rand(batch_size, time_step, grid_height, grid_width),
        "10u": torch.rand(batch_size, time_step, grid_height, grid_width),
        "10v": torch.rand(batch_size, time_step, grid_height, grid_width)
    }

    static_vars = {
        "z": torch.rand(grid_height, grid_width)  # No batch dimension for static vars
    }

    # Generate specific pressure levels (values matter!)
    if pressure_levels == 13:
        p_levels = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
    else:
        # Generate evenly distributed pressure levels (causes errors if not standard)
        step = (1000 - 100) // (pressure_levels - 1)
        p_levels = tuple(range(100, 1001, step))

    atmos_vars = {
        "z": torch.rand(batch_size, time_step, pressure_levels, grid_height, grid_width),
        "u": torch.rand(batch_size, time_step, pressure_levels, grid_height, grid_width),
        "v": torch.rand(batch_size, time_step, pressure_levels, grid_height, grid_width),
        "t": torch.rand(batch_size, time_step, pressure_levels, grid_height, grid_width),
        "q": torch.rand(batch_size, time_step, pressure_levels, grid_height, grid_width)
    }

    # Create batch with metadata
    batch = Batch(
        surf_vars=surf_vars,
        static_vars=static_vars,
        atmos_vars=atmos_vars,
        metadata=Metadata(...)
    )

    return batch
```

### Mock Data Testing Results

| Test Case | Result | Notes |
|-----------|--------|-------|
| Standard 13 pressure levels | ✅ Success | Using exactly the standard pressure levels |
| Fewer pressure levels (e.g., 7) | ✅ Success | Works if all levels match subset of standard levels |
| 25 pressure levels | ❌ Failed | Error: Cannot find normalization for non-standard level |
| Missing static variables | ✅ Success | Limited subset of static variables works |
| Custom grid dimensions | ✅ Success | Works when dimensions are multiples of patch_size |
| Regional lat/lon range | ✅ Success | Model adapts to regional coordinate ranges |

This testing revealed the model is tolerant of some variability (grid size, subset of static variables) but strictly requires pressure levels to be a subset of the standard 13 levels.

## 4. Working with WRF Regional Data

Our WRF data has different structure and resolution than Aurora's training data:

### WRF Data Characteristics

- **Temporal Structure**:
  - 2D surface data: Hourly files
  - 3D atmospheric data: 3-hourly files

- **Spatial Resolution**:
  - Grid dimensions: 1419×1429
  - Not divisible by patch_size (4)

- **Vertical Structure**:
  - 50 pressure levels vs. Aurora's 13 standard levels
  - Levels in Pa rather than hPa

### Detailed Variable Mapping

1. **STATIC VARIABLES**:
   - ✓ `z` ← `Z` (2D file (shape: (1, 1419, 1429)) and 3D file (shape: (1, 51, 1419, 1429)))
   - ✗ `lsm` (missing, needs derivation)
   - ✗ `slt` (missing, needs derivation)

2. **SURFACE VARIABLES**:
   - ✓ `t2m` ← `T2` (2D file (shape: (1, 1419, 1429)))
   - ✓ `u10` ← `U10` (2D file (shape: (1, 1419, 1429)))
   - ✓ `v10` ← `V10` (2D file (shape: (1, 1419, 1429)))
   - ✓ `msl` ← `PSFC` (2D file (shape: (1, 1419, 1429)))

3. **ATMOSPHERIC VARIABLES**:
   - ✓ `t` ← `TK` (3D file (shape: (1, 50, 1419, 1429)))
   - ✓ `u` ← `U` (3D file (shape: (1, 50, 1419, 1430))) - Note staggered grid
   - ✓ `v` ← `V` (3D file (shape: (1, 50, 1420, 1429))) - Note staggered grid
   - ✓ `q` ← `QVAPOR` (3D file (shape: (1, 50, 1419, 1429)))

**Additional Processing Required**:
- De-staggering for U and V variables (different dimensions)
- Derivation of missing static variables (lsm, slt)
- Cropping grid to multiples of patch_size

## 5. Data Preparation Steps Implemented

We successfully created Aurora-compatible batch objects by leveraging utilities from our `data_loader` directory in a sequential workflow implemented in `test.ipynb`:

1. **WRF Data Loading**:
   - Used `load_and_combine_wrf_files` from `load_wrf.py` to load WRF output files
   - Loaded two 2D files and two 3D files from the same timestamp (2015-12-01)
   ```python
   DATA_DIR = "/home/user/Documents/aurora/data_wrf"
   meta, surf, static, atom = load_and_combine_wrf_files(
       os.path.join(DATA_DIR, "2d", "wrf2d_d01_2015-12-01_00:00:00.nc"),
       os.path.join(DATA_DIR, "2d", "wrf2d_d01_2015-12-01_03:00:00.nc"),
       os.path.join(DATA_DIR, "3d", "wrf3d_d01_2015-12-01_00:00:00.nc"),
       os.path.join(DATA_DIR, "3d", "wrf3d_d01_2015-12-01_03:00:00.nc")
   )
   ```

2. **Coordinate Processing**:
   - Used `process_lat_lon` from `coord_util.py` to ensure coordinates met Aurora's requirements
   - Verified latitude was strictly decreasing (1419,) and longitude strictly increasing (1429,)
   ```python
   # Diagnostics confirmed:
   # lat1d.shape: (1419,), decreasing? True
   # lon1d.shape: (1429,), increasing? True
   ```

3. **Pressure Level Processing**:
   - Used `process_pressure_levels` from `pressure_util.py` to extract and process pressure levels
   - Processed 50 pressure levels from WRF data (values in Pa: min=5141, max=105286)
   ```python
   meta["pressure_levels"] = process_pressure_levels(meta["pressure_levels"], target=50)
   # Confirmed: length=50, min=5141, max=105286
   ```

4. **Atmospheric Variable Standardization**:
   - Used `standardize_atmospheric_vars` from `atmos_util.py` to adjust dimensions
   - Standardized all atmospheric variables to shape (2, 50, 1419, 1429)
   ```python
   H, W = surf["t2"].shape[1:]
   atom = standardize_atmospheric_vars(atom, target_levels=50, target_y=H, target_x=W)
   ```

5. **Batch Creation**:
   - Used `make_aurora_batch` from `batch_util.py` to create an Aurora batch object
   - Successfully assembled all processed components into a standardized batch
   - **Intentionally omitted** `lsm` and `slt` static variables since our mock data testing confirmed the model works without them
   ```python
   batch = make_aurora_batch(meta, surf, static, atom)
   ```

6. **Dimension Cropping**:
   - Used `crop_batch_to_multiple_of_4` from `crop_batch_mul4.py` to ensure dimensions compatible with patch_size=4
   - Cropped batch from 1419×1429 to 1416×1428
   - Removed 3 points from height and 1 point from width
   ```python
   cropped_batch = crop_batch_to_multiple_of_4(batch)
   # Output: "Cropping from 1419x1429 to 1416x1428"
   ```

7. **Validation Testing**:
   - Attempted inference with our prepared data before proceeding to fine-tuning
   - Verified shapes and structures at each step of processing with diagnostic print statements
   - Encountered compatibility issues with the pressure levels despite successful batch creation

### Final Batch Structure Example for wrf data (our)

The final batch object created had the following structure:

```python
# Structure of the cropped_batch created in our workflow
cropped_batch = Batch(
    # Surface variables - 2 time steps, 1416 height, 1428 width
    surf_vars={
        "2t": tensor of shape (1, 2, 1416, 1428),  # 2-meter temperature
        "10u": tensor of shape (1, 2, 1416, 1428),  # 10-meter u-wind
        "10v": tensor of shape (1, 2, 1416, 1428),  # 10-meter v-wind
        "msl": tensor of shape (1, 2, 1416, 1428),  # surface pressure
    },

    # Static variables - no batch dimension
    static_vars={
        "z": tensor of shape (1416, 1428),  # Geopotential height
        # 'lsm' and 'slt' intentionally omitted
    },

    # Atmospheric variables - 2 time steps, 50 pressure levels
    atmos_vars={
        "z": tensor of shape (1, 2, 50, 1416, 1428),  # Geopotential height
        "t": tensor of shape (1, 2, 50, 1416, 1428),  # Temperature
        "u": tensor of shape (1, 2, 50, 1416, 1428),  # U-wind
        "v": tensor of shape (1, 2, 50, 1416, 1428),  # V-wind
        "q": tensor of shape (1, 2, 50, 1416, 1428),  # Specific humidity
    },

    # Metadata
    metadata=Metadata(
        lat=tensor of shape (1416,),  # Decreasing latitudes
        lon=tensor of shape (1428,),  # Increasing longitudes
        time=(datetime(2015, 12, 1, 0, 0, 0), datetime(2015, 12, 1, 3, 0, 0)),
        atmos_levels=tuple of 50 pressure levels in Pa (from 5141 to 105286)
    )
)
```

## 6. Critical Issue: Pressure Level Incompatibility

Despite successful batch creation, we encountered a fundamental limitation when attempting inference:

```
Error during inference: Cannot find normalization for non-standard pressure level
```

These errors indicate:

1. The model's normalization is hardcoded for exactly 13 standard pressure levels
2. The model expects specific level values and fails when encountering non-standard levels
3. Simply providing a different number of levels or different level values causes failure
4. This appears to be a fundamental architectural constraint, not just a data formatting issue

### Specific Architecture Constraints

The model architecture has several hard constraints:

1. **Normalization Parameters**: Each variable at each pressure level has pre-defined normalization statistics
   ```python
   # The normalization function in Aurora's code:
   def normalise_atmos_var(
       x: torch.Tensor,
       name: str,
       atmos_levels: tuple[int | float, ...],
       unnormalise: bool = False
   ) -> torch.Tensor:
       """Normalise an atmospheric variable."""
       level_locations: list[int | float] = []
       level_scales: list[int | float] = []
       for level in atmos_levels:
           level_locations.append(locations[f"{name}_{level}"])
           level_scales.append(scales[f"{name}_{level}"])
   ```

2. **Grid Dimensions**:
   - Width must be divisible by patch_size (4)
   - Height can be handled by model if remainder is 1, otherwise needs preprocessing

3. **Vertical Structure**:
   - Hardcoded for specific pressure level values
   - Even standard pressure levels in non-standard order will fail

## 7. Potential Solutions

Based on our findings, we've identified two potential approaches to deal with non standard pressure levels:

### Approach 1: Data Adaptation

- **Vertical Interpolation**: Convert our 50 WRF levels to the 13 standard levels
- **Implementation**: Create interpolation utilities to transform pressure level data
- **Advantages**:
  - Works with existing model architecture (Hopefully)
  - Preserves pretrained knowledge
  - Simpler implementation

### Approach 2: Model Adaptation (Complex)

- **Architecture Modification**: Modify model to handle arbitrary pressure levels
- **Custom Normalization**: Create normalization parameters for all 50 levels in our data
  ```python
  # Would require extending the normalization dictionaries with our pressure levels
  locations.update({
      "z_118": computed_mean_for_z_at_118hPa,
      "t_118": computed_mean_for_t_at_118hPa,
      # ... for all variables at each of our 50 pressure levels
  })

  scales.update({
      "z_118": computed_std_for_z_at_118hPa,
      "t_118": computed_std_for_t_at_118hPa,
      # ... for all variables at each of our 50 pressure levels
  })
  ```
- **Compatibility Risk**: Even with custom normalization parameters, there's no guarantee the pretrained model will work with different pressure levels
