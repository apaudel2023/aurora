# Preparing WRF Data for Aurora Model Fine-tuning

This guide outlines how to prepare high-resolution regional WRF data for fine-tuning the Aurora weather forecasting model.

## 1. Understanding Aurora's Input Data Requirements

Aurora uses three types of input variables, each with specific dimensions:

### Static Variables
- Used for unchanging geographical features
- Standard shape in Aurora: `(1, latitude, longitude)` or single time point
- Variables required:
  - `z`: Geopotential height/orography
  - `lsm`: Land-sea mask (1=land, 0=water)
  - `slt`: Soil type

### Surface Variables
- Represent conditions at the Earth's surface
- Standard shape in Aurora: `(time_points, latitude, longitude)`
- Variables required:
  - `t2m`/`2t`: 2-meter temperature
  - `u10`/`10u`: 10-meter u-wind component
  - `v10`/`10v`: 10-meter v-wind component
  - `msl`: Mean sea level pressure

### Atmospheric Variables
- Represent 3D atmospheric conditions
- Standard shape in Aurora: `(time_points, pressure_levels, latitude, longitude)`
- Aurora's standard configuration uses 13 pressure levels
- Variables required:
  - `t`: Air temperature
  - `u`: U-wind component
  - `v`: V-wind component
  - `q`: Specific humidity
  - `z`: Geopotential height

## 2. Mapping WRF Variables to Aurora Requirements

Our WRF data analysis revealed the following variable mapping:

### Static Variables
- `z` ← WRF's `Z` available in both 2D (shape: (1, 1419, 1429)) and 3D files (shape: (1, 51, 1419, 1429))
- `lsm` ← Missing, needs derivation (see below)
- `slt` ← Missing, needs derivation (see below)

### Surface Variables
- `t2m` ← WRF's `T2` from 2D file (shape: (1, 1419, 1429))
- `u10` ← WRF's `U10` from 2D file (shape: (1, 1419, 1429))
- `v10` ← WRF's `V10` from 2D file (shape: (1, 1419, 1429))
- `msl` ← WRF's `PSFC` from 2D file (shape: (1, 1419, 1429))

### Atmospheric Variables
- `t` ← WRF's `TK` from 3D file (shape: (1, 50, 1419, 1429))
- `u` ← WRF's `U` from 3D file (shape: (1, 50, 1419, 1430)) - Note staggered grid
- `v` ← WRF's `V` from 3D file (shape: (1, 50, 1420, 1429)) - Note staggered grid
- `q` ← WRF's `QVAPOR` from 3D file (shape: (1, 50, 1419, 1429))

## 3. Variables to Compute Using WRF-Python API

Several required variables are missing or need transformation:

### Missing Static Variables
```python
import wrf as wrfpy

# 1. Create land-sea mask (lsm) from topography
# In WRF, land points have elevation > 0
def create_landmask(wrffile):
    ds = xr.open_dataset(wrffile)
    if 'HGT' in ds.variables:
        landmask = (ds.HGT > 0).astype(np.float32)
    elif 'Z' in ds.variables and len(ds.Z.shape) == 3:  # 2D Z
        landmask = (ds.Z > 0).astype(np.float32)
    else:
        raise ValueError("No elevation data found for landmask creation")
    return landmask

# 2. Extract soil type from WRF outputs or global dataset
# Option A: Check if LU_INDEX exists
def extract_soil_type(wrffile, auxiliary_file=None):
    ds = xr.open_dataset(wrffile)
    if 'LU_INDEX' in ds.variables:
        return ds.LU_INDEX  # Land use category as proxy for soil type
    elif auxiliary_file:
        # Load external soil type data and interpolate to WRF grid
        soil_ds = xr.open_dataset(auxiliary_file)
        # Interpolation code here
    else:
        # Create basic soil types based on landmask & elevation
        landmask = create_landmask(wrffile)
        # Simple classification based on elevation and landmask
```

### Variable Transformations
```python
# 1. Convert surface pressure to mean sea level pressure (if PSFC is used)
def psfc_to_msl(wrffile):
    ncfile = netCDF4.Dataset(wrffile)
    mslp = wrfpy.getvar(ncfile, "slp")  # WRF-Python calculates this correctly
    return mslp

# 2. De-stagger U and V wind components
def destagger_winds(wrffile):
    ncfile = netCDF4.Dataset(wrffile)
    u_destag = wrfpy.destagger(wrfpy.getvar(ncfile, "ua"), stagger_dim=3)  # Destagger in X
    v_destag = wrfpy.destagger(wrfpy.getvar(ncfile, "va"), stagger_dim=2)  # Destagger in Y
    return u_destag, v_destag
```

## 4. Handling Temporal Resolution Differences

Our WRF data has different temporal resolutions:
- 2D surface data: Hourly files
- 3D atmospheric data: 3-hourly files

For consistency in Aurora fine-tuning:

```python
def create_consistent_temporal_batches(data_dir_2d, data_dir_3d):
    # Get all 3D files (3-hourly)
    files_3d = sorted(glob.glob(f"{data_dir_3d}/*.nc"))
    
    # Extract timestamps from 3D files
    timestamps_3d = [extract_timestamp(f) for f in files_3d]
    
    # Find matching 2D files for these timestamps
    matching_files = []
    for ts in timestamps_3d:
        matching_2d = f"{data_dir_2d}/wrfout_d01_{ts}.nc"
        if os.path.exists(matching_2d):
            matching_files.append((matching_2d, files_3d[timestamps_3d.index(ts)]))
    
    return matching_files
```

## 5. Vertical Levels: 50 WRF Levels vs Aurora's 13 Standard Levels

Aurora's architecture can handle different pressure level configurations:

### Option 1: Use All 50 WRF Vertical Levels
- **Advantages**:
  - Preserves full vertical resolution of WRF data
  - Potentially captures more detailed vertical structures
  - Aurora's Perceiver encoder should handle varying numbers of levels
  
- **Disadvantages**:
  - Higher memory requirements
  - Potentially slower fine-tuning
  - Less direct correspondence with pretrained knowledge

### Option 2: Interpolate to 13 Standard Pressure Levels
- **Advantages**:
  - Better compatibility with pretrained Aurora model
  - Lower memory requirements
  - Faster training and inference
  
- **Disadvantages**:
  - Loss of vertical resolution
  - May miss important details in fine-scale vertical structure

```python
# Interpolate WRF's 50 model levels to standard pressure levels
def interpolate_to_standard_pressure_levels(wrffile):
    ncfile = netCDF4.Dataset(wrffile)
    
    # Standard pressure levels used by Aurora (hPa)
    std_press_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
    
    # Get pressure field on model levels
    p = wrfpy.getvar(ncfile, "pressure")
    
    # Get required variables
    t = wrfpy.getvar(ncfile, "temp")
    u, v = destagger_winds(wrffile)
    q = wrfpy.getvar(ncfile, "QVAPOR")
    z = wrfpy.getvar(ncfile, "z")
    
    # Interpolate to pressure levels
    t_plevs = wrfpy.vinterp(ncfile, t, "pressure", std_press_levels)
    u_plevs = wrfpy.vinterp(ncfile, u, "pressure", std_press_levels)
    v_plevs = wrfpy.vinterp(ncfile, v, "pressure", std_press_levels)
    q_plevs = wrfpy.vinterp(ncfile, q, "pressure", std_press_levels)
    z_plevs = wrfpy.vinterp(ncfile, z, "pressure", std_press_levels)
    
    return t_plevs, u_plevs, v_plevs, q_plevs, z_plevs
```

For our initial testing, we recommend trying Option 1 (using all 50 levels) first, since Aurora's architecture was designed to handle flexible pressure level configurations.

## 6. Initial Validation Approach

Before full fine-tuning, we'll conduct an initial validation to verify data compatibility:

### Step 1: Prepare Test Data Batch
```python
def prepare_validation_batch(consecutive_files, use_50_levels=True):
    # Load two consecutive timesteps
    ds1 = xr.open_dataset(consecutive_files[0])
    ds2 = xr.open_dataset(consecutive_files[1])
    
    # Process variables and create Aurora batch
    batch = Batch(
        surf_vars={
            "2t": torch.from_numpy(np.stack([ds1.T2.values, ds2.T2.values])[None]),
            "10u": torch.from_numpy(np.stack([ds1.U10.values, ds2.U10.values])[None]),
            "10v": torch.from_numpy(np.stack([ds1.V10.values, ds2.V10.values])[None]),
            "msl": torch.from_numpy(np.stack([ds1.PSFC.values, ds2.PSFC.values])[None]),
        },
        static_vars={
            "z": torch.from_numpy(ds1.Z.values[0]),
            "lsm": torch.from_numpy(create_landmask(consecutive_files[0]).values),
            "slt": torch.from_numpy(extract_soil_type(consecutive_files[0]).values),
        },
        atmos_vars={
            "t": torch.from_numpy(np.stack([ds1.TK.values, ds2.TK.values])[None]),
            "u": torch.from_numpy(np.stack([
                destagger_winds(consecutive_files[0])[0], 
                destagger_winds(consecutive_files[1])[0]
            ])[None]),
            "v": torch.from_numpy(np.stack([
                destagger_winds(consecutive_files[0])[1], 
                destagger_winds(consecutive_files[1])[1]
            ])[None]),
            "q": torch.from_numpy(np.stack([ds1.QVAPOR.values, ds2.QVAPOR.values])[None]),
            "z": torch.from_numpy(np.stack([ds1.Z.values, ds2.Z.values])[None]),
        },
        metadata=Metadata(
            lat=torch.from_numpy(ds1.XLAT.values),
            lon=torch.from_numpy(ds1.XLONG.values),
            time=(pd.to_datetime(ds2.Time.values[0]).to_pydatetime(),),
            atmos_levels=tuple(range(1, 51)) if use_50_levels else 
                        tuple([1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]),
        ),
    )
    return batch
```

### Step 2: Run Inference with Pretrained Aurora
```python
from aurora import Aurora, rollout

# Load pretrained model
model = Aurora.from_pretrained("microsoft/aurora")

# Prepare test batch from consecutive timesteps
batch = prepare_validation_batch(consecutive_files)

# Run inference
with torch.inference_mode():
    predictions = [pred for pred in rollout(model, batch, steps=2)]

# Compare with actual WRF outputs
# Load the actual next-step WRF output
actual_next = xr.open_dataset(next_file)

# Compare key fields
compare_fields(predictions[0], actual_next)
```

### Step 3: Evaluate Regional Performance
- Compare Aurora predictions against actual WRF output for the next timestep
- Analyze spatial patterns and error distribution within the region
- Pay special attention to boundary areas to assess boundary condition handling
- Create visualizations comparing predicted vs. actual fields

This initial validation will help identify any compatibility issues and provide insights into how well Aurora can handle our high-resolution regional data before proceeding with full fine-tuning.

## 7. Memory Considerations for High-Resolution Data

Our WRF data has dimensions of 1419×1429 grid points with 50 vertical levels, which is significantly higher resolution than Aurora's standard training data. To manage memory:

1. **Patch Size Optimization**:
   - Aurora processes data in patches (default 16×16)
   - For our high-resolution grid, consider larger patches (e.g., 32×32)
   - This reduces the number of patches while maintaining spatial information

2. **Gradient Checkpointing**:
   - Enable gradient checkpointing to trade computation for memory
   - Essential for fine-tuning with high-resolution data

3. **Mixed Precision Training**:
   - Use bfloat16 precision to reduce memory footprint
   - Aurora supports bf16 mixed precision training

4. **Regional Subsetting**:
   - Consider using a smaller representative subset of the full domain for initial fine-tuning experiments
   - Once optimized, scale up to the full domain

## Conclusion

Our WRF data provides an excellent opportunity to fine-tune Aurora for high-resolution regional forecasting. With proper preprocessing to handle the missing variables, staggered grids, and vertical level differences, we can create compatible input data for Aurora.

The recommended approach is to:
1. First validate Aurora's performance on our data without fine-tuning
2. Start with 1-2 month subset of data for preliminary fine-tuning experiments
3. Optimize hyperparameters and preprocessing choices
4. Scale up to full fine-tuning with larger dataset once the pipeline is validated

Aurora's flexible Perceiver architecture should theoretically handle our high-resolution regional data well, particularly with the explicit patch area encoding that enables multi-resolution inputs. 