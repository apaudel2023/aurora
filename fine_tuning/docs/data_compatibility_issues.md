# Aurora Model Data Compatibility Issues

This document outlines key compatibility issues and constraints when using custom data with the Aurora model.

## 1. Pressure Level Requirements

### Model Expectations
- **Standard Levels**: Aurora expects exactly 13 specific pressure levels
  ```python
  standard_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]  # in hPa
  ```
- **Units**: Pressure levels must be in hectopascals (hPa)
- **Normalization**: Model has pre-defined normalization parameters for each variable at each standard level

### Common Issues
- Using different number of pressure levels breaks the model architecture
- Data in Pascal (Pa) needs conversion to hPa
- Non-standard pressure levels don't have corresponding normalization parameters
- Fourier encoding in the model enforces range check: [0.01, 100000.0] hPa

### Impact
- Can't directly use data with different pressure levels
- Model's encoder expects exact number of levels
- Normalization fails with non-standard levels

## 2. Spatial Dimension Constraints

### Patch Size Requirements
- Model uses patch-based processing
- Spatial dimensions (height, width) must be multiples of patch_size (4)
- Example: For 1416x1428 dimensions:
  ```python
  # Check if dimensions are compatible
  if height % patch_size != 0 or width % patch_size != 0:
      # Need to crop or pad
  ```

## 3. Model Architecture Constraints

### Encoder Design
- The encoder expects specific pressure level structure:
  ```python
  # From encoder.py
  surf_vars = tuple(batch.surf_vars.keys())
  atmos_vars = tuple(batch.atmos_vars.keys())
  atmos_levels = batch.metadata.atmos_levels
  ```
- Separate processing paths for surface and atmospheric variables
- Level-specific normalization and encoding

### Key Constraints
- Model's normalization parameters are hardcoded for 13 specific pressure levels
- The encoder's level aggregation expects fixed number of input levels
- Surface and atmospheric variables are processed differently
- Pre-trained weights assume specific level structure and normalization

## Potential Solutions

1. **Data Adaptation**
   - Interpolate data to standard 13 pressure levels
   - Crop spatial dimensions to multiples of patch_size
   - Process in smaller spatial chunks

2. **Model Adaptation** (More Complex)
   - Modify model architecture for different number of levels
   - Retrain with new architecture
   - Adjust normalization parameters
