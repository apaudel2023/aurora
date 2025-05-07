# Aurora Vertical Resolution Test

This directory contains scripts to test fine-tuning the Aurora model with different vertical resolutions.

## Purpose

The main goal is to investigate whether the Aurora model can be fine-tuned to handle different numbers of pressure levels than its default 13 levels. This is particularly relevant for cases where we have data with more vertical levels (e.g., 50 levels).

## Files

- `mock_data.py`: Generates mock Aurora batches with customizable number of pressure levels
- `test_vertical_resolution.py`: Tests the model with different numbers of pressure levels
- `vertical_resolution_test.log`: Output log file (generated during execution)

## How to Run

```bash
python test_vertical_resolution.py
```

The script will:
1. Test the model with 13, 25, and 50 pressure levels
2. Log the results to both console and `vertical_resolution_test.log`
3. Report any errors or successful forward/backward passes

## What to Look For

1. **Forward Pass**: Can the model handle the input with different numbers of levels?
2. **Backward Pass**: Are gradients computed successfully?
3. **Gradient Norm**: Are the gradients reasonable or exploding?

## Notes

- The test uses random data to avoid any data-specific issues
- Gradient checkpointing is enabled to manage memory
- Extra layer normalization is added to help with potential gradient issues
- The test is designed to fail gracefully and continue with the next number of levels if one fails
