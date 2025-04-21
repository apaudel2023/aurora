from aurora import Batch, Metadata


def crop_batch_to_multiple_of_4(batch):
    """Crop an Aurora batch to make dimensions multiples of 4.

    Args:
        batch: Aurora Batch object
    Returns:
        Cropped Aurora Batch object
    """
    h, w = batch.spatial_shape

    # Calculate new dimensions (largest multiple of 4 that fits)
    new_h = (h // 4) * 4  # 1419 -> 1416
    new_w = (w // 4) * 4  # 1429 -> 1428

    print(f"Cropping from {h}x{w} to {new_h}x{new_w}")
    print(f"Removing {h - new_h} points from height and {w - new_w} points from width")

    # Create new batch with cropped arrays
    cropped_batch = Batch(
        surf_vars={k: v[..., :new_h, :new_w] for k, v in batch.surf_vars.items()},
        static_vars={k: v[..., :new_h, :new_w] for k, v in batch.static_vars.items()},
        atmos_vars={k: v[..., :new_h, :new_w] for k, v in batch.atmos_vars.items()},
        metadata=Metadata(
            lat=batch.metadata.lat[:new_h],
            lon=batch.metadata.lon[:new_w],
            atmos_levels=batch.metadata.atmos_levels,
            time=batch.metadata.time,
            rollout_step=batch.metadata.rollout_step,
        ),
    )

    return cropped_batch
