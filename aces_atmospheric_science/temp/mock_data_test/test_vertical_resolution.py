import logging
from datetime import timedelta

import torch
from logging_config import setup_logging
from mock_data import create_aurora_batch

from aurora import Aurora, Batch, Metadata

# Set up centralized logging
setup_logging()


def finetune_vertical_resolution(num_levels=15, num_steps=5):
    """
    Fine-tune Aurora with different vertical resolutions.

    Args:
        num_levels (int): Number of pressure levels to use
        num_steps (int): Number of fine-tuning steps
    """
    logging.info(f"Fine-tuning with {num_levels} pressure levels")

    # Set 6-hour timestep (lead time)
    timestep = timedelta(hours=6)

    # Create model with custom configuration for fine-tuning
    model = Aurora(
        use_lora=True,  # Enable LoRA for fine-tuning
        autocast=True,  # Use AMP for memory efficiency as required
        stabilise_level_agg=True,  # Add extra layer norm to mitigate exploding gradients
        timestep=timestep,  # Set the model's timestep (lead time)
    )

    # Load pretrained weights with strict=False to allow missing LoRA keys
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt", strict=False)
    model = model.cuda()
    model.train()  # Set to training mode
    model.configure_activation_checkpointing()  # Use gradient checkpointing as required

    # Create optimizer
    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters(), "lr": 3e-4},
        ]
    )

    # Create mock data with 4 timesteps
    batch = create_aurora_batch(num_levels=num_levels, num_timesteps=4)
    batch = batch.to("cuda")  # Move batch to GPU

    # Fine-tuning loop
    for step in range(num_steps):
        try:
            logging.info(f"Step {step+1}/{num_steps}")

            # Zero gradients
            optimizer.zero_grad()

            # Calculate total loss over all lead times
            total_loss = 0.0

            # Aurora uses the first 2 timesteps as input to predict future timesteps
            # We'll use timesteps 0,1 to predict timestep 2, and timesteps 1,2 to predict timestep 3

            # First prediction: use t0,t1 -> predict t2
            input_batch1 = _create_input_batch(batch, 0, 2)  # Get first 2 timesteps
            target_time1 = batch.metadata.time[2]  # Target is 3rd timestep (index 2)

            # Forward pass to predict t2 (no need to pass lead_time parameter)
            pred1 = model.forward(input_batch1)

            # Create target from the original batch (timestep 2)
            target1 = _extract_target(batch, 2)

            # Calculate loss for first prediction
            loss1 = torch.nn.functional.mse_loss(pred1, target1)
            total_loss += loss1
            logging.info(f"Loss for prediction 1 (t0,t1->t2): {loss1.item():.4f}")

            # Second prediction: use t1,t2 -> predict t3
            input_batch2 = _create_input_batch(batch, 1, 3)  # Get timesteps 1,2
            target_time2 = batch.metadata.time[3]  # Target is 4th timestep (index 3)

            # Forward pass to predict t3 (no need to pass lead_time parameter)
            pred2 = model.forward(input_batch2)

            # Create target from the original batch (timestep 3)
            target2 = _extract_target(batch, 3)

            # Calculate loss for second prediction
            loss2 = torch.nn.functional.mse_loss(pred2, target2)
            total_loss += loss2
            logging.info(f"Loss for prediction 2 (t1,t2->t3): {loss2.item():.4f}")

            # Calculate total loss and backpropagate
            logging.info(f"Total loss: {total_loss.item():.4f}")
            total_loss.backward()

            # Check gradients
            grad_norm = torch.norm(
                torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None])
            )
            logging.info(f"Gradient norm: {grad_norm.item():.4f}")

            # Clip gradients to avoid explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update weights
            optimizer.step()

            # Log memory usage
            logging.info(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            logging.info(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

        except Exception as e:
            logging.error(f"Error during fine-tuning step {step+1}: {str(e)}")
            raise

    # Test the fine-tuned model
    try:
        model.eval()
        with torch.no_grad():
            # Test prediction using timesteps 0,1 to predict timestep 2
            input_batch_test = _create_input_batch(batch, 0, 2)
            test_pred = model.forward(input_batch_test)

            # Create target from the original batch (timestep 2)
            test_target = _extract_target(batch, 2)

            # Calculate test loss
            test_loss = torch.nn.functional.mse_loss(test_pred, test_target)
            logging.info(f"Test prediction shape: {test_pred.shape}")
            logging.info(f"Test loss: {test_loss.item():.4f}")
    except Exception as e:
        logging.error(f"Error during testing: {str(e)}")

    logging.info(f"Fine-tuning with {num_levels} pressure levels completed!")


def _create_input_batch(full_batch, start_idx, end_idx):
    """
    Create a new batch with a subset of timesteps from the full batch.

    Args:
        full_batch: The original batch with all timesteps
        start_idx: Start index for timesteps to include
        end_idx: End index (exclusive) for timesteps to include

    Returns:
        A new batch with only the selected timesteps
    """
    # Select timesteps for surface variables
    subset_surf_vars = {}
    for var_name, tensor in full_batch.surf_vars.items():
        subset_surf_vars[var_name] = tensor[:, start_idx:end_idx]

    # Select timesteps for atmospheric variables
    subset_atmos_vars = {}
    for var_name, tensor in full_batch.atmos_vars.items():
        subset_atmos_vars[var_name] = tensor[:, start_idx:end_idx]

    # Create new batch with subset of timesteps
    subset_batch = Batch(
        surf_vars=subset_surf_vars,
        static_vars=full_batch.static_vars,
        atmos_vars=subset_atmos_vars,
        metadata=Metadata(
            lat=full_batch.metadata.lat,
            lon=full_batch.metadata.lon,
            time=full_batch.metadata.time[start_idx:end_idx],
            atmos_levels=full_batch.metadata.atmos_levels,
        ),
    )

    # Return subset batch on the same device as the original
    return subset_batch.to(full_batch.surf_vars[list(full_batch.surf_vars.keys())[0]].device)


def _extract_target(batch, timestep_idx):
    """
    Extract target variables for a specific timestep from the batch.

    Args:
        batch: The batch containing all timesteps
        timestep_idx: Index of the timestep to extract

    Returns:
        Tensor containing the target variables for the specified timestep
    """
    # In a real scenario, you would extract the actual prediction variables
    # For this test, we'll just create a tensor with the same shape as the prediction
    # This is a placeholder - in a real scenario, you would use actual forecast variables

    # For simplicity, we'll extract a specific atmospheric variable (e.g., temperature)
    # and use that as our target
    target = batch.atmos_vars["t"][:, timestep_idx : timestep_idx + 1]

    # Reshape if needed to match prediction shape
    # This depends on the actual Aurora prediction format

    return target


if __name__ == "__main__":
    # Fine-tune with different numbers of levels
    for num_levels in [13, 25, 50]:
        logging.info(f"\n{'='*50}")
        logging.info(f"Fine-tuning with {num_levels} pressure levels")
        logging.info(f"{'='*50}\n")

        try:
            finetune_vertical_resolution(num_levels, num_steps=5)
        except Exception as e:
            logging.error(f"Failed with {num_levels} levels: {str(e)}")
            continue
