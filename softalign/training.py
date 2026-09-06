import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim


class EventFrameDataset:
    def __init__(
        self,
        events,
        frame_points,
        batch_size=1024,
        device='cuda',
        generator=None,
    ):
        """
        Prepare dataset for training

        Args:
            events: Event data with shape (n, 4) [x, y, t, polarity]
            frame_points: Frame points with shape (m, 4) [x, y, t, intensity]
            batch_size: Batch size for training
            device: Device to use (cuda or cpu)
        """
        events = np.asarray(events)
        frame_points = np.asarray(frame_points)
        if events.ndim != 2 or events.shape[1] != 4 or not len(events):
            raise ValueError("events must be a non-empty N x 4 array")
        if frame_points.ndim != 2 or frame_points.shape[1] != 4 or not len(frame_points):
            raise ValueError("frame_points must be a non-empty N x 4 array")
        if not np.isfinite(events).all() or not np.isfinite(frame_points).all():
            raise ValueError("events and frame_points must contain only finite values")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")

        # Keep the collection on CPU and move only sampled batches to the target
        # device. Large event arrays otherwise consume GPU memory regardless of
        # the requested batch size.
        self.device = torch.device(device)
        self.generator = generator
        self.events = torch.tensor(events[:, :3], dtype=torch.float32)
        self.polarities = torch.tensor(events[:, 3], dtype=torch.float32).view(-1, 1)
        self.frame_points = torch.tensor(frame_points[:, :3], dtype=torch.float32)
        self.intensities = torch.tensor(frame_points[:, 3], dtype=torch.float32).view(-1, 1)

        self.event_batch_size = min(batch_size, len(self.events))
        self.frame_batch_size = min(batch_size, len(self.frame_points))

        self.num_events = len(self.events)
        self.num_frames = len(self.frame_points)

    def get_batch(self):
        """Get a random batch of events and frames"""
        # Random indices for events
        if self.event_batch_size == self.num_events:
            event_batch = self.events
            polarity_batch = self.polarities
        else:
            event_idx = torch.randint(
                0,
                self.num_events,
                (self.event_batch_size,),
                generator=self.generator,
            )
            event_batch = self.events[event_idx]
            polarity_batch = self.polarities[event_idx]

        # Random indices for frames
        if self.frame_batch_size == self.num_frames:
            frame_batch = self.frame_points
            intensity_batch = self.intensities
        else:
            frame_idx = torch.randint(
                0,
                self.num_frames,
                (self.frame_batch_size,),
                generator=self.generator,
            )
            frame_batch = self.frame_points[frame_idx]
            intensity_batch = self.intensities[frame_idx]

        return (
            event_batch.to(self.device),
            polarity_batch.to(self.device),
            frame_batch.to(self.device),
            intensity_batch.to(self.device),
        )

def train_model(
    model,
    dataset,
    num_epochs=1000,
    lr=1e-3,
    lambda_reg=0.001,
    checkpoint_dir='checkpoints',
    *,
    log_interval=50,
    checkpoint_interval=200,
    make_plots=True,
):
    """
    Train the model to align events and frames

    Args:
        model: EventFrameAlignmentModel instance
        dataset: EventFrameDataset instance
        num_epochs: Number of training epochs
        lr: Learning rate
        lambda_reg: Regularization strength for transformation parameters
        checkpoint_dir: Directory to save model checkpoints
    """
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive")
    if lr <= 0:
        raise ValueError("lr must be positive")
    if lambda_reg < 0:
        raise ValueError("lambda_reg must not be negative")
    if log_interval <= 0:
        raise ValueError("log_interval must be positive")
    if checkpoint_dir is not None and checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive when checkpoints are enabled")
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=50, factor=0.5
    )

    # For logging
    train_losses = []
    event_losses = []
    frame_losses = []
    parameter_history = {
        'scale': [], 'shift_x': [], 'shift_y': [], 'shift_t': [], 'threshold': [], 'dt': []
    }

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Get random batch
        event_coords, event_polarities, frame_coords, frame_intensities = dataset.get_batch()

        # Forward pass for events
        event_response = model.forward_event(event_coords)

        # Forward pass for frames
        frame_response = model.forward_frame(frame_coords)

        # Loss computation
        # Event loss: We want event_response to match the event polarities
        loss_event = F.mse_loss(event_response, event_polarities)

        # Frame loss: We want frame_response to match the frame intensities
        loss_frame = F.mse_loss(frame_response, frame_intensities)

        # Regularization on transformation parameters
        reg_loss = lambda_reg * (
            torch.abs(1.0 - model.scale) +
            torch.abs(model.shift_x) +
            torch.abs(model.shift_y) +
            torch.abs(model.shift_t)
        )

        # Total loss
        total_loss = loss_event + loss_frame + reg_loss

        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()

        # Logging
        train_losses.append(total_loss.item())
        event_losses.append(loss_event.item())
        frame_losses.append(loss_frame.item())

        # Track parameters
        parameter_history['scale'].append(model.scale.item())
        parameter_history['shift_x'].append(model.shift_x.item())
        parameter_history['shift_y'].append(model.shift_y.item())
        parameter_history['shift_t'].append(model.shift_t.item())
        parameter_history['threshold'].append(model.threshold.item())
        parameter_history['dt'].append(model.dt.item())

        # Print progress
        if epoch % log_interval == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch}/{num_epochs}: "
                  f"Loss = {total_loss.item():.6f}, "
                  f"Event Loss = {loss_event.item():.6f}, "
                  f"Frame Loss = {loss_frame.item():.6f}")
            print(f"  Scale = {model.scale.item():.4f}, "
                  f"shift_x = {model.shift_x.item():.4f}, "
                  f"shift_y = {model.shift_y.item():.4f}, "
                  f"shift_t = {model.shift_t.item():.4f}")
            print(f"  Threshold = {model.threshold.item():.4f}, "
                  f"dt = {model.dt.item():.4f}")

        # Learning rate scheduling
        scheduler.step(total_loss.item())

        # Save checkpoint
        if checkpoint_dir is not None and (
            (epoch + 1) % checkpoint_interval == 0 or epoch == num_epochs - 1
        ):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss.item(),
                'parameters': {
                    'scale': model.scale.item(),
                    'shift_x': model.shift_x.item(),
                    'shift_y': model.shift_y.item(),
                    'shift_t': model.shift_t.item(),
                    'threshold': model.threshold.item(),
                    'dt': model.dt.item()
                },
                'model_config': {
                    'hidden_dim': model.hidden_dim,
                    'num_layers': model.num_layers,
                },
            }, os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pt'))

    # Plot training curves
    if make_plots:
        if checkpoint_dir is None:
            raise ValueError("plotting requires a checkpoint/output directory")
        plot_training_curves(
            train_losses,
            event_losses,
            frame_losses,
            parameter_history,
            checkpoint_dir,
        )

    return model, train_losses, parameter_history

def plot_training_curves(train_losses, event_losses, frame_losses, parameter_history, save_dir):
    """
    Plot and save training curves and parameter history

    Args:
        train_losses: List of total loss values
        event_losses: List of event loss values
        frame_losses: List of frame loss values
        parameter_history: Dictionary of parameter histories
        save_dir: Directory to save plots
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "plotting requires the viz dependencies: pip install -e '.[viz]'"
        ) from exc

    os.makedirs(save_dir, exist_ok=True)

    # Plot losses
    plt.figure(figsize=(12, 6))
    epochs = range(len(train_losses))
    plt.semilogy(epochs, train_losses, label='Total Loss')
    plt.semilogy(epochs, event_losses, label='Event Loss')
    plt.semilogy(epochs, frame_losses, label='Frame Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'))

    # Plot transformation parameters
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, parameter_history['scale'])
    plt.xlabel('Epoch')
    plt.ylabel('Scale')
    plt.title('Scale Parameter')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(epochs, parameter_history['shift_x'], label='shift_x')
    plt.plot(epochs, parameter_history['shift_y'], label='shift_y')
    plt.xlabel('Epoch')
    plt.ylabel('Spatial Shift')
    plt.title('Spatial Shift Parameters')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(epochs, parameter_history['shift_t'])
    plt.xlabel('Epoch')
    plt.ylabel('Time Shift')
    plt.title('Time Shift Parameter')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(epochs, parameter_history['threshold'], label='threshold')
    plt.plot(epochs, parameter_history['dt'], label='dt')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Threshold and dt Parameters')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'parameter_history.png'))

    # Close figures to free memory
    plt.close('all')
