import argparse
import json
import os

import numpy as np
import torch

from softalign.implicit_model import EventFrameAlignmentModel
from softalign.training import EventFrameDataset, train_model


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train event-frame alignment model')
    parser.add_argument('--filepath', type=str, default='',
                        help='Path to a rights-cleared AEDAT4 recording')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='Duration in seconds to read from file')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory to save/load processed data')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--num_epochs', type=int, default=1000,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for training')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension of the MLP')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of layers in the MLP')
    parser.add_argument('--lambda_reg', type=float, default=0.001,
                        help='Regularization strength')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (cuda or cpu)')
    parser.add_argument('--reprocess', action='store_true',
                        help='Force reprocessing of data even if it exists')
    parser.add_argument('--synthetic-misalignment', action='store_true',
                        help='Apply the documented synthetic transform during AEDAT preprocessing')
    parser.add_argument('--seed', type=int, default=7,
                        help='Random seed for sampling and training')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip loss and parameter plots (core dependencies only)')
    args = parser.parse_args()

    if str(args.device).startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA was requested but is not available; use --device cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create output directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Check if processed data exists
    events_path = os.path.join(args.data_dir, 'events.npy')
    frame_points_path = os.path.join(args.data_dir, 'frame_points.npy')

    # Process data if needed
    if args.reprocess or not os.path.exists(events_path) or not os.path.exists(frame_points_path):
        if not args.filepath:
            parser.error('--filepath is required when processed arrays are unavailable or --reprocess is used')
        from softalign.data_processing import prepare_data

        print(f"Processing data from {args.filepath}...")
        data = prepare_data(
            args.filepath,
            duration_seconds=args.duration,
            output_dir=args.data_dir,
            synthetic_misalignment=args.synthetic_misalignment,
            seed=args.seed,
        )
        events = data['events']
        frame_points = data['frame_points']
    else:
        print(f"Loading processed data from {args.data_dir}...")
        metadata_path = os.path.join(args.data_dir, 'preprocessing.json')
        if not os.path.exists(metadata_path):
            parser.error(
                f'{metadata_path} is missing; regenerate with --reprocess so timestamp units are explicit'
            )
        with open(metadata_path, encoding='utf-8') as file:
            metadata = json.load(file)
        if metadata.get('schema') != 'softalign-processed/v1' or metadata.get('timestamp_unit') != 'seconds':
            parser.error('processed data metadata is incompatible; regenerate with --reprocess')
        events = np.load(events_path)
        frame_points = np.load(frame_points_path)

    print(f"Events shape: {events.shape}")
    print(f"Frame points shape: {frame_points.shape}")

    # Create dataset
    dataset = EventFrameDataset(events, frame_points, batch_size=args.batch_size, device=args.device)

    # Create model
    model = EventFrameAlignmentModel(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )
    model.to(args.device)
    print(f"Created model with initial parameters:")
    print(f"  Scale = {model.scale.item():.4f}, "
          f"shift_x = {model.shift_x.item():.4f}, "
          f"shift_y = {model.shift_y.item():.4f}, "
          f"shift_t = {model.shift_t.item():.4f}")
    print(f"  Threshold = {model.threshold.item():.4f}, "
          f"dt = {model.dt.item():.4f}")

    # Train model
    print(f"Training model for {args.num_epochs} epochs...")
    model, train_losses, parameter_history = train_model(
        model, dataset, num_epochs=args.num_epochs, lr=args.lr,
        lambda_reg=args.lambda_reg, checkpoint_dir=args.checkpoint_dir,
        make_plots=not args.no_plots,
    )

    # Print final parameters
    print("\nTraining complete!")
    print(f"Final parameters:")
    print(f"  Scale = {model.scale.item():.4f}, "
          f"shift_x = {model.shift_x.item():.4f}, "
          f"shift_y = {model.shift_y.item():.4f}, "
          f"shift_t = {model.shift_t.item():.4f}")
    print(f"  Threshold = {model.threshold.item():.4f}, "
          f"dt = {model.dt.item():.4f}")

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
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
    }, os.path.join(args.checkpoint_dir, 'model_final.pt'))
    print(f"Final model saved to {os.path.join(args.checkpoint_dir, 'model_final.pt')}")

if __name__ == "__main__":
    main()
