import argparse
import json
import os

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
import torch

from softalign.implicit_model import EventFrameAlignmentModel

def create_event_image(events, width=346, height=260, sigma=1.0):
    """
    Create an image from event data

    Args:
        events: Array of events with shape (n, 4) [x_norm, y_norm, t, polarity]
        width: Image width
        height: Image height
        sigma: Gaussian smoothing sigma

    Returns:
        Event image
    """
    # Convert normalized coordinates to pixel coordinates
    x = np.round((events[:, 0] + 1) / 2 * (width - 1)).astype(int)
    y = np.round((events[:, 1] + 1) / 2 * (height - 1)).astype(int)
    p = events[:, 3]  # polarity

    # Create positive and negative event images
    pos_img = np.zeros((height, width), dtype=np.float32)
    neg_img = np.zeros((height, width), dtype=np.float32)

    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    positive = valid & (p > 0)
    negative = valid & ~positive
    np.add.at(pos_img, (y[positive], x[positive]), 1)
    np.add.at(neg_img, (y[negative], x[negative]), 1)

    # Apply Gaussian smoothing
    pos_img = gaussian_filter(pos_img, sigma=sigma)
    neg_img = gaussian_filter(neg_img, sigma=sigma)

    # Normalize images to [0, 1]
    if pos_img.max() > 0:
        pos_img /= pos_img.max()
    if neg_img.max() > 0:
        neg_img /= neg_img.max()

    # Create RGB image (positive events: green, negative events: red)
    event_img = np.zeros((height, width, 3), dtype=np.float32)
    event_img[:, :, 0] = neg_img  # Red channel for negative events
    event_img[:, :, 1] = pos_img  # Green channel for positive events

    return event_img

def evaluate_model(model_path, data_dir, output_dir='evaluation', device='cuda'):
    """
    Evaluate the trained model and visualize alignment results

    Args:
        model_path: Path to trained model checkpoint
        data_dir: Directory containing processed data
        output_dir: Directory to save evaluation results
        device: Device to use for inference
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    events = np.load(os.path.join(data_dir, 'events.npy'))
    frame_points = np.load(os.path.join(data_dir, 'frame_points.npy'))
    metadata_path = os.path.join(data_dir, 'preprocessing.json')
    if not os.path.exists(metadata_path):
        raise ValueError(
            f'{metadata_path} is missing; evaluation requires explicit preprocessing metadata'
        )
    with open(metadata_path, encoding='utf-8') as file:
        metadata = json.load(file)
    if metadata.get('schema') != 'softalign-processed/v1' or metadata.get('timestamp_unit') != 'seconds':
        raise ValueError('processed data metadata is incompatible; regenerate the arrays')

    # Try to load sample frame for visualization
    sample_frame_path = os.path.join(data_dir, 'sample_frame.png')
    if os.path.exists(sample_frame_path):
        sample_frame = cv2.imread(sample_frame_path)
        if sample_frame is None:
            raise ValueError(f'OpenCV could not read {sample_frame_path}')
        sample_frame = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB)
    else:
        sample_frame = None

    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA was requested but is not available; use --device cpu')
    if events.ndim != 2 or events.shape[1] != 4 or not len(events):
        raise ValueError('events.npy must contain a non-empty N x 4 array')
    if frame_points.ndim != 2 or frame_points.shape[1] != 4 or not len(frame_points):
        raise ValueError('frame_points.npy must contain a non-empty N x 4 array')

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model_config = checkpoint.get('model_config', {})
    model = EventFrameAlignmentModel(
        hidden_dim=int(model_config.get('hidden_dim', 128)),
        num_layers=int(model_config.get('num_layers', 4)),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Print model parameters
    params = checkpoint.get('parameters') or {
        'scale': model.scale.item(),
        'shift_x': model.shift_x.item(),
        'shift_y': model.shift_y.item(),
        'shift_t': model.shift_t.item(),
        'threshold': model.threshold.item(),
        'dt': model.dt.item(),
    }
    print(f"Model parameters:")
    print(f"  Scale = {params['scale']:.4f}, "
          f"shift_x = {params['shift_x']:.4f}, "
          f"shift_y = {params['shift_y']:.4f}, "
          f"shift_t = {params['shift_t']:.4f}")
    print(f"  Threshold = {params['threshold']:.4f}, "
          f"dt = {params['dt']:.4f}")

    # Apply transformation to events
    display_scale = float(params['scale'])
    if abs(display_scale) < 1e-8:
        raise ValueError('learned scale is too close to zero for stable visualization')
    transformed_events = events.copy()
    transformed_events[:, 0] = events[:, 0] / display_scale + params['shift_x']
    transformed_events[:, 1] = events[:, 1] / display_scale + params['shift_y']
    transformed_events[:, 2] = events[:, 2] + params['shift_t']

    # Create event images for visualization
    original_event_img = create_event_image(events)
    transformed_event_img = create_event_image(transformed_events)

    # Visualize event alignment with sample frame (if available).
    if sample_frame is not None:
        # Resize frame to match event image size
        h, w = original_event_img.shape[:2]
        sample_frame_resized = cv2.resize(sample_frame, (w, h))
        sample_frame_normalized = sample_frame_resized.astype(np.float32) / 255.0

        # Overlay events on frame
        alpha = 0.7
        orig_overlay = alpha * sample_frame_normalized + (1 - alpha) * original_event_img
        transformed_overlay = alpha * sample_frame_normalized + (1 - alpha) * transformed_event_img

        # Create comparison visualization
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 2, 1)
        plt.imshow(sample_frame_normalized)
        plt.title('Sample Frame')
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.imshow(original_event_img)
        plt.title('Original Events')
        plt.axis('off')

        plt.subplot(2, 2, 3)
        plt.imshow(transformed_event_img)
        plt.title('Transformed Events')
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.imshow(transformed_overlay)
        plt.title('Overlay: Frame + Transformed Events')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'alignment_visualization.png'))
        plt.close()
    else:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(original_event_img)
        plt.title('Original Events')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(transformed_event_img)
        plt.title('Transformed Events')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'alignment_visualization.png'))
        plt.close()

    # These are bounded reconstruction diagnostics over the supplied arrays,
    # not held-out accuracy or alignment-recovery metrics.
    print("Computing reconstruction diagnostics...")

    # Convert to PyTorch tensors
    event_coords = torch.tensor(events[:5000, :3], dtype=torch.float32).to(device)
    event_polarities = torch.tensor(events[:5000, 3], dtype=torch.float32).view(-1, 1).to(device)

    frame_coords = torch.tensor(frame_points[:5000, :3], dtype=torch.float32).to(device)
    frame_intensities = torch.tensor(frame_points[:5000, 3], dtype=torch.float32).view(-1, 1).to(device)

    # Forward pass
    with torch.no_grad():
        event_response = model.forward_event(event_coords)
        frame_response = model.forward_frame(frame_coords)

    # Compute metrics
    event_loss = torch.nn.functional.mse_loss(event_response, event_polarities).item()
    frame_loss = torch.nn.functional.mse_loss(frame_response, frame_intensities).item()

    print(f"Evaluation results:")
    print(f"  Event loss = {event_loss:.6f}")
    print(f"  Frame loss = {frame_loss:.6f}")

    results = {
        'scope': 'bounded reconstruction diagnostics on supplied arrays; not a held-out accuracy benchmark',
        'preprocessing': metadata,
        'parameters': {key: float(value) for key, value in params.items()},
        'metrics': {
            'event_mse': event_loss,
            'frame_mse': frame_loss,
        },
        'sample_counts': {
            'events': min(5000, len(events)),
            'frame_points': min(5000, len(frame_points)),
        },
    }
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w', encoding='utf-8') as file:
        json.dump(results, file, indent=2, sort_keys=True)
        file.write('\n')

    print(f"Evaluation results saved to {output_dir}/")
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate event-frame alignment model')
    parser.add_argument('--model_path', type=str, default='checkpoints/model_final.pt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing processed data')
    parser.add_argument('--output_dir', type=str, default='evaluation',
                        help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for inference')
    args = parser.parse_args()
    evaluate_model(args.model_path, args.data_dir, args.output_dir, args.device)


if __name__ == "__main__":
    main()
