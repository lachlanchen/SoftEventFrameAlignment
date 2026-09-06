import json
import os
from pathlib import Path

import cv2
import numpy as np

try:
    import dv_processing as dv
except ImportError:  # The synthetic smoke test does not require AEDAT support.
    dv = None

def apply_synthetic_misalignment_to_events(events, scale=0.8, shift_x=0.1, shift_y=-0.15, shift_t=-0.5):
    """
    Apply a synthetic misalignment to event data for testing the alignment algorithm

    Args:
        events: Array of events with shape (n, 4) [x, y, t, polarity]
        scale: Scaling factor (applied as division to shrink)
        shift_x, shift_y, shift_t: Translation parameters

    Returns:
        Misaligned events
    """
    print(f"Applying synthetic misalignment to events: scale={scale}, shift_x={shift_x}, shift_y={shift_y}, shift_t={shift_t}")

    misaligned_events = events.copy()
    # Apply misalignment to events (divide by scale, add shifts)
    misaligned_events[:, 0] = (events[:, 0] - shift_x) * scale
    misaligned_events[:, 1] = (events[:, 1] - shift_y) * scale
    misaligned_events[:, 2] = events[:, 2] + shift_t

    return misaligned_events

def color_to_gray(frames_data):
    """
    Convert color frames to grayscale

    Args:
        frames_data: List of color frame images

    Returns:
        List of grayscale frame images
    """
    print("Converting color frames to grayscale...")
    gray_frames = []
    for frame in frames_data:
        if frame.ndim == 2:
            gray_frame = frame.copy()
        elif frame.ndim == 3 and frame.shape[2] in {3, 4}:
            conversion = cv2.COLOR_BGRA2GRAY if frame.shape[2] == 4 else cv2.COLOR_BGR2GRAY
            gray_frame = cv2.cvtColor(frame, conversion)
        else:
            raise ValueError(f"unsupported frame shape: {frame.shape}")
        gray_frames.append(gray_frame)
    return gray_frames


def require_dv_processing():
    if dv is None:
        raise RuntimeError(
            "AEDAT4 input requires dv-processing. Install requirements-aedat.txt "
            "or run the synthetic smoke test first."
        )

def safe_get(obj, attr):
    """Return obj.attr by calling it if callable, else return it directly."""
    val = getattr(obj, attr)
    return val() if callable(val) else val

def read_events(filepath, duration_seconds=10):
    """
    Reads events for the first 'duration_seconds' seconds (based on the first event's timestamp)
    from the AEDAT4 file.
    Each event is stored as a tuple (x, y, timestamp, polarity).

    Args:
        filepath: Path to the AEDAT4 file
        duration_seconds: Duration in seconds to read (default: 10s)

    Returns:
        events_array: numpy array of shape (n, 4) (dtype float64)
        t_start: timestamp of the first event from the first batch (used as baseline)
        cutoff: t_start + duration_seconds*1e6
    """
    require_dv_processing()
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if not Path(filepath).is_file():
        raise FileNotFoundError(filepath)

    duration_us = duration_seconds * 1_000_000  # AEDAT timestamps are microseconds.
    reader = dv.io.MonoCameraRecording(filepath)
    events_list = []
    batch_count = 0
    t_start = None

    print(f"Reading events for first {duration_seconds} seconds...")
    print("Starting to read event batches...")
    while reader.isRunning():
        batch = reader.getNextEventBatch()
        if batch is None:
            print("No more event batches available.")
            break
        batch_list = list(batch)  # ensure we can index
        batch_count += 1

        # Set t_start only from the very first event encountered.
        if t_start is None and len(batch_list) > 0:
            t_start = float(safe_get(batch_list[0], 'timestamp'))
            print(f"First event timestamp (t_start): {t_start}")

        if t_start is None:
            continue

        # Set cutoff based on t_start.
        cutoff = t_start + duration_us

        # Process events in the batch.
        for e in batch_list:
            t = float(safe_get(e, 'timestamp'))
            if t <= cutoff:
                events_list.append((
                    float(safe_get(e, 'x')),
                    float(safe_get(e, 'y')),
                    t,
                    float(safe_get(e, 'polarity'))
                ))
            else:
                print("Encountered event beyond cutoff timestamp; stopping this batch.")
                break

        if len(batch_list) > 0 and float(safe_get(batch_list[-1], 'timestamp')) > cutoff:
            print("Last event in batch exceeds cutoff; ending event read.")
            break

    if t_start is None or not events_list:
        raise ValueError("the recording did not contain events in the requested interval")
    events_array = np.asarray(events_list, dtype=np.float64)
    print(f"Total events read: {events_array.shape[0]} from {batch_count} batches.")
    print("First 5 events (x, y, timestamp, polarity):\n", events_array[:5])
    return events_array, t_start, cutoff

def read_frames(filepath, t_start, cutoff):
    """
    Reads frames that occur before the cutoff timestamp from the AEDAT4 file.

    Args:
        filepath: Path to the AEDAT4 file
        t_start: Starting timestamp (from first event)
        cutoff: Ending timestamp

    Returns:
        frames_array: numpy array of frame timestamps (float64)
        frames_data: list of frame images
    """
    require_dv_processing()
    reader = dv.io.MonoCameraRecording(filepath)
    frames_timestamps = []
    frames_data = []
    frame_count = 0

    print("Starting to read frames...")
    while reader.isRunning():
        frame = reader.getNextFrame()
        if frame is None:
            print("No more frames available.")
            break
        t = float(safe_get(frame, 'timestamp'))
        if t < t_start:
            continue
        if t <= cutoff:
            frame_count += 1
            frames_timestamps.append(t)
            frames_data.append(frame.image.copy())  # Store the image data
            if frame_count % 10 == 0:
                print(f"Frame {frame_count} timestamp: {t}")
            elif frame_count == 1:
                print(f"Frame {frame_count} timestamp: {t}")
                print("First frame acquired!")
        else:
            print(f"Encountered frame beyond cutoff timestamp; stopping frame read.")
            break

    if not frames_data:
        raise ValueError("the recording did not contain frames in the event interval")
    frames_array = np.asarray(frames_timestamps, dtype=np.float64)
    print(f"Total frames read: {frames_array.shape[0]}")
    return frames_array, frames_data

def normalize_events(events_array, t_start):
    """
    Normalizes event data:
    1. Subtracts t_start from timestamps
    2. Normalizes x, y to range [-1, 1]

    Args:
        events_array: Array of events with shape (n, 4) [x, y, timestamp, polarity]
        t_start: Starting timestamp

    Returns:
        Normalized events array
    """
    # Make a copy to avoid modifying the original
    normalized_events = events_array.copy()

    if events_array.ndim != 2 or events_array.shape[1] != 4 or not len(events_array):
        raise ValueError("events_array must be a non-empty N x 4 array")

    # Convert AEDAT microseconds to seconds relative to the first event.
    print("Converting event timestamps to seconds from t_start...")
    normalized_events[:, 2] = (normalized_events[:, 2] - t_start) / 1_000_000.0

    # Normalize x and y coordinates to [-1, 1]
    print("Normalizing x and y coordinates to range [-1, 1]...")
    x = normalized_events[:, 0]
    y = normalized_events[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    print(f"x_min: {x_min}, x_max: {x_max}")
    print(f"y_min: {y_min}, y_max: {y_max}")
    if x_max == x_min or y_max == y_min:
        raise ValueError("event coordinates must span more than one x and y position")
    normalized_events[:, 0] = 2 * (x - x_min) / (x_max - x_min) - 1
    normalized_events[:, 1] = 2 * (y - y_min) / (y_max - y_min) - 1

    print("First 5 events after normalization:\n", normalized_events[:5])
    return normalized_events, x_min, x_max, y_min, y_max

def normalize_frames_timestamps(frames_timestamps, t_start):
    """
    Normalizes frame timestamps by subtracting t_start

    Args:
        frames_timestamps: Array of frame timestamps
        t_start: Starting timestamp

    Returns:
        Normalized frame timestamps
    """
    print("Converting frame timestamps to seconds from t_start...")
    normalized_timestamps = (frames_timestamps - t_start) / 1_000_000.0
    print("First 5 frame timestamps after normalization:\n", normalized_timestamps[:5])
    return normalized_timestamps

def normalize_frames_data(frames_data):
    """
    Normalizes frame pixel values to [0, 1]

    Args:
        frames_data: List of frame images

    Returns:
        List of normalized frame images
    """
    print("Normalizing frame pixel values to range [0, 1]...")
    normalized_frames = []
    for frame in frames_data:
        # Convert to float and normalize to [0, 1]
        normalized_frame = frame.astype(np.float32) / 255.0
        normalized_frames.append(normalized_frame)
    return normalized_frames

def convert_frames_to_points(frames_data, frames_timestamps, num_points_per_frame=1000, rng=None):
    """
    Converts frame data to point-based representation (x, y, t, intensity)

    Args:
        frames_data: List of normalized frame images
        frames_timestamps: Array of normalized frame timestamps
        num_points_per_frame: Number of points to sample per frame

    Returns:
        Array of points with shape (n, 4) [x, y, t, intensity]
    """
    print("Converting frames to point-based representation...")
    if num_points_per_frame <= 0:
        raise ValueError("num_points_per_frame must be positive")
    if len(frames_data) != len(frames_timestamps) or not frames_data:
        raise ValueError("frames and timestamps must be non-empty and have equal length")
    rng = np.random.default_rng() if rng is None else rng
    all_points = []

    for i, (frame, timestamp) in enumerate(zip(frames_data, frames_timestamps)):
        # print("frame.shape: ", frame.shape)
        h, w = frame.shape
        if h < 2 or w < 2:
            raise ValueError("frames must be at least 2 x 2 pixels")

        # Randomly sample pixel coordinates for this frame
        sample_count = min(num_points_per_frame, h * w)
        indices = rng.choice(h * w, size=sample_count, replace=False)
        y_coords, x_coords = np.unravel_index(indices, (h, w))

        # Convert to normalized coordinates [-1, 1]
        x_norm = 2 * (x_coords / (w - 1)) - 1
        y_norm = 2 * (y_coords / (h - 1)) - 1

        # Get intensity values at the sampled coordinates
        intensities = frame[y_coords, x_coords]

        # Create points array for this frame
        t_values = np.full_like(x_norm, timestamp)
        frame_points = np.column_stack((x_norm, y_norm, t_values, intensities))
        all_points.append(frame_points)

    # Combine all frame points
    points_array = np.vstack(all_points)
    print(f"Converted frames to {points_array.shape[0]} points")
    print("First 5 frame points (x, y, t, intensity):\n", points_array[:5])
    return points_array

def prepare_data(
    filepath,
    duration_seconds=10,
    output_dir="data",
    *,
    synthetic_misalignment=False,
    seed=7,
):
    """
    Main function to prepare data for training:
    1. Read events and frames
    2. Normalize data
    3. Convert frames to point-based representation
    4. Save processed data to NPY files

    Args:
        filepath: Path to the AEDAT4 file
        duration_seconds: Duration in seconds to read
        output_dir: Directory to save processed data

    Returns:
        Dictionary containing processed data
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read events and frames
    events_array, t_start, cutoff = read_events(filepath, duration_seconds)
    frames_timestamps, frames_data = read_frames(filepath, t_start, cutoff)

    # Normalize events
    events_normalized, x_min, x_max, y_min, y_max = normalize_events(events_array, t_start)

    # In the prepare_data function, around line 250
    # Before normalizing, convert to grayscale
    frames_data = color_to_gray(frames_data)


    # Normalize frame timestamps and data
    frames_timestamps_normalized = normalize_frames_timestamps(frames_timestamps, t_start)
    frames_data_normalized = normalize_frames_data(frames_data)

    # Convert frames to point-based representation
    frame_points = convert_frames_to_points(
        frames_data_normalized,
        frames_timestamps_normalized,
        rng=np.random.default_rng(seed),
    )

    if synthetic_misalignment:
        events_normalized = apply_synthetic_misalignment_to_events(
            events_normalized,
            scale=0.8,
            shift_x=0.1,
            shift_y=-0.15,
            shift_t=-0.5,
        )

    # Save processed data
    np.save(os.path.join(output_dir, "events.npy"), events_normalized)
    np.save(os.path.join(output_dir, "frames_timestamps.npy"), frames_timestamps_normalized)
    np.save(os.path.join(output_dir, "frame_points.npy"), frame_points)
    metadata = {
        "schema": "softalign-processed/v1",
        "timestamp_unit": "seconds",
        "synthetic_misalignment": bool(synthetic_misalignment),
        "seed": int(seed),
        "event_count": int(len(events_normalized)),
        "frame_point_count": int(len(frame_points)),
    }
    with open(os.path.join(output_dir, "preprocessing.json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)
        file.write("\n")

    # Create a dictionary containing all processed data
    data = {
        "events": events_normalized,
        "frames_timestamps": frames_timestamps_normalized,
        "frame_points": frame_points,
        "metadata": metadata,
        "spatial_bounds": {
            "x_min": x_min, "x_max": x_max,
            "y_min": y_min, "y_max": y_max
        }
    }

    # Save a sample frame as an image for visualization
    if frames_data:
        cv2.imwrite(os.path.join(output_dir, "sample_frame.png"), frames_data[0])

    print(f"Processed data saved to {output_dir}/")
    return data
