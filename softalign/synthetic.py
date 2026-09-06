"""Small deterministic data for installation and pipeline checks.

This module verifies that preprocessing-shaped arrays, training, checkpoints,
and evaluation can execute. It is not a benchmark of alignment accuracy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def make_synthetic_alignment_data(
    *,
    event_count: int = 256,
    frame_point_count: int = 256,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    if event_count <= 0 or frame_point_count <= 0:
        raise ValueError("sample counts must be positive")

    rng = np.random.default_rng(seed)
    aligned_event_coords = rng.uniform(
        (-0.75, -0.75, 0.0),
        (0.75, 0.75, 0.75),
        (event_count, 3),
    )
    frame_coords = rng.uniform(
        (-1.0, -1.0, 0.0),
        (1.0, 1.0, 1.0),
        (frame_point_count, 3),
    )

    def field(coords):
        phase = coords[:, 0] + 0.5 * coords[:, 1] + coords[:, 2]
        return 0.5 + 0.2 * np.sin(np.pi * phase)

    known_scale = 0.8
    known_shift_x = 0.05
    known_shift_y = 0.08
    known_shift_t = 1.0
    known_dt = 0.1
    event_coords = aligned_event_coords.copy()
    event_coords[:, 0] = (aligned_event_coords[:, 0] - known_shift_x) * known_scale
    event_coords[:, 1] = (aligned_event_coords[:, 1] - known_shift_y) * known_scale
    event_coords[:, 2] = aligned_event_coords[:, 2] - known_shift_t
    step = np.array([0.0, 0.0, known_dt])
    derivative = (
        field(aligned_event_coords + step) - field(aligned_event_coords)
    ) / known_dt
    event_values = (1.0 / (1.0 + np.exp(-derivative))).astype(np.float32)

    intensities = np.clip(field(frame_coords), 0.0, 1.0).astype(np.float32)

    events = np.column_stack((event_coords, event_values)).astype(np.float32)
    frame_points = np.column_stack((frame_coords, intensities)).astype(np.float32)
    return events, frame_points


def write_synthetic_dataset(
    output_dir: str | Path,
    *,
    event_count: int = 256,
    frame_point_count: int = 256,
    seed: int = 7,
) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    events, frame_points = make_synthetic_alignment_data(
        event_count=event_count,
        frame_point_count=frame_point_count,
        seed=seed,
    )
    np.save(output / "events.npy", events)
    np.save(output / "frame_points.npy", frame_points)
    metadata = {
        "schema": "softalign-processed/v1",
        "timestamp_unit": "seconds",
        "source": "deterministic-project-generated-synthetic-data",
        "seed": int(seed),
        "event_count": int(len(events)),
        "frame_point_count": int(len(frame_points)),
        "benchmark_claim": False,
    }
    (output / "preprocessing.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata
