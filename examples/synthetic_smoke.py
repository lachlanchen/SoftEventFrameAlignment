#!/usr/bin/env python3
"""Run a tiny, deterministic CPU train/evaluate cycle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import evaluate_model
from softalign.implicit_model import EventFrameAlignmentModel
from softalign.synthetic import make_synthetic_alignment_data, write_synthetic_dataset
from softalign.training import EventFrameDataset, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=".smoke-output")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    output = Path(args.output_dir)
    data_dir = output / "data"
    checkpoint_dir = output / "checkpoints"
    evaluation_dir = output / "evaluation"
    write_synthetic_dataset(data_dir, seed=args.seed)
    events, frame_points = make_synthetic_alignment_data(seed=args.seed)

    torch.manual_seed(args.seed)
    dataset = EventFrameDataset(events, frame_points, batch_size=128, device="cpu")
    model = EventFrameAlignmentModel(hidden_dim=32, num_layers=3).to("cpu")
    model, losses, _ = train_model(
        model,
        dataset,
        num_epochs=args.epochs,
        lr=1e-3,
        checkpoint_dir=checkpoint_dir,
        log_interval=max(1, args.epochs),
        checkpoint_interval=max(1, args.epochs),
    )
    final_model = checkpoint_dir / "model_final.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "parameters": {
                "scale": model.scale.item(),
                "shift_x": model.shift_x.item(),
                "shift_y": model.shift_y.item(),
                "shift_t": model.shift_t.item(),
                "threshold": model.threshold.item(),
                "dt": model.dt.item(),
            },
            "model_config": {
                "hidden_dim": model.hidden_dim,
                "num_layers": model.num_layers,
            },
        },
        final_model,
    )
    results = evaluate_model(final_model, data_dir, evaluation_dir, device="cpu")
    print(
        json.dumps(
            {
                "ok": True,
                "epochs": args.epochs,
                "first_training_loss": losses[0],
                "last_training_loss": losses[-1],
                "evaluation": results["metrics"],
                "claim": "execution smoke test only; not an accuracy benchmark",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
