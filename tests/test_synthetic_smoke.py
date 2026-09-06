from __future__ import annotations

import math
import unittest

import torch
import torch.nn.functional as F

from softalign.implicit_model import EventFrameAlignmentModel
from softalign.synthetic import make_synthetic_alignment_data
from softalign.training import EventFrameDataset, train_model


class SyntheticSmokeTests(unittest.TestCase):
    def test_cpu_training_and_evaluation_are_finite(self):
        events, frame_points = make_synthetic_alignment_data(
            event_count=64,
            frame_point_count=64,
            seed=11,
        )
        dataset = EventFrameDataset(events, frame_points, batch_size=64, device="cpu")
        torch.manual_seed(11)
        model = EventFrameAlignmentModel(hidden_dim=16, num_layers=2)

        def full_loss():
            return F.mse_loss(
                model.forward_event(dataset.events), dataset.polarities
            ) + F.mse_loss(
                model.forward_frame(dataset.frame_points), dataset.intensities
            )

        with torch.no_grad():
            before = full_loss().item()
        model, losses, history = train_model(
            model,
            dataset,
            num_epochs=25,
            lr=2e-3,
            lambda_reg=0.0,
            checkpoint_dir=None,
            make_plots=False,
            log_interval=25,
        )
        with torch.no_grad():
            after = full_loss().item()

        self.assertEqual(len(losses), 25)
        self.assertTrue(all(math.isfinite(loss) for loss in losses))
        self.assertLess(after, before * 0.8)
        self.assertEqual(model.forward_event(dataset.events).shape, (64, 1))
        self.assertEqual(model.forward_frame(dataset.frame_points).shape, (64, 1))
        self.assertEqual(
            set(history),
            {"scale", "shift_x", "shift_y", "shift_t", "threshold", "dt"},
        )
        self.assertTrue(all(torch.isfinite(value).all() for value in model.parameters()))


if __name__ == "__main__":
    unittest.main()
