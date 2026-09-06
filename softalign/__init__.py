"""Public core API for the minimal Soft Event-Frame Alignment release."""

from .implicit_model import EventFrameAlignmentModel, ImplicitMLP
from .training import EventFrameDataset, train_model

__all__ = [
    "EventFrameAlignmentModel",
    "EventFrameDataset",
    "ImplicitMLP",
    "train_model",
]
