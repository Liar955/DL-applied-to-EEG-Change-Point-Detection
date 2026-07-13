"""Neural network architecture and checkpoint utilities."""

from __future__ import annotations

from ._core import core


AttnPool = core.AttnPool
EEG_CNN_LSTM = core.EEG_CNN_LSTM
load_saved_model = core.load_saved_model
build_model_for_inference = core.build_model_for_inference


def input_channels_from_config() -> int:
    """Infer the configured channel count from ``CHANNEL_SLICE``."""
    return int(core.CHANNEL_SLICE.stop - core.CHANNEL_SLICE.start)

