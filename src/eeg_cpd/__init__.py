"""Modular access layer for the EEG change point detection project.

The original research implementation is preserved in ``cpd_legacy_core.py``.
This package provides a clearer GitHub-facing API organized by function:
data loading, preprocessing, model definition, training, inference, change
point detection, visualization, and full-pipeline runners.
"""

from . import change_point, config, data, inference, models, pipeline, preprocessing, training, visualization

__all__ = [
    "change_point",
    "config",
    "data",
    "inference",
    "models",
    "pipeline",
    "preprocessing",
    "training",
    "visualization",
]

