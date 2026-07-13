"""Runtime configuration helpers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._core import core, reset_label_cache


@dataclass
class RuntimeConfig:
    data_dir: str | None = None
    labels_csv: str | None = None
    output_dir: str = "results"
    checkpoint: str | None = None
    selected_subjects_json: str | None = None
    subject_token: str | None = None
    max_files: int = 6


def read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_runtime_config(path: str | Path | None) -> RuntimeConfig:
    values = read_json(path)
    return RuntimeConfig(**{k: v for k, v in values.items() if k in RuntimeConfig.__annotations__})


def apply_runtime_config(cfg: RuntimeConfig) -> Path:
    """Apply path and selection settings to the legacy core module."""
    if cfg.data_dir:
        core.DATA_DIR = str(cfg.data_dir)
    if cfg.labels_csv:
        core.CSV_PATH = str(cfg.labels_csv)
        reset_label_cache()

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    core.PLOT_DIR = str(plot_dir)
    os.makedirs(core.PLOT_DIR, exist_ok=True)

    if cfg.selected_subjects_json:
        core.SELECTED_SUBJECTS = read_json(cfg.selected_subjects_json)

    return output_dir

