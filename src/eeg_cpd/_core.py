"""Internal bridge to the preserved legacy implementation."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

core = importlib.import_module("cpd_legacy_core")


def reset_label_cache() -> None:
    """Force the legacy label loader to reload CSV labels on the next access."""
    core.LABEL_MAP = None

