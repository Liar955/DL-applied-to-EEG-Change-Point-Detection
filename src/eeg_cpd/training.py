"""Training and validation entry points."""

from __future__ import annotations

from ._core import core


train_and_validate = core.train_and_validate
chronological_split_by_file = core.chronological_split_by_file
safe_roc_auc = core.safe_roc_auc

