"""Window-level probability and embedding inference helpers."""

from __future__ import annotations

from ._core import core


predict_probs_on_signal = core.predict_probs_on_signal
predict_feats_on_signal = core.predict_feats_on_signal
predict_probs_on_file = core.predict_probs_on_file
predict_probs_on_array = core.predict_probs_on_array
predict_feats_on_file = core.predict_feats_on_file
extract_embeddings_from_files = core.extract_embeddings_from_files
audit_invariance = core.audit_invariance
run_audit_over_dir = core.run_audit_over_dir

