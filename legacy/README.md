# Legacy source

`CPD_visualized_original.py` is a copy of the original research script kept for
traceability. It is not edited by the wrapper.

The importable copy at the project root, `cpd_legacy_core.py`, contains the same core implementation and is used by `scripts/run_modular.py` and the modular package in `src/eeg_cpd/`.

The modular package in `src/eeg_cpd/` wraps functions from `cpd_legacy_core.py` to provide a clearer API. 
The legacy core is kept to preserve the original experimental implementation and ensure reproducibility.

