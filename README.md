# Deep Learning Applied to EEG Change Point Detection

This is a conservative modular cleanup of the original EEG change point
detection research script. The goal is to keep the original working algorithm
intact while exposing a clearer GitHub-facing project structure.

## What changed

- The original script is preserved in `legacy/CPD_visualizaed_original.py`.
- The importable core implementation is kept as `cpd_legacy_core.py`.
- `src/eeg_cpd/` exposes the project through functional modules.
- `scripts/run_modular.py` provides the recommended command-line entry point.
- `notebooks/eeg_cpd_workflow.ipynb` provides a notebook-style research workflow.
- Local paths are passed through command-line arguments or `configs/config.example.json`.
- Generated plots and text outputs go under `results/`.
- Private EEG files, H5 files, EDF files, and model checkpoints are ignored by Git.

This version intentionally avoids a large algorithm rewrite. The model, dataset,
normalization, embedding extraction, and CPD functions still come from the
original implementation, but the visible API is split by function.

## Project structure

```text
.
|-- cpd_legacy_core.py
|-- src/eeg_cpd/
|   |-- config.py
|   |-- data.py
|   |-- preprocessing.py
|   |-- models.py
|   |-- training.py
|   |-- inference.py
|   |-- change_point.py
|   |-- visualization.py
|   `-- pipeline.py
|-- scripts/
|   |-- run_script.py
|   `-- run_modular.py
|-- notebooks/
|   `-- eeg_cpd_workflow.ipynb
|-- configs/
|   |-- config.example.json
|   `-- selected_subjects.example.json
|-- legacy/
|   |-- CPD_visualizaed_original.py
|   `-- README.md
|-- matlab/
|   `-- xfyp_data_modified.m
|-- results/
|-- plots/
|-- requirements.txt
`-- .gitignore
```

## Install dependencies

Create a clean environment if you want to run the project:

```bash
conda create -n eeg-cpd python=3.10
conda activate eeg-cpd
pip install -r requirements.txt
```

For GitHub upload only, no conda environment is required.

## Configure paths

Copy the example config and edit paths if needed:

```bash
copy configs\config.example.json configs\config.local.json
```

Important fields:

- `data_dir`: root folder containing filtered `.h5` EEG files
- `labels_csv`: CSV file with file-level labels
- `output_dir`: directory for checkpoints, CPD tables, and plots
- `checkpoint`: path to the trained model checkpoint
- `subject_token`: optional filename token for embedding visualization

## Run

cpd_legacy_core.py is retained for reproducibility and should not be run directly; use scripts/run_modular.py instead.

Scan selected H5 files:

```bash
python scripts/run_modular.py --config configs/config.local.json --mode scan
```

Train the original CNN-BiLSTM-Attention model:

```bash
python scripts/run_modular.py --config configs/config.local.json --mode train
```

Run two-block subject-level CPD:

```bash
python scripts/run_modular.py --config configs/config.local.json --mode detect-two-blocks
```

Run full-timeline subject-level CPD:

```bash
python scripts/run_modular.py --config configs/config.local.json --mode detect-full-timeline
```

Visualize learned embeddings:

```bash
python scripts/run_modular.py --config configs/config.local.json --mode embedding
```

## Modular API

The recommended Python-facing modules are:

- `eeg_cpd.data`: file scanning, label handling, subject timeline grouping
- `eeg_cpd.preprocessing`: robust normalization and signal concatenation
- `eeg_cpd.models`: CNN-BiLSTM-Attention model and checkpoint loading
- `eeg_cpd.training`: training and validation entry points
- `eeg_cpd.inference`: probability and embedding extraction
- `eeg_cpd.change_point`: feature-space CPD and statistical filtering
- `eeg_cpd.visualization`: probability, heatmap, PCA/t-SNE, and EEG plots
- `eeg_cpd.pipeline`: high-level scan/train/detect/visualize workflows

Example:

```python
from eeg_cpd.config import load_runtime_config
from eeg_cpd.pipeline import prepare_project, visualize_embeddings

cfg = load_runtime_config("configs/config.local.json")
output_dir, files, subject_map = prepare_project(cfg)
times, feats, file_idx = visualize_embeddings(cfg)
```

## Data source and exclusions

The project is based on EEG data associated with a rat status epilepticus study
on hippocampal ripple oscillation energy and gap junction blockers. The local
dataset is organized into five folders: blank control, VPA, sponges, PILO, and
mir overexpression groups. Each group contains individual animal folders with
recordings across the modeling timeline.

Raw EEG data and trained checkpoints are not included because of privacy and file-size constraints.

The blank control group and some recordings from the remaining groups were not
used when EDF header or acquisition-format problems prevented reliable reading.
These exclusions should be documented as data-quality filters.
