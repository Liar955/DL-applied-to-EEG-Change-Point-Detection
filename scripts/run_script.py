"""Command-line runner for the original EEG CPD research script.

This wrapper is intentionally conservative: it imports the original script as
`cpd_legacy_core` and overrides only runtime settings such as data paths,
label paths, checkpoint paths, and output directories. The model, dataset,
normalization, CPD, and plotting functions remain the original implementation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cpd_legacy_core as core  # noqa: E402


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _value(args, cfg: dict[str, Any], name: str, default=None):
    value = getattr(args, name, None)
    if value is not None:
        return value
    return cfg.get(name, default)


def apply_runtime_config(args, cfg: dict[str, Any]) -> Path:
    data_dir = _value(args, cfg, "data_dir", core.DATA_DIR)
    labels_csv = _value(args, cfg, "labels_csv", core.CSV_PATH)
    output_dir = Path(_value(args, cfg, "output_dir", PROJECT_ROOT / "results"))
    selected_subjects_json = _value(args, cfg, "selected_subjects_json", None)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    core.DATA_DIR = str(data_dir)
    core.CSV_PATH = str(labels_csv) if labels_csv else ""
    core.LABEL_MAP = None
    core.PLOT_DIR = str(plot_dir)
    os.makedirs(core.PLOT_DIR, exist_ok=True)

    if selected_subjects_json:
        core.SELECTED_SUBJECTS = _read_json(selected_subjects_json)

    return output_dir


def collect_subset_files():
    files, subject_map = core.list_h5_by_group_and_subject_root(core.DATA_DIR, core.SELECTED_SUBJECTS)
    print(f"[scan] selected H5 files: {len(files)}")
    print(f"[scan] selected subjects: {len(subject_map)}")
    return files, subject_map


def load_model(checkpoint: str | Path):
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}. Run --mode train first or pass --checkpoint."
        )
    in_ch = core.CHANNEL_SLICE.stop - core.CHANNEL_SLICE.start
    return core.load_saved_model(str(checkpoint), in_ch=in_ch, n_classes=core.N_CLASSES)


def run_train(files, output_dir: Path):
    old_cwd = Path.cwd()
    os.chdir(output_dir)
    try:
        model, used_files = core.train_and_validate(files)
    finally:
        os.chdir(old_cwd)

    saved = output_dir / "cnn_eeg_improved_subset.pth"
    print(f"[train] checkpoint saved to: {saved}")
    return model, used_files


def run_detect_two_blocks(model, files, output_dir: Path):
    save_txt = output_dir / "cp_times_two_blocks.txt"
    core.cpd_on_subject_two_merged_blocks(model, files, save_txt=str(save_txt))
    print(f"[detect-two-blocks] saved change point table to: {save_txt}")


def run_detect_full_timeline(model, files, output_dir: Path):
    save_txt = output_dir / "cp_times_full_timeline.txt"
    core.cpd_on_subject_full_timeline(model, files, save_txt=str(save_txt))
    print(f"[detect-full-timeline] saved change point table to: {save_txt}")


def run_embedding_plot(model, files, output_dir: Path, subject_token: str | None, max_files: int):
    if subject_token:
        files_for_vis = [fp for fp in files if subject_token in os.path.basename(fp)]
    else:
        files_for_vis = []
    if not files_for_vis:
        files_for_vis = files[:max_files]

    times, feats, file_idx = core.extract_embeddings_from_files(
        model,
        files_for_vis,
        win_sec=core.WIN_SEC,
        stride_sec=core.STRIDE_SEC,
        ds_step=core.DS_STEP,
        device=core.DEVICE,
    )
    print(f"[embedding] files={len(files_for_vis)} embeddings={getattr(feats, 'shape', None)}")
    core.plot_embedding_2d(
        feats,
        file_idx,
        methods=["pca", "tsne"],
        title_prefix="EEG embeddings",
        save_dir=str(output_dir / "plots"),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Conservative runner for EEG change point detection.")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config file.")
    parser.add_argument("--mode", choices=["scan", "train", "detect-two-blocks", "detect-full-timeline", "embedding"], required=True)
    parser.add_argument("--data-dir", type=str, default=None, help="Root directory containing filtered H5 files.")
    parser.add_argument("--labels-csv", type=str, default=None, help="CSV file with filename,label columns.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for generated results and plots.")
    parser.add_argument("--selected-subjects-json", type=str, default=None, help="Optional JSON file overriding SELECTED_SUBJECTS.")
    parser.add_argument("--subject-token", type=str, default=None, help="Filename token used by --mode embedding.")
    parser.add_argument("--max-files", type=int, default=6, help="Fallback number of files for embedding visualization.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = _read_json(args.config)
    output_dir = apply_runtime_config(args, cfg)
    files, _ = collect_subset_files()

    if args.mode == "scan":
        return

    if args.mode == "train":
        run_train(files, output_dir)
        return

    checkpoint = _value(args, cfg, "checkpoint", output_dir / "cnn_eeg_improved_subset.pth")
    model = load_model(checkpoint)

    if args.mode == "detect-two-blocks":
        run_detect_two_blocks(model, files, output_dir)
    elif args.mode == "detect-full-timeline":
        run_detect_full_timeline(model, files, output_dir)
    elif args.mode == "embedding":
        subject_token = _value(args, cfg, "subject_token", None)
        max_files = int(_value(args, cfg, "max_files", args.max_files))
        run_embedding_plot(model, files, output_dir, subject_token, max_files)


if __name__ == "__main__":
    main()

