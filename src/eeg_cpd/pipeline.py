"""High-level workflow functions for scan, train, detect, and visualize."""

from __future__ import annotations

import os
from pathlib import Path

from ._core import core
from .config import RuntimeConfig, apply_runtime_config
from .data import scan_selected_files
from .models import input_channels_from_config, load_saved_model


def prepare_project(cfg: RuntimeConfig):
    """Apply config and return ``(output_dir, files, subject_map)``."""
    output_dir = apply_runtime_config(cfg)
    files, subject_map = scan_selected_files()
    print(f"[scan] selected H5 files: {len(files)}")
    print(f"[scan] selected subjects: {len(subject_map)}")
    return output_dir, files, subject_map


def train(cfg: RuntimeConfig):
    """Train the original CNN-BiLSTM-Attention model with configured files."""
    output_dir, files, _ = prepare_project(cfg)
    old_cwd = Path.cwd()
    os.chdir(output_dir)
    try:
        model, used_files = core.train_and_validate(files)
    finally:
        os.chdir(old_cwd)
    checkpoint = output_dir / "cnn_eeg_improved_subset.pth"
    print(f"[train] checkpoint saved to: {checkpoint}")
    return model, used_files


def load_configured_model(cfg: RuntimeConfig, output_dir: Path):
    checkpoint = Path(cfg.checkpoint) if cfg.checkpoint else output_dir / "cnn_eeg_improved_subset.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return load_saved_model(str(checkpoint), in_ch=input_channels_from_config(), n_classes=core.N_CLASSES)


def detect_two_blocks(cfg: RuntimeConfig):
    output_dir, files, _ = prepare_project(cfg)
    model = load_configured_model(cfg, output_dir)
    save_txt = output_dir / "cp_times_two_blocks.txt"
    core.cpd_on_subject_two_merged_blocks(model, files, save_txt=str(save_txt))
    print(f"[detect-two-blocks] saved: {save_txt}")
    return save_txt


def detect_full_timeline(cfg: RuntimeConfig):
    output_dir, files, _ = prepare_project(cfg)
    model = load_configured_model(cfg, output_dir)
    save_txt = output_dir / "cp_times_full_timeline.txt"
    core.cpd_on_subject_full_timeline(model, files, save_txt=str(save_txt))
    print(f"[detect-full-timeline] saved: {save_txt}")
    return save_txt


def visualize_embeddings(cfg: RuntimeConfig):
    output_dir, files, _ = prepare_project(cfg)
    model = load_configured_model(cfg, output_dir)
    if cfg.subject_token:
        files_for_vis = [fp for fp in files if cfg.subject_token in os.path.basename(fp)]
    else:
        files_for_vis = []
    if not files_for_vis:
        files_for_vis = files[: cfg.max_files]

    feats, times, file_idx = core.extract_embeddings_from_files(
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
    return times, feats, file_idx

