"""Modular command-line entry point for the EEG CPD project."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from eeg_cpd.config import load_runtime_config  # noqa: E402
from eeg_cpd.pipeline import detect_full_timeline, detect_two_blocks, prepare_project, train, visualize_embeddings  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run modular EEG change point detection workflows.")
    parser.add_argument("--config", default="configs/config.example.json", help="JSON config path.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["scan", "train", "detect-two-blocks", "detect-full-timeline", "embedding"],
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_runtime_config(args.config)

    if args.mode == "scan":
        prepare_project(cfg)
    elif args.mode == "train":
        train(cfg)
    elif args.mode == "detect-two-blocks":
        detect_two_blocks(cfg)
    elif args.mode == "detect-full-timeline":
        detect_full_timeline(cfg)
    elif args.mode == "embedding":
        visualize_embeddings(cfg)


if __name__ == "__main__":
    main()

