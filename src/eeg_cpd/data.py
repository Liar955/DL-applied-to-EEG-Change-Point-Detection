"""Data discovery, label handling, and subject timeline organization."""

from __future__ import annotations

from ._core import core


list_h5_by_group_and_subject_root = core.list_h5_by_group_and_subject_root
list_h5_in_target_subdir = core.list_h5_in_target_subdir
group_files_by_subject_then_state = core.group_files_by_subject_then_state
read_label = core.read_label
read_label_h5only = core.read_label_h5only
debug_check_csv_labels = core.debug_check_csv_labels
select_files_balanced_from_subset = core.select_files_balanced_from_subset
SubsetSequentialSampler = core.SubsetSequentialSampler
EEGWindowDataset = core.EEGWindowDataset


def scan_selected_files():
    """Return selected H5 files and the subject-to-files map."""
    return core.list_h5_by_group_and_subject_root(core.DATA_DIR, core.SELECTED_SUBJECTS)

