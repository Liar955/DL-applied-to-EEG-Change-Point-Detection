"""Change point detection and statistical filtering functions."""

from __future__ import annotations

from ._core import core


moving_average = core.moving_average
cpd_from_embeddings = core.cpd_from_embeddings
remove_cps_near_bounds = core.remove_cps_near_bounds
first_enter_exit_times = core.first_enter_exit_times
keep_if_significant_ttest = core.keep_if_significant_ttest
cpd_on_files = core.cpd_on_files
cpd_on_subject_two_merged_blocks = core.cpd_on_subject_two_merged_blocks
cpd_on_subject_full_timeline = core.cpd_on_subject_full_timeline

