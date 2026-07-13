"""Plotting, embedding visualization, and signal-summary figures."""

from __future__ import annotations

from ._core import core


plot_prob_with_cps = core.plot_prob_with_cps
plot_channel_energy_heatmap = core.plot_channel_energy_heatmap
plot_subject_three_phases = core.plot_subject_three_phases
boxplot_pe_by_timeslots = core.boxplot_pe_by_timeslots
plot_hfo_topography_for_subject = core.plot_hfo_topography_for_subject
plot_group_cumulative_cps = core.plot_group_cumulative_cps
plot_embedding_2d = core.plot_embedding_2d
enhanced_tsne_visualization = core.enhanced_tsne_visualization
temporal_embedding_visualization = core.temporal_embedding_visualization
plot_coherence_matrix_for_subject = core.plot_coherence_matrix_for_subject
evaluate_embedding_quality = core.evaluate_embedding_quality
compute_silhouette = core.compute_silhouette

