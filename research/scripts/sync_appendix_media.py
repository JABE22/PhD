"""Sync dissertation media assets for Chapter 4 and appendices.

Run from project root:
    python3 research/scripts/sync_appendix_media.py

Use --dry-run to print planned updates without copying files.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DISSERTATION_ROOT = ROOT.parents[2] / "TeXstudio" / "Dissertation_v2"
MEDIA_MAP: list[tuple[str, str]] = [
    # chapter4
    ("research/data/example_scenarios_publication.png", "media/chapter4/example_scenarios_publication.png"),
    ("research/data/scenario_distribution.png", "media/chapter4/dataset_scenario_distribution.png"),
    ("research/test0_overview/results/figures/cross-test_model-comparison.png", "media/chapter4/cross-test_model-comparison.png"),
    ("research/test0_overview/results/figures/findings_dashboard_A.png", "media/chapter4/findings_dashboard_A.png"),
    ("research/test0_overview/results/figures/findings_dashboard_B.png", "media/chapter4/findings_dashboard_B.png"),
    ("research/test1_ontological-innovation/results/figures/classification_summary.png", "media/chapter4/t1_classification_summary.png"),
    ("research/test1_ontological-innovation/results/figures/novelty_frontier_bands_by_model.png", "media/chapter4/t1_novelty_frontier_bands_by_model.png"),
    ("research/test1_ontological-innovation/results/figures/novelty_frontier_decomposition_by_model_combined.png", "media/chapter4/t1_novelty_frontier_decomposition_by_model_combined.png"),
    ("research/test1_ontological-innovation/results/figures/embedding_space_analysis.png", "media/chapter4/t1_embedding_space_analysis.png"),
    ("research/test1_ontological-innovation/results/figures/similarity_heatmap_v1_optionA_aggregated.png", "media/chapter4/t1_similarity_heatmap_v1_optionA_aggregated.png"),
    ("research/test1_ontological-innovation/results/figures/06_canonical_by_model_heatmap.png", "media/chapter4/t1_canonical_by_model_heatmap.png"),
    ("research/test1_ontological-innovation/results/figures/continuous_novelty_score_by_model_ridgeline.png", "media/chapter4/t1_continuous_novelty_score_by_model_ridgeline.png"),
    ("research/test1_ontological-innovation/results/figures/raw_cluster_size_diagnostics.png", "media/chapter4/t1_raw_cluster_size_diagnostics.png"),
    ("research/test2_epistemic-agency/results/figures/category_distribution.png", "media/chapter4/t2_category_distribution.png"),
    ("research/test2_epistemic-agency/results/figures/taxonomy_distributions.png", "media/chapter4/t2_taxonomy_distributions.png"),
    ("research/test2_epistemic-agency/results/figures/similarity_analysis.png", "media/chapter4/t2_similarity_analysis.png"),
    ("research/test2_epistemic-agency/results/figures/class_decision_bias_map.png", "media/chapter4/t2_decision_bias_map.png"),
    ("research/test2_epistemic-agency/results/figures/question_embedding_space_sane.png", "media/chapter4/t2_embedding_space_sane.png"),
    ("research/test3_theory-generation/results/figures/category_distribution.png", "media/chapter4/t3_category_distribution.png"),
    ("research/test3_theory-generation/results/figures/detected_theories.png", "media/chapter4/t3_detected_theories.png"),
    ("research/test3_theory-generation/results/figures/theory_classification.png", "media/chapter4/t3_theory_classification.png"),
    ("research/test3_theory-generation/results/figures/alt_reduced_space_hull_diagnostic.png", "media/chapter4/t3_hull_manifold_novelty.png"),
    ("research/test3_theory-generation/results/figures/hybrid_and_nonhierarchical_categories.png", "media/chapter4/t3_hybrid_and_nonhierarchical_categories.png"),
    ("research/test4_category-recognition/results/figures/category_awareness_scores_by_model.png", "media/chapter4/t4_category_awareness_scores_by_model.png"),
    ("research/test4_category-recognition/results/figures/score_distributions.png", "media/chapter4/t4_score_distributions.png"),
    ("research/test4_category-recognition/results/figures/embedding_space_analysis.png", "media/chapter4/t4_embedding_space_analysis.png"),
    ("research/test5_mechanistic/results/figures/attention_heatmap_real.png", "media/chapter4/t5_attention_diagnostics.png"),
    ("research/test5_mechanistic/results/figures/activation_space_comparison.png", "media/chapter4/t5_activation_space_comparison.png"),
    ("research/test5_mechanistic/results/figures/layer_progression.png", "media/chapter4/t5_layer_progression.png"),
    ("research/test5_mechanistic/results/figures/token_importance.png", "media/chapter4/t5_next_token_attention.png"),
    # appendices
    ("research/data/example_scenarios_compact.png", "media/appendices/example_scenarios_compact.png"),
    ("research/data/scenario_distribution.png", "media/appendices/scenario_distribution.png"),
    ("research/test1_ontological-innovation/results/figures/similarity_heatmap_v1_part_a.png", "media/appendices/t1_similarity_heatmap_v1_part_a.png"),
    ("research/test1_ontological-innovation/results/figures/similarity_heatmap_v1_part_b.png", "media/appendices/t1_similarity_heatmap_v1_part_b.png"),
    ("research/test1_ontological-innovation/results/figures/hierarc_clustering_dendrogram.png", "media/appendices/t1_hierarc_clustering_dendrogram.png"),
    ("research/test1_ontological-innovation/results/figures/tsne_with_labels.png", "media/appendices/t1_tsne_with_labels.png"),
    ("research/test1_ontological-innovation/results/figures/extra_visualizations_row1.png", "media/appendices/t1_extra_visualizations_row1.png"),
    ("research/test1_ontological-innovation/results/figures/extra_visualizations_row2.png", "media/appendices/t1_extra_visualizations_row2.png"),
    ("research/test2_epistemic-agency/results/figures/question_classification.png", "media/appendices/t2_question_classification.png"),
    ("research/test2_epistemic-agency/results/figures/projection_reliability_diagnostics.png", "media/appendices/t2_projection_reliability_diagnostics.png"),
    ("research/test2_epistemic-agency/results/figures/diagnostic_visualizations.png", "media/appendices/t2_diagnostic_visualizations.png"),
    ("research/test2_epistemic-agency/results/figures/projection_method_comparison.png", "media/appendices/t2_projection_method_comparison.png"),
    ("research/test3_theory-generation/results/figures/cf_vs_novelty.png", "media/appendices/t3_cf_vs_novelty.png"),
    ("research/test3_theory-generation/results/figures/score_distributions.png", "media/appendices/t3_score_distributions.png"),
    ("research/test3_theory-generation/results/figures/alternative_hypothesis_visualizations.png", "media/appendices/t3_alternative_hypothesis_visualizations.png"),
    ("research/test3_theory-generation/results/figures/theory_similarity_heatmap.png", "media/appendices/t3_similarity_heatmap.png"),
    ("research/test3_theory-generation/results/figures/theory_dendrogram_full.png", "media/appendices/t3_dendrogram_full.png"),
    ("research/test3_theory-generation/results/figures/theory_dendrogram_truncated.png", "media/appendices/t3_dendrogram_truncated.png"),
    ("research/test3_theory-generation/results/figures/theory_cluster_map_setupA_centroids_hulls.png", "media/appendices/t3_cluster_centroids_hulls.png"),
    ("research/test3_theory-generation/results/figures/theory_cluster_map_setupB_novelty_colors.png", "media/appendices/t3_novelty_colors.png"),
    ("research/test3_theory-generation/results/figures/theory_cluster_map_setupC_coarse_clusters.png", "media/appendices/t3_coarse_clusters.png"),
    ("research/test4_category-recognition/results/figures/total_score_by_model.png", "media/appendices/t4_total_score_by_model.png"),
    ("research/test5_mechanistic/results/figures/attention_heads_grid.png", "media/appendices/t5_attention_heads_grid.png"),
    ("research/test5_mechanistic/results/figures/attention_rollout.png", "media/appendices/t5_attention_rollout.png"),
    ("research/test5_mechanistic/results/figures/neuron_view.html", "media/appendices/t5_neuron_view.html"),
    ("research/test6_cross_model/results/figures/cross_test_summary_ab.png", "media/appendices/cross_model_robustness_summary_ab.png"),
    ("research/test6_cross_model/results/figures/cross_test_summary_cd.png", "media/appendices/cross_model_robustness_summary_cd.png"),
    ("research/test6_cross_model/results/figures/model_performance_heatmap.png", "media/appendices/cross_model_performance_heatmap.png"),

]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync dissertation media assets from research outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Report updates without copying files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    updated: list[str] = []
    current: list[str] = []
    missing: list[str] = []

    for src_rel, dst_rel in MEDIA_MAP:
        src = ROOT / src_rel
        dst = DISSERTATION_ROOT / dst_rel

        if not src.exists():
            missing.append(f"{src_rel} -> {dst_rel}")
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        should_copy = (not dst.exists()) or (src.stat().st_mtime > dst.stat().st_mtime + 1e-6)
        if should_copy:
            if not args.dry_run:
                shutil.copy2(src, dst)
            updated.append(f"{dst} <= {src_rel}")
        else:
            current.append(str(dst))

    mode = "DRY-RUN" if args.dry_run else "SYNC"
    print(f"[{mode}] pairs checked: {len(MEDIA_MAP)}")
    print(f"[{mode}] updated: {len(updated)}")
    print(f"[{mode}] current: {len(current)}")
    print(f"[{mode}] missing sources: {len(missing)}")

    if updated:
        print("Updated assets:")
        for item in updated:
            print(f" - {item}")

    if missing:
        print("Missing source mappings:")
        for item in missing:
            print(f" - {item}")


if __name__ == "__main__":
    main()
