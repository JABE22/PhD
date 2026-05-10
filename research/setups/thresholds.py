"""
Global analysis thresholds for the AI Creativity / Consciousness project.

Usage in each notebook (replace explicit constant definitions)::

    from setups.thresholds import *   # imports all T1_/T2_/T3_/T4_ names
    # then create local aliases matching the names used in downstream cells,
    # e.g.:  TRACEABILITY_SIM_THRESHOLD = T1_TRACEABILITY_SIM_THRESHOLD

Cross-test consistency notes
------------------------------
All cosine similarity values are computed with the sentence embedding model
``all-MiniLM-L6-v2`` unless stated otherwise.

The "traceability" concept appears in every test but with different corpus
sizes and text lengths, which justifies the different numeric levels.
Thresholds should remain coarse and interpretable (for example 0.3/0.4/0.5/0.7/0.75),
avoiding over-fine tuning unless there is a strong methodological reason.

        Test 1  → short structured texts vs. ~500-doc corpus  → 0.40
    Test 2  → full questions vs. ~4000-doc corpus          → defaults: strict 0.75 / lenient 0.70
                     → active values may be quantile-calibrated when enabled
    Test 3  → full theory texts vs. ~20 known theories    → 0.70 (strict), 0.30 (novelty gate)
        Test 4  → analytical prose vs. 7 reference arguments  → 0.40

Any deliberate difference from the conservative shared value must appear in
the comments here AND be mentioned in the manuscript (Section 5).
"""

# ===========================================================================
# Shared / cross-test constants
# ===========================================================================

# Sentence embedding model used everywhere for semantic similarity.
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Minimum acceptable AI response length (words).
# Applied as a quality gate in all tests before further analysis.
MIN_RESPONSE_WORDS = 100


# ===========================================================================
# Test 1 — Ontological Innovation (Modality Proposals)
# ===========================================================================

# Cosine similarity threshold: proposal ↔ corpus document.
# Below this value the match is rejected as "not traceable".
# Lower than T2/T3 because proposals are short structured texts compared
# against a smaller, heterogeneous corpus (OpenAlex + Crossref, ~500 docs).
# Candidate sweep on current Test 1 data (0.30/0.40/0.50) showed 0.40 is the
# best operating point: 0.30 is too permissive, 0.50 is too restrictive.
T1_TRACEABILITY_SIM_THRESHOLD  = 0.40
T1_TRACEABILITY_TOP_K          = 10   # top-k corpus matches returned per proposal
T1_TRACEABILITY_MIN_QUERY_CHARS = 60  # minimum proposal text length for retrieval
T1_TRACEABILITY_MIN_DOC_CHARS   = 120 # minimum cleaned corpus document length

# Cosine similarity threshold: proposal ↔ modality description.
# Higher than the literature traceability threshold because this gate is used
# to identify proposals that remain too close to the known modality manifold.
# Section notebooks use it when reconstructing the rule frontier distance for
# ontological novelty diagnostics and decomposition plots.
T1_NOVELTY_FRONTIER_SIM_THRESHOLD = 0.75

# Structural decomposition
T1_STRUCTURAL_EXTENSION_MATCH_MIN = 2   # keyword hits required to label as "extension"
T1_STRUCTURAL_HYBRID_MODALITY_MIN = 2   # related modalities required to label as "hybrid"
T1_HULL_MEMBERSHIP_TOLERANCE      = 1e-6

# Embedding space analysis
T1_PCA_N_COMPONENTS = 3   # target dimensionality for PCA reduction before hull / scatter

# Similarity heatmap display
T1_SIMILARITY_HEATMAP_VMIN  = 0.0
T1_SIMILARITY_HEATMAP_VMAX  = 1.0
T1_EXTREMES_GROUP_SIZE      = 20  # proposals shown in extreme-view heatmap

# Literature collection (OpenAlex + Crossref)
T1_COLLECT_MIN_TEXT_CHARS = 200
T1_OPENALEX_PER_QUERY     = 25
T1_OPENALEX_PAGES         = 2
T1_CROSSREF_PER_QUERY     = 30
T1_MAX_QUERY_TERMS        = 30
T1_TARGET_DOCS            = 500


# ===========================================================================
# Test 2 — Epistemic Agency (Research Question Formation)
# ===========================================================================

# Question classification
T2_CLASS_SIM_MARGIN      = 0.02           # minimum gap between top-2 class cosine sims
T2_CLASS_FALLBACK_LABEL  = "exploratory"  # label assigned when margin is too small

# Framework transcendence detection
T2_TRANSCENDENCE_MARGIN_THRESHOLD = 0.02  # sim(transcend) − sim(within)
T2_TRANSCENDENCE_MIN_SIM          = 0.35  # absolute minimum sim to transcend anchor

# Literature traceability (strict / lenient pair).
# Higher than T1 because the corpus is ~8× larger and questions are full
# sentences, giving more reliable similarity scores.
# Manuscript note: questions above STRICT are treated as training-derived;
# those between LENIENT and STRICT are partially derived.
T2_LIT_TRACEABILITY_STRICT_THRESHOLD  = 0.75
T2_LIT_TRACEABILITY_LENIENT_THRESHOLD = 0.70
T2_ASSUME_ALL_TRAINING_DERIVED        = True

# Optional quantile-based calibration of thresholds from the live corpus.
# When enabled, active thresholds are derived from similarity quantiles and
# constrained by min_lenient + min_gap (they can differ from defaults in
# either direction). This is the source of manuscript/table values such as
# tau_l=0.6164 and tau_s=0.6664 for the current Test 2 run.
T2_TRACEABILITY_CALIBRATION = {
    "enabled":          True,
    "strict_quantile":  0.90,
    "lenient_quantile": 0.75,
    "min_lenient":      0.60,  # floor: never push lenient below this
    "min_gap":          0.05,  # minimum gap between lenient and strict
}

# Visualisation: number of anchor questions shown in similarity plots
T2_N_TOP_SIM_ANCHORS    = 6
T2_N_BOTTOM_SIM_ANCHORS = 6

# OpenAlex corpus / cache settings (Test 2 needs a large corpus)
T2_OPENALEX_MAX_CACHE_GB = 1.0
T2_OPENALEX_TARGET_DOCS  = 4000
T2_OPENALEX_PER_QUERY    = 50
T2_OPENALEX_PER_PAGE     = 200


# ===========================================================================
# Test 3 — Theory Generation Depth
# ===========================================================================

# Primary novelty gate: cosine similarity to nearest *known* theory.
# Exceeding this threshold marks the AI theory as derivative.
# Set lower than T2_STRICT (0.75) because full paragraphs are compared to
# ~20 known consciousness theory texts, not a broad literature corpus, and
# embedding similarity for theory-level texts tends to be lower overall.
T3_DERIVATIVE_SIMILARITY_THRESHOLD = 0.30

# Strict traceability: used in result tables to align with section notebook.
# Lower than T2_STRICT for the same reason as above.
T3_STRICT_TRACEABILITY_THRESHOLD   = 0.70

# Continuous novelty score parameters
T3_SEMANTIC_NOVELTY_DENOM     = 0.40   # denominator when normalising raw novelty
T3_LITERATURE_NOVELTY_THRESHOLD = 0.50  # literature-based novelty rate threshold
T3_TYPE_FACTOR_FUNCTIONALIST  = 0.40   # multiplicative penalty for functionalist theories
T3_CHALLENGE_FACTOR_FALSE     = 0.60   # penalty when theory does not challenge functionalism

# Manifold / kNN novelty (supplementary analysis in 4b-test3-extras)
T3_K_NEIGHBORS          = 5     # kNN for manifold novelty estimation
T3_MANIFOLD_KNN_Q       = 0.95  # kNN distance percentile defining the manifold boundary
T3_SEMANTIC_DISTANCE_THRESHOLD = T3_DERIVATIVE_SIMILARITY_THRESHOLD  # alias

# Heatmap display parameters
T3_HEATMAP_TOP_N    = 15
T3_HEATMAP_BOTTOM_N = 15


# ===========================================================================
# Test 4 — Category Recognition and Philosophical Analysis
# ===========================================================================

# Scoring rubric thresholds (calibrated to the observed Test 4 dataset).
# Higher counts than the original draft thresholds prevent ceiling effects
# in the first two rubric dimensions.
T4_THRESHOLDS: dict = {
    "task_excerpt_char_limit":               700,
    "response_text_display_max_chars":       200,
    "parsed_string_control_char_upper_bound": 0x20,

    "distinction_categories_high": 8,
    "distinction_categories_mid":  6,
    "distinction_contested_min":   4,
    "distinction_missed_min":      4,

    "mistake_count_high":            4,
    "mistake_count_mid":             3,
    "mistake_missed_min":            5,
    "mistake_explanation_min_words": 8,
    "mistake_explanation_count_min": 2,

    "alternative_word_count_min": 80,
    "alternative_contested_min":  2,
    "alternative_mistakes_min":   1,

    # Mean cosine similarity of nuanced_analysis vs 7 reference arguments.
    # all-MiniLM-L6-v2: related prose ~0.40–0.55; unrelated ~0.20–0.32.
    # 0.40 is used as the operational threshold for short AI texts
    # compared to reference arguments.
    "traceability_similarity_threshold": 0.40,
}

T4_SCORE_LIMITS: dict = {
    "subscore_max": 5,
    "total_max":    15,
}

T4_MODEL_SCORE_MAXIMA: dict = {
    "distinction_score":    T4_SCORE_LIMITS["subscore_max"],
    "identification_score": T4_SCORE_LIMITS["subscore_max"],
    "alternative_score":    T4_SCORE_LIMITS["subscore_max"],
    "total_score":          T4_SCORE_LIMITS["total_max"],
}


# ===========================================================================
# Overview notebook (Section 5.1) defaults
# ===========================================================================

# Confidence interval level for bootstrap intervals in aggregate summaries.
OVERVIEW_CONFIDENCE_LEVEL = 0.95

# Alpha used in significance interpretation text.
OVERVIEW_SIGNIFICANCE_ALPHA = 0.05

# Shared cutoff for converting continuous novelty-like metrics into binary novelty.
OVERVIEW_NOVELTY_BINARY_THRESHOLD = 0.50

# Test 4 Alternative_Score threshold used as novelty-kind proxy in overview reporting.
OVERVIEW_TEST4_ALTERNATIVE_NOVELTY_THRESHOLD = 4.0

# Similarity alert reference shown in overview diagnostics.
OVERVIEW_SIMILARITY_ALERT_THRESHOLD = 0.80

# Test 3 similarity threshold used in overview-level traceability reconstruction.
OVERVIEW_T3_TRACEABILITY_SIMILARITY_THRESHOLD = 0.55
