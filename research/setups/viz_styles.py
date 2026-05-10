"""
Global visualization styles for the AI Creativity / Consciousness project.
Includes:
- Custom colormaps for heatmaps and density maps
- A 6-step blue-to-gray color palette for categorical distinctions (e.g., traceability bands)
- A 9-color palette for multi-model comparisons (ordered by visual distinctiveness)
- Specific color mappings for Test-2 epistemic classes and category scheme

Typography and font sizing are intentionally configured only via
`setups/plotstyles.mplstyle` to keep style control centralized.
"""

from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Colormaps
# ---------------------------------------------------------------------------

# White → deep-blue (for similarity heatmaps, density maps)
WHITE_SKY_CMAP = LinearSegmentedColormap.from_list(
    "white_sky", ["#ffffff", "#0057e7"]
)

# Sky-blue → white → mid-gray (diverging; below/above a central value)
SKY_WHITE_GRAY_CMAP = LinearSegmentedColormap.from_list(
    "skywhitegray", ["#1e90ff", "#ffffff", "#808080"]
)

# Very-light-gray → deep-blue (density maps without a white midpoint)
DENSITY_GRAY_BLUE_CMAP = LinearSegmentedColormap.from_list(
    "density_gray_blue", ["#ececec", "#0057e7"]
)


# ---------------------------------------------------------------------------
# Color palette — 6-step blue-to-gray ramp
#
# Semantic: index 0 = darkest (most familiar / highly traced)
#           index 5 = lightest (most novel / lowest traceability)
# ---------------------------------------------------------------------------

BLUE_GRAY_RAMP = [
    "#0b3c8c",  # 0 – dark navy
    "#2f6fc2",  # 1 – mid blue
    "#79aeea",  # 2 – light blue
    "#9a9a9a",  # 3 – mid gray
    "#c7c7c7",  # 4 – light gray
    "#efefef",  # 5 – near-white
]

BLUE_GRAY_CMAP = LinearSegmentedColormap.from_list(
    "blue_gray_ramp", BLUE_GRAY_RAMP
)

# Single-purpose color constants
SIMILARITY_HIST_COLOR = "#79aeea"   # histogram bar fill
THRESHOLD_COLOR       = "#5f5f5f"   # threshold / reference lines
TEXT_COLOR            = "#000000"   # general text and annotations
ANCHOR_COLOR          = "#111111"   # scatter anchor / reference markers

# Multi-model line/bar palette: 9 distinguishable blue/gray tones
# Ordered so the first models are most visually distinct (dark blues first,
# then grays for additional models)
MODEL_VISIBLE_COLORS = [
    "#0b3c8c",
    "#1f5fb3",
    "#2f6fc2",
    "#4f86cf",
    "#6d9edc",
    "#6f6f6f",
    "#8a8a8a",
    "#a8a8a8",
    "#c7c7c7",
]


# ---------------------------------------------------------------------------
# Test-2 specific — question-class colors
# ---------------------------------------------------------------------------

# Three epistemic classes used in Test 2 (Epistemic Agency)
CLASS_COLORS = {
    "instrumental":         "#ececec",  # light gray – routine optimisation
    "exploratory":          "#1e90ff",  # sky blue   – within-framework exploration
    "paradigm_challenging": "#808080",  # mid gray   – framework-transcending
}


# ---------------------------------------------------------------------------
# Test-2 specific — two-axis category scheme
# (framework transcendence × literature traceability band)
# ---------------------------------------------------------------------------

CATEGORY_COLORS = {
    "within_framework_high_traceability":   "#0b3c8c",  # darkest blue
    "within_framework_medium_traceability": "#2f6fc2",
    "within_framework_low_traceability":    "#79aeea",
    "transcendent_high_traceability":       "#9a9a9a",
    "transcendent_medium_traceability":     "#c7c7c7",
    "transcendent_low_traceability":        "#efefef",  # near-white
}

# Ordered from most-constrained (known + within-framework) to most-novel
CATEGORY_ORDER = list(CATEGORY_COLORS.keys())

CATEGORY_CMAP = LinearSegmentedColormap.from_list(
    "t2_category_ramp", [CATEGORY_COLORS[key] for key in CATEGORY_ORDER]
)

MODEL_VISIBLE_CMAP = LinearSegmentedColormap.from_list(
    "model_visible_ramp", MODEL_VISIBLE_COLORS
)


# Shared marker standards
MARKER_SIZE_L = 100
MARKER_SIZE_M = 70
MARKER_SIZE_S = 30
MARKER_EDGE_WIDTH = 0.8
MARKER_EDGE_WIDTH_STANDARD = 0.6
