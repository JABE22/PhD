# Project Summary

## Project Overview

This document introduces the repository structure, analysis workflow, and main outputs for third parties who may later use, reproduce, or extend the project. The repository contains the implemented workflow for the empirical studies reported in dissertation chapter 4.

### 1. Core Data Generation ✓
**File:** `research/data/notebooks/dataset-generation.ipynb`
- Complete implementation of 8-modality sensory dataset
- Generates 7,000 examples (5K train, 1K val, 1K test)
- 359 dimensions per timestep, 60 timesteps per example
- Includes visualization and statistics
- Creates machine-readable ontology

### 2. Overview Analysis ✓
**File:** `research/results/overview/overview-analysis.ipynb`
- Aggregates results from all 4 tests
- Computes cross-test statistics
- Bootstrap confidence intervals
- Cross-model comparisons
- Generates summary visualizations

### 3. Test 1: Ontological Innovation ✓
**File:** `research/test1_ontological-innovation/test1_analysis.ipynb`
- Structural decomposition algorithm
- Literature traceability search
- Convex hull membership testing
- Multi-criteria classification
- Embedding space visualization

### 4. Supporting Infrastructure ✓
**Files:**
- `requirements.txt` - All Python dependencies
- `README.md` - Complete documentation
- `research/data/scripts/collect_ai_responses.py` - Automated AI query script
- `research/setups/` - Shared project utilities and analysis helpers

## Analysis Notebook Coverage

All analysis notebooks referenced in this workflow are already present in the repository:

1. **research/test2_epistemic-agency/test2_analysis.ipynb** - Epistemic Agency
2. **research/test3_theory-generation/test3_analysis.ipynb** - Theory Generation
3. **research/test4_category-recognition/test4_analysis.ipynb** - Category Recognition
4. **research/test5/test5_mechanistic-analysis.ipynb** - Mechanistic Interpretability
5. **research/test6/test6_cross-model-analysis.ipynb** - Cross-Model Robustness

## Usage Workflow

### Step 1: Environment Setup
```bash
conda activate aitrust
pip install -r requirements.txt
```

### Step 2: Create .env File
```bash
# Create .env with your API keys
OPENROUTER2_API_KEY=your_key_here
PERPLEXITY_API_KEY=your_key_here
MISTRALAI_API_KEY=your_key_here
```

### Step 3: Generate Dataset
```bash
jupyter notebook research/data/notebooks/dataset-generation.ipynb
# Run all cells → creates data/ directory with 7,000 examples
```

### Step 4: Collect AI Responses
```bash
python research/data/scripts/collect_ai_responses.py --provider openrouter --list-models
python research/data/scripts/collect_ai_responses.py --provider openrouter --test all --models gpt-5.2,claude-3.7-sonnet,gemini-3.1-pro-preview --n-samples 50
# This queries AI models and saves responses to research/test*/ai_responses/ directory
```

### Step 5: Run Analysis Notebooks
Execute the existing notebooks for each test and analysis stage:
- research/test1_ontological-innovation/test1_analysis.ipynb
- research/test2_epistemic-agency/test2_analysis.ipynb
- research/test3_theory-generation/test3_analysis.ipynb
- research/test4_category-recognition/test4_analysis.ipynb
- research/test5/test5_mechanistic-analysis.ipynb
- research/test6/test6_cross-model-analysis.ipynb

### Step 6: Run Complete Analysis
```bash
# Run test-specific analyses
jupyter notebook research/test1_ontological-innovation/test1_analysis.ipynb
jupyter notebook research/test2_epistemic-agency/test2_analysis.ipynb
jupyter notebook research/test3_theory-generation/test3_analysis.ipynb
jupyter notebook research/test4_category-recognition/test4_analysis.ipynb

# Run mechanistic and cross-model analyses
jupyter notebook research/test5/test5_mechanistic-analysis.ipynb
jupyter notebook research/test6/test6_cross-model-analysis.ipynb

# Generate overview
jupyter notebook research/results/overview/overview-analysis.ipynb
```

### Step 7: Extract Results for Chapter 4
From each notebook, extract:
- Quantitative statistics (proportions, means, CIs)
- Representative examples (quoted AI responses)
- Visualization figures (save as high-res PNGs)
- Key findings text

## Key Algorithms Provided

### 1. Structural Decomposition
Tests if proposed concept decomposes into training primitives:
```python
def structural_decomposition(proposal: dict) -> dict:
    # Returns: is_range_extension, is_hybrid, novelty_score
```

### 2. Convex Hull Testing
Tests if embeddings lie within training data hull:
```python
def test_convex_hull_membership(existing, new):
    # Returns: inside (bool array), hull object
```

### 3. Literature Traceability
Searches corpus for similar concepts:
```python
def literature_search(text, corpus, threshold=0.75):
    # Returns: max_similarity, matching_sources, is_traceable
```

### 4. Bootstrap Confidence Intervals
Computes robust CIs for proportions:
```python
def bootstrap_ci(data, n_bootstrap=10000, ci=0.95):
    # Returns: (lower, upper)
```

## Expected Output Structure

```
project/
├── research/data/
│   ├── train/           # 5,000 training examples
│   ├── val/             # 1,000 validation examples
│   ├── test/            # 1,000 test examples
│   ├── ontology.json    # Machine-readable ontology
│   └── dataset_summary.json
│
├── research/test*/ai_responses/
│   ├── test1_ontological_innovation/
│   │   ├── prompt.txt
│   │   ├── gpt-4_sample_000.json
│   │   ├── claude-3.5_sample_000.json
│   │   └── all_responses.json
│   ├── test2_epistemic_agency/
│   ├── test3_theory_generation/
│   └── test4_category_recognition/
│
├── research/results/
│   ├── test1/
│   │   ├── detailed_results.csv
│   │   ├── embedding_space_analysis.png
│   │   └── classification_summary.png
│   ├── test2/
│   ├── test3/
│   ├── test4/
│   ├── mechanistic/
│   ├── cross_model/
│   ├── figures/
│   │   └── overview_findings.png
│   └── overview_report.json
│
└── [all the notebooks and scripts I created]
```

## What Each File Does

### research/data/notebooks/dataset-generation.ipynb
- **Input:** Nothing (generates synthetic data)
- **Output:** 7,000 JSON files in data/ with sensory recordings
- **Runtime:** ~15-30 minutes
- **Purpose:** Creates controlled conceptual universe for testing

### research/data/scripts/collect_ai_responses.py
- **Input:** Prompts (defined in script), API keys (.env)
- **Output:** JSON files with AI responses
- **Runtime:** ~1-2 hours (with rate limiting)
- **Purpose:** Automates querying multiple AI models

### research/test1_ontological-innovation/test1_analysis.ipynb (and similar for tests 2-4)
- **Input:** AI responses from research/test*/ai_responses/
- **Output:** Classified results, statistics, visualizations
- **Runtime:** ~5-15 minutes per test
- **Purpose:** Analyzes whether AI outputs are genuinely novel

### research/test5/test5_mechanistic-analysis.ipynb
- **Input:** AI model internals (requires model access)
- **Output:** Attention maps, activation plots, attribution scores
- **Runtime:** ~30-60 minutes (depending on model size)
- **Purpose:** Understands internal generation mechanisms

### research/test6/test6_cross-model-analysis.ipynb
- **Input:** Results from all tests
- **Output:** Cross-model statistics, robustness analyses
- **Runtime:** ~10-20 minutes
- **Purpose:** Tests whether limitations generalize across models

### research/results/overview/overview-analysis.ipynb
- **Input:** Results from all other analyses
- **Output:** Aggregate statistics, summary report, overview figures
- **Runtime:** ~5 minutes
- **Purpose:** Creates chapter-ready summary of all findings

## Key Metrics to Report in Chapter 4

The following outputs are especially relevant for chapter 4 reporting:

1. **Proportion Novel** (with 95% CI)
   - Expected: 0-5%
   
2. **Proportion Traceable** (with 95% CI)
   - Expected: 90-95%
   
3. **Classification Distribution**
   - Range extensions: ~40-50%
   - Hybrids: ~20-30%
   - Literature traceable: ~20-30%
   - Genuinely novel: ~0-5%

4. **Cross-Model Consistency**
   - Chi-square test p-value
   - Effect sizes (Cramér's V)

5. **Representative Examples**
   - 2-3 detailed examples per category
   - Quoted AI responses
   - Analysis showing decomposition/traceability

## Tips for Success

### Do's ✓
- Start with dataset generation - it's the foundation
- Save intermediate results - API calls are expensive
- Document everything for reproducibility
- Use version control (git)
- Test with small samples first (n=5) before full run (n=50)
- Check visualizations early to catch errors

### Don'ts ✗
- Don't skip the data generation step
- Don't run without API keys properly configured
- Respect provider rate limits to avoid request throttling or temporary suspension
- Don't modify datasets between test runs (breaks reproducibility)
- Don't skip documenting parameter choices

## Troubleshooting Common Issues

### "API rate limit exceeded"
→ Add longer delays in `collect_ai_responses.py`
→ Or spread collection over multiple days

### "Out of memory"
→ Process data in smaller batches
→ Use CPU instead of GPU for embeddings
→ Reduce dataset size for testing

### "Results don't match predictions"
→ Check prompt wording carefully
→ Verify evaluation thresholds
→ Examine edge cases manually

### "Can't reproduce results"
→ Set random seeds in all notebooks
→ Document all parameter choices
→ Save API response timestamps

## Recommended Workflow

1. **Immediate (1-2 hours):**
   - Review all created files
   - Setup Python environment
   - Test dataset generation on small sample

2. **Short term (1-2 days):**
   - Test AI response collection with n=5 samples
   - Run the test-specific notebooks end-to-end
   - Verify the analysis pipeline works end-to-end

3. **Medium term (1-2 weeks):**
   - Collect full AI responses (n=50 per model)
   - Run all analyses
   - Generate figures and results

4. **Chapter 4 integration (1 week):**
   - Extract quantitative results for chapter 4
   - Select representative examples
   - Create publication-quality figures
   - Write up findings

## Documentation Sources

For algorithm details, workflow guidance, debugging, and result interpretation, use the following sources:

Refer to:
1. `README.md` - General documentation
2. Notebook markdown cells - Workflow and interpretation notes
3. Inline comments in notebooks - Implementation details
4. The dissertation discussion framing chapter 4

## Project Status

This is a complete, production-ready framework. All the hard work of:
- Dataset design and generation ✓
- Evaluation algorithm implementation ✓
- Statistical analysis methods ✓
- Visualization code ✓
- Documentation ✓

The current workflow supports:
1. Running the complete pipeline
2. Reviewing the generated outputs
3. Extracting results for chapter 4

The repository is structured for reproducible execution, later reuse, and further methodological extension.
