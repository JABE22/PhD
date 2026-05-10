# PhD-Research
PhD. Dissertation Research Studies

# Empirical Studies in Ontological Creativity of AI: Analysis Notebooks

This repository contains the complete computational notebooks for conducting the empirical investigations described in **"Structural Limits of Neural Networks in Ontological Creativity"**.

## Overview

The notebooks implement the four empirical tests described in Section 4 and generate the results and visualizations for Section 5 of the paper.

## Repository Structure

```
├── research/data/notebooks/dataset-generation.ipynb          # Generate multimodal sensory dataset (Section 4.2)
├── research/results/overview/overview-analysis.ipynb        # Aggregate analysis across all tests (Section 5.1)
├── test1_analysis.ipynb           # Test 1: Ontological Innovation (Section 5.2)
├── test2_analysis.ipynb           # Test 2: Epistemic Agency (Section 5.3)
├── test3_analysis.ipynb           # Test 3: Theory Generation (Section 5.4)
├── test4_analysis.ipynb           # Test 4: Category Recognition (Section 5.5)
├── test5_mechanistic-analysis.ipynb     # Mechanistic interpretability analysis (Section 5.6)
├── test6_cross-model-analysis.ipynb     # Cross-model comparison and robustness (Section 5.7)
├── data/                             # Generated datasets
├── results/                          # Analysis outputs and figures
└── README.md                         # This file
```

## Requirements

### Python Environment

```bash
python >= 3.8
numpy >= 1.20
pandas >= 1.3
matplotlib >= 3.4
seaborn >= 0.11
scikit-learn >= 0.24
scipy >= 1.7
sentence-transformers >= 2.0
transformers >= 4.20
torch >= 1.10
networkx >= 2.6
jupyter >= 1.0
```

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# For GPU support (optional but recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### API Keys (Required for AI Model Testing)

Create a `.env` file with your API keys:

```
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
```

## Usage

### Step 1: Generate Dataset

```bash
jupyter notebook research/data/notebooks/dataset-generation.ipynb
```

This creates the 8-dimensional multimodal sensory dataset:
- 5,000 training examples
- 1,000 validation examples  
- 1,000 test examples
- Complete ontology in `data/ontology.json`

**Expected runtime:** 15-30 minutes

**Output:** `data/train/`, `data/val/`, `data/test/` directories with JSON files

### Step 2: Run Empirical Tests

Each test notebook can be run independently after dataset generation:

#### Test 1: Ontological Innovation
```bash
jupyter notebook test1_analysis.ipynb
```
Tests whether AI proposes genuinely novel sensory modalities.

**Methods:**
- Structural decomposition analysis
- Literature traceability search
- Embedding space convex hull testing
- Multi-criteria classification

**Output:** `results/test1/` with classifications, visualizations, and detailed examples

#### Test 2: Epistemic Agency
```bash
jupyter notebook test2_analysis.ipynb
```
Analyzes AI-generated research questions for paradigm-challenging capacity.

**Methods:**
- Question type classification (instrumental/exploratory/paradigm-challenging)
- Framework transcendence analysis
- Originality assessment via literature comparison
- Dependency graph construction

**Output:** `results/test2/` with question classifications and framework analysis

#### Test 3: Theory Generation
```bash
jupyter notebook test3_analysis.ipynb
```
Examines consciousness theories for computational functionalism vs. novel frameworks.

**Methods:**
- Ontological commitment extraction
- Explanatory structure analysis
- Computational functionalism detection
- Theory traceability to known frameworks

**Output:** `results/test3/` with theory classifications and commitment graphs

#### Test 4: Category Recognition
```bash
jupyter notebook test4_analysis.ipynb
```
Tests recognition of category mistakes and ontological boundaries.

**Methods:**
- Category distinction scoring
- Category mistake identification
- Alternative framework proposal analysis
- Standard vs. novel scenario comparison

**Output:** `results/test4/` with category awareness metrics

### Step 3: Collect Activations For Mechanistic Analysis

Before running the mechanistic notebook, collect activations from a local
open-source model using the prompts already exported in `ai_responses/`.

```bash
python research/data/scripts/collect_activations.py \
    --model-name mistralai/Mistral-7B-v0.1 \
    --per-test-limit 4 \
    --max-new-tokens 64 \
    --save-attentions
```

This writes one `.pt` artifact per prompt into `data/activations/` with prompt
metadata, hidden states, attention tensors, and the generated continuation.

### Step 4: Mechanistic Interpretability Analysis

```bash
jupyter notebook test5_mechanistic-analysis.ipynb
```

Probes internal representations to understand generation mechanisms.

**Methods:**
- Attention pattern analysis
- Activation space clustering
- Gradient-based attribution
- Layer-wise representation tracking

**Output:** `results/mechanistic/` with attention visualizations and attribution maps

### Step 5: Cross-Model Comparison

```bash
jupyter notebook test6_cross-model-analysis.ipynb
```

Evaluates robustness across models and parameters.

**Methods:**
- Statistical comparison tests
- Temperature variation analysis
- Prompt robustness testing
- Architecture comparison

**Output:** `results/cross_model/` with comparative statistics and robustness plots

### Step 6: Generate Overview Report

```bash
jupyter notebook research/results/overview/overview-analysis.ipynb
```

Aggregates results from all tests into comprehensive overview.

**Output:** `results/overview_report.json` and summary figures

## Data Format

### Multimodal Sensory Dataset

Each example is a JSON file with structure:

```json
{
  "scenario": "walking",
  "timesteps": 60,
  "modalities": {
    "visual": [[...]],      // 60 x 140
    "auditory": [[...]],    // 60 x 128
    "tactile": [[...]],     // 60 x 32
    "olfactory": [[...]],   // 60 x 10
    "gustatory": [[...]],   // 60 x 5
    "proprioceptive": [[...]], // 60 x 30
    "vestibular": [[...]],  // 60 x 6
    "interoceptive": [[...]] // 60 x 8
  }
}
```

**Total dimensions:** 359 per timestep, 21,540 per example

### Ontology Format

The `data/ontology.json` file provides machine-readable ontology:

```json
{
  "modalities": [
    {
      "name": "visual",
      "dimensions": 140,
      "physical_basis": "Electromagnetic radiation",
      "range": "300-1000nm",
      "sampling": "5nm intervals"
    },
    ...
  ],
  "total_dimensions": 359
}
```

## Key Functions and Algorithms

### Structural Decomposition (Test 1)

```python
def structural_decomposition(proposal: dict) -> dict:
    """
    Analyze if proposed modality decomposes into existing 8 modalities.
    
    Returns:
        - is_range_extension: bool
        - is_hybrid: bool
        - base_modalities: list
        - novelty_score: float [0,1]
    """
```

### Convex Hull Membership Test

```python
def test_convex_hull_membership(existing_embeddings, new_embeddings):
    """
    Test if new concepts lie within convex hull of training concepts.
    Uses Delaunay triangulation for efficient membership testing.
    
    Returns:
        - inside: bool array
        - hull: ConvexHull object
    """
```

### Literature Traceability

```python
def literature_search(text: str, corpus: list, threshold: float = 0.75):
    """
    Search training corpus using:
    - BM25 keyword matching
    - Sentence-BERT embedding similarity
    - Structural pattern matching
    
    Returns:
        - max_similarity: float
        - matching_sources: list
        - is_traceable: bool
    """
```

### Framework Transcendence Analysis (Test 2)

```python
def analyze_framework_transcendence(question: str, framework: dict):
    """
    Determine if answering question requires resources beyond framework.
    
    Methods:
    - Concept extraction
    - Dependency graph construction  
    - Framework boundary detection
    
    Returns:
        - transcendence_score: float
        - required_concepts: list
        - is_paradigm_challenging: bool
    """
```

## Interpreting Results

### Classification Categories (Test 1)

- **Genuinely Novel:** Passes all tests (rare, expected ~0-2%)
- **Range Extension:** Extends existing modality parameters
- **Hybrid/Combination:** Combines multiple existing modalities
- **Literature Traceable:** Matches known concepts from training
- **Recombination:** Novel combination within training space

### Confidence Intervals

All proportions reported with 95% bootstrap confidence intervals:

```python
# Example output
Novel outputs: 3.2% [2.1%, 4.5%]
Traceable: 94.8% [92.3%, 96.7%]
```

### Statistical Tests

- **Chi-square:** Testing independence between models
- **Mann-Whitney U:** Non-parametric comparison of distributions
- **Kruskal-Wallis:** Multi-group comparison
- **Bootstrap:** Confidence interval estimation

## Troubleshooting

### Common Issues

**1. Memory Error During Dataset Generation**

```python
# Reduce batch size or generate in chunks
generator.generate_dataset(5000, save_every=500)
```

**2. API Rate Limiting**

```python
# Add delay between requests
import time
for model in models:
    response = call_api(model, prompt)
    time.sleep(1)  # 1 second delay
```

**3. Missing Dependencies**

```bash
# Install specific version
pip install sentence-transformers==2.2.2
```

**4. CUDA Out of Memory**

```python
# Use CPU for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
```

## Customization

### Adding New Modalities to Test

Edit `research/data/notebooks/dataset-generation.ipynb`:

```python
class CustomModality(SensoryModality):
    def __init__(self):
        super().__init__('custom_name', dimensions=50)
    
    def generate(self, scenario: str, timesteps: int):
        # Implement generation logic
        return data
```

### Modifying Evaluation Criteria

Edit classification functions in test notebooks:

```python
def custom_classification(row):
    # Custom logic for determining novelty
    if custom_condition(row):
        return 'Custom Category'
    return 'Standard Category'
```

### Adding New AI Models

```python
models_to_test = [
    'gpt-4',
    'claude-3.5-sonnet',
    'gemini-1.5-pro',
    'llama-3-70b',
    'your-custom-model'  # Add here
]
```

## Citation

If you use these notebooks in your research, please cite:

```bibtex
@article{ai_creativity,
  title={Empirical Studies: Structural Limits of Information Processing Systems in Ontological Creativity},
  author={Matarmaa, J.O.},
  journal={Ural Federal University},
  year={2026},
  address={Yekaterinburg}
}
```

## License

MIT License - see LICENSE file for details

## Contact

For questions or issues:
- Open an issue on GitHub
- Email: iarnoolavi.matarmaa@urfu.me

## Acknowledgments

- Dataset generation inspired by biological and machine sensing capabilities
- Evaluation methods adapted from computational creativity literature
- Statistical approaches follow standard practices in empirical AI research

## Version History

- **v1.0 (2026-02-13):** Initial release with all 7 notebooks
- Includes complete dataset generation and all four empirical tests
- Mechanistic interpretability and cross-model analysis included

## Future Extensions

Planned additions:
- [ ] Additional AI models (GPT-5, Claude 4, etc.)
- [ ] Extended modality set (>8 modalities)
- [ ] Interactive visualization dashboard
- [ ] Real-time data collection from actual sensors
- [ ] Longitudinal tracking of AI capabilities over time
