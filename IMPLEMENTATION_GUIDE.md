# Project Implementation Guide

## Overview

This document provides guidance for implementing the complete analysis described in the paper "Beyond Recombination: Structural Limits of Neural Networks in Ontological Creativity."

## File Structure

I've created the following files for you:

### 1. **dataset_generation.ipynb** ✓
Complete implementation of Section 4.2 multimodal sensory dataset generation.
- 8 sensory modality classes
- Scenario-based data generation
- 7,000 total examples (5K train, 1K val, 1K test)
- Ontology creation
- Visualization and statistics

### 2. **overview-analysis.ipynb** ✓
Aggregate analysis across all four tests (Section 5.1).
- Load and combine results from all tests
- Cross-test comparison statistics
- Confidence interval computation
- Overview visualizations
- Summary report generation

### 3. **test1_analysis.ipynb** ✓
Test 1: Ontological Innovation analysis (Section 5.2).
- Structural decomposition of proposals
- Literature traceability search
- Embedding space convex hull testing
- Multi-criteria classification
- Detailed example analysis

### 4-7. **Remaining Test Notebooks** (Need to be created)

You need to create:
- `test2_analysis.ipynb` - Epistemic Agency (questions analysis)
- `test3_analysis.ipynb` - Theory Generation (consciousness theories)
- `test4_analysis.ipynb` - Category Recognition (category mistakes)
- `test5_mechanistic-analysis.ipynb` - Mechanistic interpretability
- `test6_cross-model-analysis.ipynb` - Cross-model robustness

## Templates for Remaining Notebooks

I'll provide detailed templates below that follow the same structure as the completed notebooks.

---

## Section 5.3 Template: Epistemic Agency

```python
# Key components needed:

1. Question Classification System
   - Instrumental (optimizing existing)
   - Exploratory (within framework)
   - Paradigm-challenging (questioning framework)

2. Framework Transcendence Analysis
   - Extract concepts from questions
   - Build dependency graph
   - Determine if concepts exist in framework

3. Originality Assessment
   - Search training corpus for similar questions
   - Compute semantic similarity
   - Classify as novel vs. reproduced

4. Visualizations
   - Question type distribution by model
   - Framework dependency graphs
   - Similarity heatmaps
```

---

## Section 5.4 Template: Theory Generation

```python
# Key components needed:

1. Theoretical Framework Extraction
   - Parse theory text for ontological commitments
   - Identify explanatory structure (reductive/emergent/etc.)
   - Detect computational functionalism markers

2. Computational Functionalism Detection
   - Keywords: "computation", "information processing", "algorithm"
   - Check if consciousness = f(computation)
   - Classify: strong/weak/non-functionalist

3. Theory Traceability
   - Compare to known theories (IIT, GWT, HOT, etc.)
   - Measure similarity to training corpus
   - Identify novel vs. derivative proposals

4. Visualizations
   - Theory ontology graphs
   - Functionalism spectrum plots
   - Similarity dendrograms
```

---

## Section 5.5 Template: Category Recognition

```python
# Key components needed:

1. Category Awareness Scoring
   - Does response distinguish measurement vs. experience?
   - Does it identify ontological types?
   - Score 0-1 for category distinction recognition

2. Category Mistake Identification
   - Present scenarios with category confusions
   - Score ability to identify the mistake
   - Analyze explanation quality

3. Alternative Framework Proposals
   - Does response offer non-confused alternative?
   - Is alternative original or reproduced?

4. Standard vs. Novel Scenario Testing
   - Test on standard philosophy examples (Mary's Room, etc.)
   - Test on novel scenarios not in training
   - Compare performance degradation

5. Visualizations
   - Category awareness scores by model
   - Performance: standard vs. novel scenarios
   - Example response analysis
```

---

## Section 5.6 Template: Mechanistic Interpretability

```python
# Key components needed:

1. Attention Pattern Analysis
   - Extract attention weights during generation
   - Identify which training examples attended to
   - Visualize attention flow across layers

2. Activation Space Analysis
   - Extract hidden states during generation
   - Project to 2D/3D for visualization
   - Cluster and compare to training activations

3. Gradient-Based Attribution
   - Compute gradients of outputs w.r.t. inputs
   - Identify most influential input tokens
   - Link back to training data sources

4. Layer-wise Representation Tracking
   - Track how representations evolve through network
   - Identify where "novelty" emerges (if at all)
   - Test if late layers show training-independent patterns

5. Visualizations
   - Attention heatmaps
   - Activation space t-SNE/UMAP plots
   - Attribution bar charts
   - Layer evolution animations
```

---

## Section 5.7 Template: Cross-Model Robustness

```python
# Key components needed:

1. Statistical Cross-Model Comparison
   - ANOVA/Kruskal-Wallis tests across models
   - Effect size computation (Cohen's d, eta-squared)
   - Post-hoc pairwise comparisons

2. Temperature Variation Experiments
   - Test at T ∈ {0.0, 0.3, 0.5, 0.7, 1.0}
   - Measure novelty vs. temperature
   - Test if higher T produces genuine novelty

3. Prompt Robustness Testing
   - Vary prompt phrasing
   - Test different presentation formats
   - Measure consistency of limitations

4. Architecture Comparison
   - Transformer vs. other architectures
   - Parameter count effects
   - Training data size effects

5. Visualizations
   - Cross-model comparison forest plots
   - Temperature effect curves
   - Prompt robustness matrices
   - Architecture comparison tables
```

---

## Implementation Workflow

### Phase 1: Data Generation (Done)
Run `dataset_generation.ipynb`

### Phase 2: Collect AI Responses

For each test, you need to:

1. **Craft prompts** based on Section 4.3 specifications
2. **Query multiple AI models:**
   ```python
   import openai
   from anthropic import Anthropic
   import google.generativeai as genai
   
   # Example for Test 1
   prompt = """
   Given these 8 sensory modalities:
   [list modalities with specs]
   
   Propose a ninth sensory modality that is:
   - Genuinely new (not just extension of existing)
   - Not reducible to combinations of the 8
   - Has distinct physical basis and information content
   
   Explain: physical basis, information content, functional role
   """
   
   # Collect responses from all models
   responses = {
       'gpt-4': openai.chat.completions.create(...),
       'claude': anthropic.messages.create(...),
       'gemini': genai.generate_content(...)
   }
   ```

3. **Repeat 50 times** per model (with different seeds/temperatures)

4. **Save responses** in structured format:
   ```json
   {
       "test": "test1_ontological_innovation",
       "model": "gpt-4",
       "sample_id": 1,
       "temperature": 0.7,
       "prompt": "...",
       "response": "...",
       "timestamp": "2026-02-13T..."
   }
   ```

### Phase 3: Run Analysis Notebooks

1. Run test-specific notebooks (5.2-5.5)
2. Run mechanistic analysis (5.6)  
3. Run cross-model comparison (5.7)
4. Run overview aggregation (5.1)

### Phase 4: Insert Results into Paper

Extract key findings from notebooks:
- Quantitative statistics
- Confidence intervals
- Example quotes from AI responses
- Visualization figures

---

## Quick Start Commands

```bash
# 1. Setup environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Generate dataset
jupyter notebook dataset_generation.ipynb
# Run all cells → creates data/ directory

# 3. Collect AI responses (write your own script)
python research/data/scripts/collect_ai_responses.py --test all --models gpt-4,claude-3.5,gemini-1.5

# 4. Run analysis notebooks
jupyter notebook test1_analysis.ipynb
# Repeat for 5.3, 5.4, 5.5, 5.6, 5.7

# 5. Generate overview
jupyter notebook overview-analysis.ipynb
```

---

## Key Algorithms Reference

### Convex Hull Testing
```python
from scipy.spatial import Delaunay

def is_inside_hull(points_train, points_test):
    hull = Delaunay(points_train)
    return hull.find_simplex(points_test) >= 0
```

### Bootstrap Confidence Intervals
```python
def bootstrap_ci(data, n_bootstrap=10000, ci=0.95):
    means = [np.mean(np.random.choice(data, len(data))) 
             for _ in range(n_bootstrap)]
    alpha = 1 - ci
    return np.percentile(means, [alpha/2*100, (1-alpha/2)*100])
```

### Literature Similarity
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_similarity(text1, text2):
    emb1, emb2 = model.encode([text1, text2])
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
```

---

## Expected Results Pattern

Based on the paper's theoretical predictions, you should find:

- **Test 1:** ~0-5% genuinely novel, ~90-95% traceable/decomposable
- **Test 2:** ~80-90% instrumental/exploratory, ~5-15% paradigm-challenging (but these trace to literature)
- **Test 3:** ~90-95% computational functionalism, ~0-5% genuinely novel frameworks
- **Test 4:** High verbal awareness (~80%), low genuine comprehension on novel scenarios (~20-30%)
- **Mechanistic:** Attention concentrated on relevant training examples
- **Cross-model:** High consistency across models (supporting structural constraint hypothesis)

---

## Tips for Success

1. **Start with dataset generation** - Foundation for everything
2. **Test prompts carefully** - Bad prompts = bad data
3. **Use multiple models** - Cross-model consistency is key finding
4. **Save intermediate results** - API calls are expensive
5. **Document everything** - Reproducibility matters
6. **Visualize early and often** - Helps catch errors
7. **Use version control** - Git track your notebooks
8. **Write helper functions** - Don't repeat code across notebooks

---

## Troubleshooting

**Problem:** Dataset generation is slow  
**Solution:** Reduce sample size or parallelize

**Problem:** API rate limits  
**Solution:** Add delays, use async requests, or spread over time

**Problem:** Memory issues with embeddings  
**Solution:** Process in batches, use smaller models, or use CPU

**Problem:** Results don't match predictions  
**Solution:** Check prompt wording, verify evaluation logic, examine edge cases

---

## Next Steps

1. Create the remaining 5 notebooks using templates above
2. Write `collect_ai_responses.py` script for automated data collection
3. Run complete pipeline
4. Extract results for paper Section 5
5. Create supplementary materials with detailed examples

Good luck with your implementation! The foundation is solid - now build on it.
