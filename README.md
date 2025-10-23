# SAE Helix mRNA

**Interpretability Analysis of Helix-mRNA Foundation Model using Sparse Autoencoders**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

This project uses **Sparse Autoencoders (SAE)** to interpret the Helix-mRNA foundation model by discovering biologically meaningful features from internal representations.

**Key Features:** Non-invasive analysis via PyTorch hooks • Multi-layer support • Modular design • Comprehensive testing

---

## Installation

```bash
pip install torch numpy tqdm helical matplotlib seaborn scipy scikit-learn
```

---

## Analysis Pipeline

### **Step 1: SAE Training**
Train sparse autoencoders on Helix-mRNA activations to learn interpretable feature dictionaries.

```bash
python step1_SAE_analyse_db.py \
    --data_dir /path/to/data \
    --output_dir /path/to/output \
    --expansion_factor 4 \
    --l1_coefficient 1e-3
```

**What it does:**
- Loads mRNA sequences from JSON files
- Extracts activations from all transformer blocks
- Trains SAE models for each layer
- Saves sparse activation matrices and checkpoints

**Output:** SAE models, sparse activations, training history

---

### **Step 2: Correlation Analysis**
Identify which biological properties correlate with learned SAE features.

```bash
python step2_corr_db.py \
    --data_dir /path/to/data \
    --sparse_activation_dir /path/to/step1/output \
    --output_dir /path/to/step2/output \
    --num_samples 1000
```

**What it does:**
- Loads biological annotations (functional, structural, regulatory)
- Computes correlations between features and properties
- Applies statistical significance testing (FDR correction)
- Analyzes feature hierarchy across blocks
- Validates top features

**Methods:** Spearman correlation (continuous), Mann-Whitney U-test (discrete)

**Output:** Correlation matrices, significant associations, feature rankings

---

### **Step 3: Causal Feature Analysis (ACDC)**
Verify causal importance of features using intervention experiments.

```bash
python step3_causal_feature.py \
    --target_feature mrl \
    --data_dir /path/to/data \
    --step1_output_dir /path/to/step1/output \
    --output_dir /path/to/step3/output \
    --num_sequence_pairs 100 \
    --target_blocks 0 1 2 3
```

**What it does:**
1. Build prediction probe (regression/classification)
2. Construct sequence pairs with functional differences
3. Collect activations and sparse codes
4. Evaluate feature importance via ACDC algorithm
5. Generate cumulative intervention curves
6. Compare across blocks and baseline methods
7. Visualize results

**Output:** Feature importance rankings, intervention curves, causal validation

---

## Core Concepts

### Sparse Autoencoder (SAE)

```
Encode: c = ReLU(Wx + b)    # Sparse activations
Decode: x̂ = W^T c           # Reconstruction

Loss: L(x) = ||x - x̂||² + α||c||₁
```

**Key parameters:**
- `expansion_factor`: 2-8 (overcompleteness ratio)
- `l1_coefficient`: 1e-4 to 1e-2 (sparsity penalty)

---

## Project Structure

```
SAE_Helix_mRNA/
├── src/
│   ├── model/              # SAE model, activation extractor, trainer
│   ├── pipeline/           # SAE analysis and correlation pipelines
│   └── causal/             # ACDC causal analysis (10 modules)
├── step1_SAE_analyse_db.py # Step 1: Train SAE
├── step2_corr_db.py        # Step 2: Correlation analysis
├── step3_causal_feature.py # Step 3: Causal validation
├── test/                   # Unit and integration tests
└── examples/               # Usage examples
```

---

## Quick Example

```python
from src.pipeline import SAEAnalysisPipeline
from helical.models.helix_mrna import HelixmRNA, HelixmRNAConfig

# Initialize
model = HelixmRNA(configurer=HelixmRNAConfig(device='cuda'))
pipeline = SAEAnalysisPipeline(model, expansion_factor=4, l1_coefficient=1e-3)

# Run analysis
results = pipeline.run_full_analysis(
    dataset=model.process_data(sequences),
    layer_filter=lambda name, m: 'mixer' in name.lower(),
    num_epochs=100
)
```

---

## References

1. **Towards Monosemanticity** | Anthropic, 2023 | [arXiv:2309.08600](https://arxiv.org/abs/2309.08600)
2. **Sparse Autoencoders Find Highly Interpretable Features** | Cunningham et al., 2023

---