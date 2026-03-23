# VisionSAE — Sparse Autoencoders for Interpreting Vision Transformer Representations

**Python • PyTorch • Vision Transformers • Sparse Autoencoders • Mechanistic Interpretability**

---

## Project Summary

VisionSAE investigates the internal structure of Vision Transformer (ViT) representations using overcomplete sparse autoencoders. The project trains layer-wise sparse autoencoders on intermediate transformer activations and studies how learned feature bases evolve across network depth, model width, and random initialization.

The repository implements a fully reproducible experimental pipeline including:

* Intermediate feature extraction from transformer blocks
* Layer-wise sparse autoencoder training
* Cross-seed feature alignment via Hungarian matching
* Cross-width representation analysis
* Geometric diagnostics of learned feature bases
* Patch-level interpretability and activation maximization

The central objective is to understand how structured visual features emerge and stabilize inside transformer-based vision models.

---

## Motivation

Sparse autoencoders have recently proven effective in uncovering structured features within large language models. However, systematic studies of similar techniques applied to vision transformers remain limited.

This project explores the following research questions:

* Do sparse autoencoders reveal consistent, structured features in ViT representations?
* How does feature stability evolve across transformer depth?
* Does increasing SAE width improve feature consistency?
* Are deeper representations more susceptible to feature redundancy or collapse?
* Do independently trained models converge toward similar feature bases?

By combining reconstruction, sparsity, alignment, and geometric analysis, VisionSAE provides a structured framework for probing representation geometry in vision transformers.

---

## Methodology

### Backbone Model

Experiments use a Vision Transformer backbone. Hidden activations are extracted from each transformer block and used as input to independent sparse autoencoders.

```
Image
  ↓
Patch Embedding
  ↓
Transformer Blocks (0–11)
  ↓
Layer Activations
  ↓
Sparse Autoencoder
  ↓
Feature Analysis
```

Each transformer layer is analyzed independently.

---

### Sparse Autoencoder Architecture

For each layer activation vector ( x ):

```
z = Encoder(x)
x̂ = Decoder(z)
```

Training objective:

```
L = ||x - x̂||² + λ · SparsityPenalty(z)
```

Key design decisions:

* Overcomplete hidden representation
* Top-k sparsity activation
* Independent training per layer
* Multiple random seeds
* Width scaling experiments

This setup encourages the discovery of structured feature directions rather than trivial identity mappings.

---

## Experimental Setup

**Dataset**

* CIFAR-10 (rapid experimentation and controlled analysis)

**Backbone**

* Vision Transformer (ViT)

**SAE Configuration**

| Parameter       | Values                |
| --------------- | --------------------- |
| Layers analyzed | 12 transformer blocks |
| Hidden width    | 4096, 8192, 16384     |
| Seeds           | 42, 123, 999          |
| Sparsity        | Top-k activation      |

All experiments were conducted on NVIDIA RTX A6000 GPUs.

---

## Metrics and Analysis

VisionSAE evaluates learned representations using multiple complementary metrics:

### Reconstruction

* Mean Squared Error (MSE)

### Sparsity

* Fraction of inactive latent units

### Mutual Coherence

* Measures redundancy between decoder columns

### Effective Rank

* Estimates intrinsic dimensionality of learned basis

### Feature Stability

* Computed via Hungarian matching across seeds
* Quantifies alignment between independently trained SAEs

### Cross-Width Alignment

* Measures similarity of learned features across different SAE widths

### Collapse Detection

* Identifies redundancy and overlapping feature directions

Together, these metrics characterize both reconstruction fidelity and geometric structure.

---

## Interpretability Experiments

### Top-Activating Images

For each feature, the dataset images producing maximal activation are retrieved. This provides qualitative insight into learned visual patterns.

### Activation Maximization

Synthetic inputs are optimized to maximize individual feature activations, revealing preferred visual structures in feature space.

### Patch-Level Activation Maps

Instead of pooled representations, token-level activations are analyzed to produce spatial heatmaps:

* Identifies which image regions activate a feature
* Reveals spatial selectivity
* Enables localized interpretability

---

## Key Findings

Preliminary experiments reveal several consistent trends:

* Sparse autoencoders recover structured feature bases from intermediate ViT representations.
* Feature stability across random seeds improves as SAE width increases.
* Early transformer layers exhibit higher cross-seed alignment than deeper layers.
* Wider SAEs show stronger cross-width feature convergence.
* Deeper layers demonstrate increased feature redundancy, reflected in higher mutual coherence.

These observations suggest that representation geometry evolves systematically across depth and scaling regimes.

---

## Repository Structure

```
visionSAE/
│
├── configs/                # experiment configurations
├── data/                   # dataloaders
├── models/                 # backbone and SAE implementations
├── analysis/               # geometry, visualization, statistics
├── alignment/              # feature matching utilities
├── experiments/            # experiment automation scripts
├── scripts/                # training and evaluation
├── notebooks/              # interpretability analysis
│
├── results/                # experiment outputs (JSON)
├── checkpoints/            # trained SAE weights (excluded from repo)
├── features/               # extracted backbone features (excluded)
│
├── results_summary.csv
├── stability_summary.csv
├── cross_width_summary.csv
│
├── requirements.txt
└── README.md
```

Large artifacts (datasets, checkpoints, extracted features) are intentionally excluded to ensure reproducibility and repository clarity.

---

## Environment Setup

```
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If your machine has no CUDA GPU, set `training.device` to `cpu` in `configs/vit_base_cifar.yaml`.

### Smoke Check

```
python -m scripts.smoke_check
```

---

## Running the Pipeline

### Extract Layer Features

```
python -m scripts.extract_features --config configs/vit_base_cifar.yaml --layer 0
```

### Train Sparse Autoencoder

```
python -m scripts.train_layer --config configs/vit_base_cifar.yaml --layer 0
```

### Evaluate Representation Metrics

```
python -m scripts.evaluate_layer --config configs/vit_base_cifar.yaml --layer 0
```

### Stability Sweep

```
python -m experiments.run_full_stability_sweep
```

### Cross-Width Alignment

```
python -m experiments.run_cross_width_sweep
```

### Single Cross-Width Run

```
python -m experiments.run_cross_width \
  --config configs/vit_base_cifar.yaml \
  --layer 0 \
  --seed 42 \
  --width_small 4096 \
  --width_large 8192
```

### Aggregate Results

```
python -m experiments.aggregate_results
python -m experiments.aggregate_stability
python -m experiments.aggregate_cross_width
```

### Visualization

```
notebooks/interpretability_analysis.ipynb
```

---

## Reproducibility

The project emphasizes experimental discipline:

* Deterministic seed control
* Structured experiment runners
* JSON-based logging
* CSV aggregation pipelines
* Modular analysis utilities

All experiments can be replicated or extended by modifying configuration files.

---

## Future Work

Potential extensions include:

* Scaling experiments to ImageNet
* Applying the framework to larger ViT variants
* Comparing SAE features with attention head behavior
* Tracking feature evolution across backbone training checkpoints
* Evaluating transferability of discovered features

---

## Closing Remarks

VisionSAE provides a structured framework for analyzing representation geometry in vision transformers using sparse autoencoders. By combining alignment metrics, width scaling, and interpretability tools, the project moves toward a more systematic understanding of feature formation in transformer-based vision models.

The repository is intended as both a research exploration and a foundation for further investigation into mechanistic interpretability in vision systems.
