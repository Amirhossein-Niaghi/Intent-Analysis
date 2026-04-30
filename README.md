# Learning Noise-Invariant Representations for Robust Intent Detection in Persian Conversational Text

![Reproducible Research](https://img.shields.io/badge/Reproducible-Yes-brightgreen)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![CPU Only](https://img.shields.io/badge/Compute-CPU%20Only-blue)
![Deterministic](https://img.shields.io/badge/Training-Deterministic-important)
![Paper Aligned](https://img.shields.io/badge/Paper--Code-Aligned-success)
![Appendix Included](https://img.shields.io/badge/Appendix-Included-blueviolet)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Libraries](https://img.shields.io/badge/Framework-PyTorch%20%7C%20Transformers-red)
---

## Overview
This repository provides the complete implementation for the paper:

**“Learning Noise-Invariant Representations for Robust Intent Detection in Persian Conversational Text”**
The project investigates robustness degradation in transformer-based intent detection models under structured conversational noise and proposes a supervised contrastive learning framework to enforce representation-level invariance between clean and perturbed utterances.
The repository is designed with **scientific rigor, reproducibility, and reviewer accessibility** as primary goals.
---

## Key Contributions Implemented
- Structured perturbation modeling with controllable noise intensity (α)
- Supervised contrastive learning for representation invariance
- Hard negative mining for boundary refinement
- Robustness evaluation via degradation curves and AUC
- Deterministic, CPU-friendly experimental pipeline
- Full alignment with Sections 4–6 and Appendix of the paper
---

## Repository Structure

.
├── data/
│   ├── raw/                    # Raw conversational data
│   ├── processed/              # Preprocessed datasets
│   └── splits/                 # Train/val/test splits
│
├── preprocessing/
│   ├── preprocessing.py        # Text normalization & cleaning
│   └── dataset.py              # Dataset loading and splitting
│
├── noise/
│   └── noise.py                # Structured noise operator T_α
│
├── models/
│   ├── baseline_models.py      # CE, augmentation, adversarial baselines
│   ├── model.py                # Encoder + projection head
│   └── losses.py               # Supervised contrastive loss
│
├── training/
│   └── trainer.py              # Joint CE + contrastive optimization
│
├── evaluation/
│   ├── metrics.py              # Accuracy, Macro-F1
│   ├── robustness.py           # Degradation & robustness AUC
│   └── statistical_tests.py    # Paired t-tests
│
├── experiments/
│   ├── run_baselines.py
│   ├── run_proposed.py
│   └── evaluate_robustness.py
│
├── results/
│   ├── tables/                 # CSV outputs for Excel plotting
│   └── figures/                # Generated plots (optional)
│
├── configs/
│   └── hyperparameters.yaml    # Training configurations
│
├── README.md
└── requirements.txt
Each module directly corresponds to a specific methodological component discussed in the paper.

---
## Environment Setup

### Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate
Install dependencies
pip install -r requirements.txt
Core libraries only:
-torch
-transformers
-numpy
-scikit-learn

Dataset Interface
-Input format: CSV (text, label)
-Train/validation/test split is deterministic
-No dependency on external datasets
-Persian text preprocessing fully implemented
See: preprocessing/preprocessing.py

Text Preprocessing
Implemented steps include:
-Unicode normalization
-Persian character mapping
-Noise filtering
-Deterministic transformations
All preprocessing is fully reproducible and independent of model training.

Structured Noise Generation
Noise operators include:
-Character substitution
-Deletion
-Repetition
-Emoji injection
Noise intensity is controlled via parameter α ∈ [0, 0.5]
Implemented in: noise/noise.py
Models
Baselines
-Fine-tuned transformer with cross-entropy
-Data augmentation baseline
-Character-level perturbation baseline

Proposed Method
-Shared encoder (ParsBERT)
-Projection head (discarded at inference)
-Supervised contrastive loss
-Hard negative mining
See:
-models/model.py
-models/losses.py
-models/baseline_models.py

Evaluation & Robustness Metrics
Evaluation follows the exact protocol described in the paper:
-Accuracy
-Macro-F1
-Weighted-F1
-Degradation curves
-Robustness AUC
-Statistical significance across random seeds
Implemented in:
-evaluation/metrics.py
-evaluation/robustness.py
-evaluation/statistical_tests.py
All results are saved as CSV files for external plotting (e.g., Excel).

Minimal Reproduction (One Command)
To reproduce the main experimental results:
python experiments/run_experiment.py

This command:
-Trains baseline and proposed models
-Evaluates robustness across noise levels
-Generates CSV outputs for all figures in the paper
-Runs entirely on CPU.

Reproducibility Statement
-Fixed random seeds across all experiments
-Deterministic preprocessing and training
-No stochastic data augmentation
-All hyperparameters explicitly reported
-Code–paper alignment verified
For detailed reproduction steps, see:
reproduction_instructions.md

Appendix & Hyperparameters
All experimental settings, including:
-Learning rates
-Batch sizes
-λ values
-Projection dimensions
-Noise schedules
are documented in:
appendix/hyperparameters.md
This appendix directly corresponds to the Appendix section of the paper.

Open Science Commitment
To support transparency and reuse:
-All source code is publicly available
-No proprietary components are used
-Results can be independently verified
The repository is suitable for reviewer inspection

License
This project is released under the MIT License, allowing unrestricted academic and research use.

Contact
For questions regarding reproducibility or experimental details, please refer to the documentation provided in this repository.

Robust conversational systems must operate under variability rather than idealized input conditions.
This repository demonstrates that controlling representation geometry is a principled and effective strategy for achieving robustness under structured noise.
