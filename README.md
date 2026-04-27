# PyTorch CV Classification Lab

A practical PyTorch repository for image classification experiments, transfer learning, and model comparison in computer vision.

## Overview

This repository contains a set of computer vision experiments for image classification using PyTorch.
It serves as a practice and research workspace for studying deep learning workflows, comparing architectures, and testing different training strategies.

The project includes experimental scripts, notebooks, and helper utilities that make it easier to explore:

- transfer learning
- frozen backbone approaches
- classifier head tuning
- training pipeline variations
- evaluation and comparison of results

This repository is intended as a structured experimentation environment rather than a production-ready ML service.

## Goals

The main goals of this project are:

- to gain practical experience with image classification in PyTorch;
- to compare different training approaches on the same problem;
- to organize experiments in a reproducible and understandable way;
- to better understand the full CV workflow from data preparation to evaluation.

## Repository Structure

```text
.
├── notebooks/      # exploratory notebooks
├── seminars/       # seminar / practice materials
├── tools/          # helper scripts
├── assets/         # images and visual materials
├── docs/           # experiment and dataset notes
├── *.py            # experiment scripts, if available
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Documentation

- [Experiment Guide](docs/EXPERIMENT_GUIDE.md)

## Typical Workflow

1. prepare or download the dataset;
2. open the relevant notebook or experiment script;
3. configure model, transforms, training settings, and dataset paths;
4. run training or feature-extraction experiments;
5. evaluate metrics;
6. compare results across experiments.

## Running Experiments

This repository is notebook- and experiment-oriented. Start by checking available notebooks and scripts:

```bash
ls notebooks
ls seminars
ls tools
ls *.py
```

Then run the selected notebook or script depending on the experiment you want to reproduce.

## Recommended Result Tracking

| Experiment | Model | Input Size | Epochs | Validation Accuracy | Notes |
|---|---|---:|---:|---:|---|
| baseline | simple CNN | 224x224 | 5 | to be measured | first comparison point |
| transfer-learning | pretrained backbone | 224x224 | 5 | to be measured | frozen feature extractor |

## What Is Covered

Depending on the experiment, the repository may include:

- transfer learning;
- feature extraction with frozen backbones;
- classifier fine-tuning;
- validation and metric tracking;
- notebook-based exploration of results;
- comparison of different image classification approaches.

## What This Project Demonstrates

- practical PyTorch usage;
- computer vision experimentation;
- organization of ML scripts and research code;
- model comparison workflows;
- hands-on work with image classification pipelines.

## Possible Improvements

- add configuration files for experiments;
- add metric logging and experiment tracking;
- save checkpoints in a standardized structure;
- add inference scripts;
- provide benchmark tables in the README;
- pin dependency versions for more reproducible environments.

## Tech Stack

- Python
- PyTorch
- TorchVision
- Jupyter Notebook
- NumPy / Pandas
- Matplotlib
- scikit-learn

## Notes

This repository is intended as a learning and experimentation workspace focused on image classification and model evaluation.
