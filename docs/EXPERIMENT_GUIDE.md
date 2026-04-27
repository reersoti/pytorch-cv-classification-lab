# Experiment Guide

This repository is a workspace for computer vision image classification experiments with PyTorch.

## Recommended Data Layout

```text
data/
├── train/
│   ├── class_1/
│   └── class_2/
├── val/
│   ├── class_1/
│   └── class_2/
└── test/
    ├── class_1/
    └── class_2/
```

This structure works well with `torchvision.datasets.ImageFolder`.

## Typical Experiment Flow

1. Prepare image folders.
2. Choose a notebook or script.
3. Configure transforms and dataset paths.
4. Select model architecture.
5. Run training.
6. Save metrics and compare experiments.

## Suggested Result Table

| Experiment | Model | Input Size | Epochs | Validation Accuracy | Notes |
|---|---|---:|---:|---:|---|
| baseline | simple CNN | 224x224 | 5 | to be measured | first comparison point |
| transfer-learning | pretrained backbone | 224x224 | 5 | to be measured | frozen feature extractor |

## Files That Should Usually Stay Out of Git

```gitignore
data/
checkpoints/
runs/
*.pt
*.pth
```

## Reproducibility Checklist

- Dataset folder structure is documented.
- Model architecture is written down.
- Training parameters are recorded.
- Metrics are saved in a table or notebook output.
- Random seed is fixed where possible.
