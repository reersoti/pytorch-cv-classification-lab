# PyTorch CV Classification Lab

A compact research repository with **image classification experiments in PyTorch**.
The project focuses on transfer learning with **frozen backbones**, feature extraction, fusion heads, and ensemble-style comparison scripts.

## What is inside
- multiple final experiment scripts for EfficientNet, ResNet, ConvNeXt, DenseNet, MobileNet, and VGG
- reusable training components in `experiment_template.py`
- feature extraction pipeline in `extract_all_features.py`
- fusion and pseudo-labeling experiments
- seminar notebooks and auxiliary practice scripts
- result images in `assets/`

## Project structure
```text
.
├── assets/                    # plots and result screenshots
├── notebooks/                 # seminar notebooks
├── seminars/                  # seminar practice scripts
├── tools/                     # utility and analysis scripts
├── competition_beater.py
├── experiment_template.py     # shared dataset/model helpers
├── extract_all_features.py    # offline feature extraction
├── *_final.py                 # main experiment entry points
├── research*.py               # research iterations
└── requirements.txt
```

## Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset layout
By default, the scripts expect the dataset in `./tnn2025`.
A typical layout is:

```text
tnn2025/
├── train.csv
├── train/
│   └── train_256/
└── test/
    └── test_256/
```

## Typical workflow
### 1) Extract backbone features
```bash
python extract_all_features.py
```

### 2) Run an experiment
Examples:
```bash
python efficientnet_b2_dense_final.py
python triple_fusion_final.py
python ultimate_fusion_final.py
```

### 3) Analyse or compare outputs
Utility scripts are stored in `tools/`.

## Notes
- Some scripts are competition-oriented and assume a specific dataset format.
- This repository keeps experiment files explicit instead of hiding them behind a large framework.
- Temporary local files were removed to keep the repository clean for GitHub.

## Recommended repository name
**pytorch-cv-classification-lab**

Alternative names:
- `image-classification-experiments`
- `frozen-backbone-cv-benchmarks`
- `computer-vision-practice-pytorch`
