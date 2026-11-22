# Pneumonia Detection from Chest X-rays

A deep learning project to detect pneumonia from chest X-ray images using transfer learning (DenseNet121).

## Project Structure

```
Pneumonia Detection/
├── README.md
├── requirements.txt
├── config.py                  # All hyperparameters and paths
├── dataset.py                 # Dataset class and data loaders
├── model.py                   # Model architecture
├── train.py                   # Training loop
├── evaluate.py                # Evaluation metrics and plots
├── predict.py                 # Single image inference
├── gradcam.py                 # Grad-CAM visualization
├── utils.py                   # Helper utilities
├── notebooks/
│   └── exploration.ipynb      # EDA and full walkthrough notebook
└── outputs/                   # Saved models, plots (auto-created)
```

## Dataset

Download the dataset from Kaggle:
- **Dataset:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Extract to `data/chest_xray/` so the structure looks like:

```
data/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Train
```bash
python train.py
```

### Evaluate
```bash
python evaluate.py
```

### Predict on a single image
```bash
python predict.py --image path/to/xray.jpg
```

### Grad-CAM visualization
```bash
python gradcam.py --image path/to/xray.jpg
```

## Model

- **Architecture:** DenseNet121 pretrained on ImageNet, fine-tuned for binary classification
- **Input:** 224×224 RGB chest X-ray images
- **Output:** Normal / Pneumonia probability

## Novel Contribution — Confidence-Calibrated Triage System

Most pneumonia detection projects output a binary label (Normal/Pneumonia). This project goes further with a **3-level clinical triage system**:

| Level | Condition | Clinical Action |
|---|---|---|
| 🟢 LOW RISK | P(Pneumonia) < 20% | Deprioritize — safe |
| 🟡 UNCERTAIN | 20% ≤ P ≤ 80% | Send to radiologist immediately |
| 🔴 HIGH RISK | P(Pneumonia) > 80% | Urgent treatment required |

The threshold is **adjustable** per deployment context (ICU vs. outpatient). This is what real clinical AI systems actually need — a model that says "I'm not sure" is more useful than one that always forces a binary answer.

```bash
# Single image triage
python triage.py --image path/to/xray.jpeg

# Batch triage entire folder (generates priority-sorted report)
python triage.py --folder path/to/folder/

# Analyze how threshold choice affects triage distribution
python triage.py --analyze-thresholds
```

## Key Results (expected)

| Metric      | Value  |
|-------------|--------|
| Accuracy    | ~94%   |
| AUC-ROC     | ~97%   |
| Sensitivity | ~96%   |
| Specificity | ~90%   |

## Clinical Note

Sensitivity (recall for Pneumonia) is prioritized over specificity to minimize false negatives, as missing a pneumonia case is more dangerous than a false alarm.
