"""
config.py
---------
Central configuration for all hyperparameters, paths, and settings.
Edit this file to change any training or model parameters.
"""

import os
import torch

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "chest_xray")
TRAIN_DIR   = os.path.join(DATA_DIR, "train")
VAL_DIR     = os.path.join(DATA_DIR, "val")
TEST_DIR    = os.path.join(DATA_DIR, "test")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
MODEL_PATH  = os.path.join(OUTPUT_DIR, "best_model.pth")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Classes
# ─────────────────────────────────────────────
CLASSES     = ["NORMAL", "PNEUMONIA"]
NUM_CLASSES = 2

# ─────────────────────────────────────────────
# Image settings
# ─────────────────────────────────────────────
IMAGE_SIZE  = 224          # Input image size (224x224)
CHANNELS    = 3            # RGB

# ImageNet normalization (used since we apply transfer learning)
MEAN        = [0.485, 0.456, 0.406]
STD         = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────
# Training hyperparameters
# ─────────────────────────────────────────────
BATCH_SIZE      = 32
NUM_EPOCHS      = 20
LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-4
DROPOUT_RATE    = 0.5

# Learning rate scheduler
LR_STEP_SIZE    = 7        # Reduce LR every N epochs
LR_GAMMA        = 0.1      # LR multiplication factor

# Early stopping
PATIENCE        = 5        # Stop if val loss doesn't improve for N epochs

# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
MODEL_NAME      = "densenet121"   # options: densenet121, resnet50, efficientnet_b0, vgg16, mobilenet_v3
PRETRAINED      = True
FREEZE_BACKBONE = False           # Set True to only train the classifier head

# ─────────────────────────────────────────────
# Class weights (to handle imbalance)
# ─────────────────────────────────────────────
# NORMAL ~1341, PNEUMONIA ~3875 in training set → ratio ~1:2.9
# Will be computed dynamically in dataset.py; these are fallback defaults
CLASS_WEIGHTS = [2.9, 1.0]       # [NORMAL weight, PNEUMONIA weight]

# ─────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
SEED = 42

# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────
NUM_WORKERS = 4 if torch.cuda.is_available() else 0
PIN_MEMORY  = True if torch.cuda.is_available() else False
