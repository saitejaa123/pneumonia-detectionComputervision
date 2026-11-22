"""
utils.py
--------
Helper utilities: seed setting, logging, checkpoint saving/loading,
metric tracking, and miscellaneous helpers.
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import config


def set_seed(seed: int = config.SEED):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────

def save_checkpoint(state: dict, filepath: str):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    print(f"  ✔ Checkpoint saved → {filepath}")


def load_checkpoint(filepath: str, model, optimizer=None):
    """
    Load a saved checkpoint.

    Returns
    -------
    model, optimizer, start_epoch, best_val_loss
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No checkpoint found at: {filepath}")

    checkpoint = torch.load(filepath, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    start_epoch   = checkpoint.get("epoch", 0) + 1
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"  ✔ Checkpoint loaded from {filepath} (epoch {start_epoch - 1})")
    return model, optimizer, start_epoch, best_val_loss


# ─────────────────────────────────────────────
# Metric history tracker
# ─────────────────────────────────────────────

class MetricTracker:
    """Track train/val losses and accuracies across epochs."""

    def __init__(self):
        self.history = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [],
        }

    def update(self, train_loss, train_acc, val_loss, val_acc):
        self.history["train_loss"].append(train_loss)
        self.history["train_acc"].append(train_acc)
        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_acc)

    def plot(self, save_path: str = None):
        """Plot loss and accuracy curves side by side."""
        epochs = range(1, len(self.history["train_loss"]) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss
        axes[0].plot(epochs, self.history["train_loss"], "b-o", label="Train Loss")
        axes[0].plot(epochs, self.history["val_loss"],   "r-o", label="Val Loss")
        axes[0].set_title("Loss over Epochs")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True)

        # Accuracy
        axes[1].plot(epochs, self.history["train_acc"], "b-o", label="Train Acc")
        axes[1].plot(epochs, self.history["val_acc"],   "r-o", label="Val Acc")
        axes[1].set_title("Accuracy over Epochs")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  ✔ Training curves saved → {save_path}")
        plt.show()


# ─────────────────────────────────────────────
# Confusion matrix plot
# ─────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, classes=config.CLASSES, save_path=None):
    """Plot and optionally save a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✔ Confusion matrix saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────
# Early stopping
# ─────────────────────────────────────────────

class EarlyStopping:
    """
    Stop training when validation loss stops improving.

    Parameters
    ----------
    patience : int   – epochs to wait before stopping
    delta    : float – minimum improvement to count as better
    """

    def __init__(self, patience: int = config.PATIENCE, delta: float = 1e-4):
        self.patience   = patience
        self.delta      = delta
        self.counter    = 0
        self.best_loss  = float("inf")
        self.stop       = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


# ─────────────────────────────────────────────
# Misc
# ─────────────────────────────────────────────

def count_parameters(model) -> int:
    """Return number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_device_info():
    """Print GPU/CPU info."""
    if torch.cuda.is_available():
        print(f"  Device : GPU — {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  Device : CPU")
