"""
evaluate.py
-----------
Full evaluation on the test set.

Metrics reported
  - Accuracy
  - Precision, Recall (Sensitivity), F1-score
  - Specificity
  - AUC-ROC
  - Confusion matrix
  - Per-class classification report

All plots and the metrics summary are saved to outputs/results/.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    confusion_matrix,
)

import config
from dataset import get_dataloaders
from model import get_model
from utils import load_checkpoint, plot_confusion_matrix


# ─────────────────────────────────────────────
# Collect predictions
# ─────────────────────────────────────────────

@torch.no_grad()
def get_predictions(model, loader, device):
    """
    Run inference on the full loader.

    Returns
    -------
    all_labels  : np.ndarray (N,)
    all_preds   : np.ndarray (N,)   – argmax predictions
    all_probs   : np.ndarray (N, C) – softmax probabilities
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for images, labels in tqdm(loader, desc="  Evaluating", ncols=80):
        images = images.to(device, non_blocking=True)

        logits = model(images)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = np.argmax(probs, axis=1)

        all_labels.extend(labels.numpy())
        all_preds.extend(preds)
        all_probs.extend(probs)

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


# ─────────────────────────────────────────────
# ROC curve plot
# ─────────────────────────────────────────────

def plot_roc_curve(y_true, y_prob, save_path=None):
    """Plot the ROC curve for the PNEUMONIA class (index 1)."""
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    auc_score   = roc_auc_score(y_true, y_prob[:, 1])

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2,
             label=f"ROC (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0, 1])
    plt.ylim([0, 1.02])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("ROC Curve — Pneumonia Detection")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✔ ROC curve saved → {save_path}")
    plt.show()
    return auc_score


# ─────────────────────────────────────────────
# Main evaluation function
# ─────────────────────────────────────────────

def evaluate(model=None):
    print("\n" + "=" * 55)
    print("  Pneumonia Detection — Evaluation")
    print("=" * 55)

    # ── Load model ──────────────────────────
    if model is None:
        print("\nLoading model from checkpoint …")
        model = get_model()
        model, _, _, _ = load_checkpoint(config.MODEL_PATH, model)
    model.to(config.DEVICE)

    # ── Load test data ──────────────────────
    print("\nLoading test data …")
    _, _, test_loader = get_dataloaders(use_weighted_sampler=False)

    # ── Predictions ─────────────────────────
    print("\nRunning inference …")
    y_true, y_pred, y_prob = get_predictions(model, test_loader, config.DEVICE)

    # ── Metrics ─────────────────────────────
    acc = accuracy_score(y_true, y_pred) * 100

    # Confusion matrix values
    cm                      = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp          = cm.ravel()
    sensitivity             = tp / (tp + fn) * 100   # recall for PNEUMONIA
    specificity             = tn / (tn + fp) * 100   # recall for NORMAL
    precision               = tp / (tp + fp) * 100
    f1                      = 2 * precision * sensitivity / (precision + sensitivity)
    auc                     = roc_auc_score(y_true, y_prob[:, 1]) * 100

    # ── Print report ────────────────────────
    print("\n" + "─" * 45)
    print(f"  Accuracy         : {acc:.2f}%")
    print(f"  AUC-ROC          : {auc:.2f}%")
    print(f"  Sensitivity      : {sensitivity:.2f}%   (Recall – Pneumonia)")
    print(f"  Specificity      : {specificity:.2f}%   (Recall – Normal)")
    print(f"  Precision        : {precision:.2f}%")
    print(f"  F1-Score         : {f1:.2f}%")
    print("─" * 45)

    print("\nPer-class report:")
    print(classification_report(y_true, y_pred,
                                target_names=config.CLASSES, digits=4))

    # ── Save metrics JSON ───────────────────
    metrics = {
        "accuracy":    round(acc, 4),
        "auc_roc":     round(auc, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "precision":   round(precision, 4),
        "f1_score":    round(f1, 4),
    }
    metrics_path = os.path.join(config.RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\n  ✔ Metrics saved → {metrics_path}")

    # ── Confusion matrix ────────────────────
    cm_path = os.path.join(config.RESULTS_DIR, "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred,
                          classes=config.CLASSES,
                          save_path=cm_path)

    # ── ROC curve ───────────────────────────
    roc_path = os.path.join(config.RESULTS_DIR, "roc_curve.png")
    plot_roc_curve(y_true, y_prob, save_path=roc_path)

    return metrics


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    evaluate()
