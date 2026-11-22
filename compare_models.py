"""
compare_models.py
-----------------
Loads saved checkpoints for all 5 models, evaluates each on the test set,
generates a comparison bar chart, radar chart, and saves a summary JSON.

Run AFTER all models have been trained (via run_all_models.py).
If a checkpoint doesn't exist yet it is skipped gracefully.

Usage:
    python compare_models.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

import config
from model import get_model
from utils import load_checkpoint
from evaluate import get_predictions
from dataset import get_dataloaders
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix
)

# ─────────────────────────────────────────────
MODELS = ["densenet121", "resnet50", "efficientnet_b0", "vgg16", "mobilenet_v3"]
COMPARISON_DIR = os.path.join(config.OUTPUT_DIR, "comparison")
os.makedirs(COMPARISON_DIR, exist_ok=True)

# Dark theme colours per model
MODEL_COLORS = {
    "densenet121":     "#58a6ff",
    "resnet50":        "#3fb950",
    "efficientnet_b0": "#bc8cff",
    "vgg16":           "#ffa657",
    "mobilenet_v3":    "#f85149",
}

BG    = "#0d1117"
PANEL = "#161b22"
WHITE = "#e6edf3"


# ─────────────────────────────────────────────
# Evaluate one checkpoint
# ─────────────────────────────────────────────

def evaluate_checkpoint(model_name: str, test_loader) -> dict | None:
    ckpt_path = os.path.join(config.OUTPUT_DIR, f"best_model_{model_name}.pth")

    # Also accept the default best_model.pth for densenet121
    if not os.path.exists(ckpt_path) and model_name == "densenet121":
        ckpt_path = config.MODEL_PATH

    if not os.path.exists(ckpt_path):
        print(f"  [SKIP] No checkpoint for {model_name} at {ckpt_path}")
        return None

    print(f"  Evaluating {model_name} …")

    # Must set config.MODEL_NAME BEFORE calling get_model so the
    # correct architecture is built before loading state_dict
    config.MODEL_NAME = model_name

    # Load checkpoint metadata to verify architecture match
    raw = torch.load(ckpt_path, map_location=config.DEVICE)
    saved_name = raw.get("model_name", model_name)
    if saved_name != model_name:
        print(f"  [WARN] Checkpoint says model_name='{saved_name}', expected '{model_name}'. Skipping.")
        return None

    model = get_model(name=model_name)
    model, _, _, _ = load_checkpoint(ckpt_path, model)
    model.eval()

    y_true, y_pred, y_prob = get_predictions(model, test_loader, config.DEVICE)

    acc  = accuracy_score(y_true, y_pred) * 100
    auc  = roc_auc_score(y_true, y_prob[:, 1]) * 100
    cm   = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) * 100
    spec = tn / (tn + fp) * 100
    prec = tp / (tp + fp) * 100
    f1   = 2 * prec * sens / (prec + sens)

    return {
        "accuracy":    round(acc,  4),
        "auc_roc":     round(auc,  4),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "precision":   round(prec, 4),
        "f1_score":    round(f1,   4),
    }


# ─────────────────────────────────────────────
# Bar chart comparison
# ─────────────────────────────────────────────

def plot_bar_comparison(results: dict):
    metrics   = ["accuracy", "auc_roc", "sensitivity", "specificity", "f1_score"]
    labels    = ["Accuracy", "AUC-ROC", "Sensitivity", "Specificity", "F1-Score"]
    models    = list(results.keys())
    n_metrics = len(metrics)
    n_models  = len(models)
    x         = np.arange(n_metrics)
    width     = 0.75 / n_models

    fig, ax = plt.subplots(figsize=(16, 7), facecolor=BG)
    ax.set_facecolor(PANEL)

    for i, model_name in enumerate(models):
        vals   = [results[model_name].get(m, 0) for m in metrics]
        offset = (i - n_models / 2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width,
                        label=model_name,
                        color=MODEL_COLORS.get(model_name, "#888"),
                        edgecolor=BG, linewidth=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{val:.1f}",
                    ha="center", va="bottom",
                    color=WHITE, fontsize=6.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=WHITE, fontsize=11)
    ax.set_ylabel("Score (%)", color=WHITE, fontsize=11)
    ax.set_ylim(60, 105)
    ax.set_title("Model Comparison — All Metrics on Test Set",
                 color=WHITE, fontsize=14, fontweight="bold", pad=14)
    ax.tick_params(colors=WHITE)
    ax.yaxis.set_tick_params(labelcolor=WHITE)
    ax.legend(facecolor=PANEL, edgecolor="#30363d",
              labelcolor=WHITE, fontsize=10, loc="lower right")
    ax.grid(axis="y", color="#30363d", linewidth=0.8, alpha=0.6)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    out = os.path.join(COMPARISON_DIR, "bar_comparison.png")
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✔ Bar chart saved → {out}")


# ─────────────────────────────────────────────
# Radar chart
# ─────────────────────────────────────────────

def plot_radar_comparison(results: dict):
    metrics    = ["accuracy", "auc_roc", "sensitivity", "specificity", "precision", "f1_score"]
    labels     = ["Accuracy", "AUC-ROC", "Sensitivity", "Specificity", "Precision", "F1-Score"]
    n          = len(metrics)
    angles     = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles    += angles[:1]                          # close the polygon

    fig, ax = plt.subplots(figsize=(9, 9),
                           subplot_kw=dict(polar=True),
                           facecolor=BG)
    ax.set_facecolor(PANEL)

    # Reference rings
    for r in [70, 80, 90, 95, 100]:
        ax.plot(angles, [r] * (n + 1), "--", color="#30363d", linewidth=0.7, zorder=0)
        ax.text(0, r + 0.5, f"{r}%", color="#8b949e", fontsize=7, ha="center")

    for model_name, m in results.items():
        vals  = [m.get(metric, 0) for metric in metrics]
        vals += vals[:1]
        color = MODEL_COLORS.get(model_name, "#888")
        ax.plot(angles, vals, "-o", color=color, linewidth=2,
                markersize=5, label=model_name)
        ax.fill(angles, vals, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color=WHITE, fontsize=11)
    ax.set_yticks([])
    ax.set_ylim(60, 105)
    ax.set_title("Model Radar Comparison", color=WHITE,
                 fontsize=14, fontweight="bold", pad=22)
    ax.spines["polar"].set_color("#30363d")

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1),
              facecolor=PANEL, edgecolor="#30363d",
              labelcolor=WHITE, fontsize=10)

    out = os.path.join(COMPARISON_DIR, "radar_comparison.png")
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✔ Radar chart saved → {out}")


# ─────────────────────────────────────────────
# Summary table image
# ─────────────────────────────────────────────

def plot_summary_table(results: dict):
    metrics     = ["accuracy", "auc_roc", "sensitivity", "specificity", "precision", "f1_score"]
    col_headers = ["Model", "Accuracy", "AUC-ROC", "Sensitivity", "Specificity", "Precision", "F1-Score"]
    rows        = []
    for model_name, m in results.items():
        row = [model_name] + [f"{m.get(k, 0):.2f}%" for k in metrics]
        rows.append(row)

    fig, ax = plt.subplots(figsize=(14, 1.2 + 0.6 * len(rows)), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    table = ax.table(
        cellText   = rows,
        colLabels  = col_headers,
        cellLoc    = "center",
        loc        = "center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # Style header
    for col in range(len(col_headers)):
        cell = table[0, col]
        cell.set_facecolor("#21262d")
        cell.set_text_props(color=WHITE, fontweight="bold")

    # Style data rows
    for row_i, (model_name, _) in enumerate(results.items(), start=1):
        color = MODEL_COLORS.get(model_name, "#888")
        for col in range(len(col_headers)):
            cell = table[row_i, col]
            cell.set_facecolor(PANEL)
            cell.set_edgecolor("#30363d")
            if col == 0:
                cell.set_text_props(color=color, fontweight="bold")
            else:
                cell.set_text_props(color=WHITE)

    ax.set_title("Model Performance Summary", color=WHITE,
                 fontsize=14, fontweight="bold", pad=10)

    out = os.path.join(COMPARISON_DIR, "summary_table.png")
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✔ Summary table saved → {out}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  Model Comparison — Evaluation")
    print("=" * 60)

    # Load test set once (shared across all models)
    print("\nLoading test data …")
    _, _, test_loader = get_dataloaders(use_weighted_sampler=False)

    results = {}
    for model_name in MODELS:
        m = evaluate_checkpoint(model_name, test_loader)
        if m is not None:
            results[model_name] = m

    if not results:
        print("\nNo checkpoints found. Run run_all_models.py first.")
        return

    # ── Print table ──────────────────────────
    print("\n" + "=" * 78)
    print(f"  {'Model':<20} {'Acc':>8} {'AUC':>8} {'Sens':>8} {'Spec':>8} {'Prec':>8} {'F1':>8}")
    print("─" * 78)
    for name, m in results.items():
        print(f"  {name:<20}"
              f"  {m['accuracy']:>6.2f}%"
              f"  {m['auc_roc']:>6.2f}%"
              f"  {m['sensitivity']:>6.2f}%"
              f"  {m['specificity']:>6.2f}%"
              f"  {m['precision']:>6.2f}%"
              f"  {m['f1_score']:>6.2f}%")
    print("=" * 78)

    # ── Charts ───────────────────────────────
    print("\nGenerating comparison charts …")
    plot_bar_comparison(results)
    plot_radar_comparison(results)
    plot_summary_table(results)

    # ── Save JSON ─────────────────────────────
    out_json = os.path.join(COMPARISON_DIR, "comparison_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  ✔ JSON results  → {out_json}")

    print("\nAll comparison outputs saved to:", COMPARISON_DIR)


if __name__ == "__main__":
    main()
