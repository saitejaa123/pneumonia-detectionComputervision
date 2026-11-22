"""
triage.py
---------
Confidence-Calibrated Triage System — the novel contribution of this project.

Instead of a hard binary prediction, the system outputs a 3-level clinical
triage decision based on model confidence:

    🟢 LOW RISK    : High confidence NORMAL   → Can be safely deprioritized
    🟡 UNCERTAIN   : Low confidence either way → Must go to radiologist ASAP
    🔴 HIGH RISK   : High confidence PNEUMONIA → Urgent treatment required

This directly addresses a real clinical need: radiologists want to know not
just WHAT the model predicts, but HOW SURE it is. A low-confidence prediction
is more dangerous than no prediction at all.

The confidence threshold is tunable — hospitals can adjust it based on their
risk tolerance (e.g., ICU vs. outpatient clinic).

Usage:
    python triage.py --image path/to/xray.jpeg
    python triage.py --folder path/to/folder/  --save-report
"""

import os
import json
import argparse
import numpy as np
from PIL import Image
from datetime import datetime
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

import config
from model import get_model
from utils import load_checkpoint
from dataset import get_transforms


# ─────────────────────────────────────────────
# Triage level definitions
# ─────────────────────────────────────────────

TRIAGE_LEVELS = {
    "LOW_RISK": {
        "label":       "🟢 LOW RISK",
        "short":       "LOW RISK",
        "color":       "#00b894",
        "description": "High confidence NORMAL. Safe to deprioritize.",
        "action":      "Routine follow-up. No immediate intervention needed.",
        "urgency":     1,
    },
    "UNCERTAIN": {
        "label":       "🟡 UNCERTAIN",
        "short":       "UNCERTAIN",
        "color":       "#fdcb6e",
        "description": "Low model confidence. Cannot make reliable prediction.",
        "action":      "REFER TO RADIOLOGIST IMMEDIATELY for manual review.",
        "urgency":     2,
    },
    "HIGH_RISK": {
        "label":       "🔴 HIGH RISK",
        "short":       "HIGH RISK",
        "color":       "#d63031",
        "description": "High confidence PNEUMONIA detected.",
        "action":      "URGENT: Begin treatment protocol. Notify attending physician.",
        "urgency":     3,
    },
}


# ─────────────────────────────────────────────
# Triage logic
# ─────────────────────────────────────────────

def assign_triage_level(
    pneumonia_prob: float,
    high_conf_threshold: float = 0.80,
    low_conf_threshold:  float = 0.20,
) -> dict:
    """
    Convert a pneumonia probability into a clinical triage decision.

    Parameters
    ----------
    pneumonia_prob       : P(PNEUMONIA) from softmax, in [0, 1]
    high_conf_threshold  : above this → HIGH RISK (default 0.80)
    low_conf_threshold   : below this → LOW RISK  (default 0.20)
    anything in between  → UNCERTAIN

    Returns
    -------
    dict with triage level info + probability
    """
    if pneumonia_prob >= high_conf_threshold:
        level = "HIGH_RISK"
    elif pneumonia_prob <= low_conf_threshold:
        level = "LOW_RISK"
    else:
        level = "UNCERTAIN"

    result = TRIAGE_LEVELS[level].copy()
    result["pneumonia_prob"] = round(float(pneumonia_prob) * 100, 2)
    result["normal_prob"]    = round((1 - float(pneumonia_prob)) * 100, 2)
    result["level_key"]      = level
    return result


# ─────────────────────────────────────────────
# Single image triage
# ─────────────────────────────────────────────

@torch.no_grad()
def triage_image(
    image_path: str,
    model=None,
    high_conf_threshold: float = 0.80,
    low_conf_threshold:  float = 0.20,
    show_plot: bool = True,
    save_path: str = None,
) -> dict:
    """
    Run triage on a single X-ray image.

    Returns
    -------
    dict with: image_path, triage_level, pneumonia_prob, normal_prob,
               action, description
    """
    # ── Load model ──────────────────────────
    if model is None:
        model = get_model()
        model, _, _, _ = load_checkpoint(config.MODEL_PATH, model)
    model.to(config.DEVICE)
    model.eval()

    # ── Preprocess ──────────────────────────
    transform    = get_transforms("test")
    pil_image    = Image.open(image_path).convert("RGB")
    input_tensor = transform(pil_image).unsqueeze(0).to(config.DEVICE)

    # ── Inference ───────────────────────────
    logits          = model(input_tensor)
    probs           = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    pneumonia_prob  = float(probs[1])

    # ── Triage decision ──────────────────────
    triage = assign_triage_level(pneumonia_prob,
                                  high_conf_threshold,
                                  low_conf_threshold)
    triage["image_path"] = image_path

    # ── Print ───────────────────────────────
    _print_triage_result(triage, os.path.basename(image_path))

    # ── Visualize ───────────────────────────
    if show_plot or save_path:
        _plot_triage(pil_image, triage, save_path)

    return triage


def _print_triage_result(triage: dict, filename: str):
    """Pretty-print triage result to terminal."""
    print("\n" + "═" * 52)
    print(f"  FILE    : {filename}")
    print(f"  TRIAGE  : {triage['label']}")
    print(f"  P(Pneumonia) : {triage['pneumonia_prob']:.1f}%")
    print(f"  P(Normal)    : {triage['normal_prob']:.1f}%")
    print(f"  ACTION  : {triage['action']}")
    print("═" * 52)


def _plot_triage(pil_image: Image.Image, triage: dict, save_path: str = None):
    """Plot the triage result with clinical styling."""
    color   = triage["color"]
    level   = triage["short"]

    fig = plt.figure(figsize=(13, 6), facecolor="#1a1a2e")
    gs  = fig.add_gridspec(1, 3, width_ratios=[1.8, 1, 1.2])

    # ── X-ray panel ─────────────────────────
    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(pil_image, cmap="gray")
    ax_img.set_title("Chest X-Ray", color="white", fontsize=12, pad=8)
    ax_img.axis("off")
    # Border colour = triage level
    for spine in ax_img.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(3)
        spine.set_visible(True)

    # ── Probability gauge ───────────────────
    ax_gauge = fig.add_subplot(gs[1])
    ax_gauge.set_facecolor("#16213e")
    pneumonia_p = triage["pneumonia_prob"]
    normal_p    = triage["normal_prob"]

    bars = ax_gauge.barh(
        ["NORMAL", "PNEUMONIA"],
        [normal_p, pneumonia_p],
        color=["#00b894", "#d63031"],
        height=0.5,
    )
    ax_gauge.set_xlim(0, 110)
    ax_gauge.set_xlabel("Probability (%)", color="white", fontsize=9)
    ax_gauge.set_title("Model Confidence", color="white", fontsize=11)
    ax_gauge.tick_params(colors="white")
    for spine in ax_gauge.spines.values():
        spine.set_edgecolor("#444")
    for bar, val in zip(bars, [normal_p, pneumonia_p]):
        ax_gauge.text(bar.get_width() + 1,
                      bar.get_y() + bar.get_height() / 2,
                      f"{val:.1f}%", va="center",
                      color="white", fontsize=10, fontweight="bold")

    # ── Triage decision panel ────────────────
    ax_info = fig.add_subplot(gs[2])
    ax_info.set_facecolor("#16213e")
    ax_info.axis("off")

    # Big triage badge
    ax_info.add_patch(
        mpatches.FancyBboxPatch(
            (0.05, 0.65), 0.90, 0.28,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor="white", linewidth=1.5,
            transform=ax_info.transAxes, zorder=2,
        )
    )
    ax_info.text(0.5, 0.79, level,
                 transform=ax_info.transAxes,
                 ha="center", va="center",
                 fontsize=16, fontweight="bold", color="white", zorder=3)

    ax_info.text(0.5, 0.56,
                 triage["description"],
                 transform=ax_info.transAxes,
                 ha="center", va="center",
                 fontsize=8, color="#dfe6e9",
                 wrap=True)

    ax_info.text(0.5, 0.35,
                 "RECOMMENDED ACTION:",
                 transform=ax_info.transAxes,
                 ha="center", fontsize=8,
                 color="#b2bec3", fontweight="bold")

    ax_info.text(0.5, 0.18,
                 triage["action"],
                 transform=ax_info.transAxes,
                 ha="center", va="center",
                 fontsize=8, color=color,
                 fontweight="bold",
                 wrap=True)

    # Timestamp
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    ax_info.text(0.5, 0.04, f"Generated: {ts}",
                 transform=ax_info.transAxes,
                 ha="center", fontsize=7, color="#636e72")

    ax_info.set_title("Clinical Triage", color="white", fontsize=11)

    plt.suptitle("AI-Assisted Pneumonia Triage System",
                 color="white", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  ✔ Triage plot saved → {save_path}")

    plt.show()
    plt.close()


# ─────────────────────────────────────────────
# Batch / Folder triage
# ─────────────────────────────────────────────

@torch.no_grad()
def triage_folder(
    folder_path: str,
    model=None,
    high_conf_threshold: float = 0.80,
    low_conf_threshold:  float = 0.20,
    save_report: bool = True,
) -> list:
    """
    Run triage on all images in a folder and produce a summary report.

    Returns list of triage result dicts, sorted by urgency (highest first).
    """
    supported = {".jpg", ".jpeg", ".png"}
    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in supported
    ]

    if not files:
        print(f"No images found in {folder_path}")
        return []

    # ── Load model once ──────────────────────
    if model is None:
        model = get_model()
        model, _, _, _ = load_checkpoint(config.MODEL_PATH, model)
    model.to(config.DEVICE)
    model.eval()

    transform = get_transforms("test")
    results   = []

    print(f"\nTriaging {len(files)} images …\n")
    for fpath in tqdm(files, desc="  Triage", ncols=70):
        try:
            pil_image    = Image.open(fpath).convert("RGB")
            input_tensor = transform(pil_image).unsqueeze(0).to(config.DEVICE)
            logits       = model(input_tensor)
            probs        = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            pneumonia_p  = float(probs[1])

            triage = assign_triage_level(
                pneumonia_p, high_conf_threshold, low_conf_threshold
            )
            triage["image_path"] = fpath
            triage["filename"]   = os.path.basename(fpath)
            results.append(triage)
        except Exception as e:
            print(f"  Error processing {fpath}: {e}")

    # Sort: HIGH_RISK first, then UNCERTAIN, then LOW_RISK
    results.sort(key=lambda x: x["urgency"], reverse=True)

    # ── Summary stats ────────────────────────
    counts = {"HIGH_RISK": 0, "UNCERTAIN": 0, "LOW_RISK": 0}
    for r in results:
        counts[r["level_key"]] += 1

    print("\n" + "═" * 52)
    print("  TRIAGE SUMMARY")
    print("═" * 52)
    print(f"  Total images   : {len(results)}")
    print(f"  🔴 HIGH RISK   : {counts['HIGH_RISK']}")
    print(f"  🟡 UNCERTAIN   : {counts['UNCERTAIN']}")
    print(f"  🟢 LOW RISK    : {counts['LOW_RISK']}")
    print("═" * 52)

    if counts["HIGH_RISK"] > 0:
        print(f"\n  ⚠ {counts['HIGH_RISK']} patient(s) require URGENT attention:\n")
        for r in results:
            if r["level_key"] == "HIGH_RISK":
                print(f"    → {r['filename']}  (P_pneumonia={r['pneumonia_prob']:.1f}%)")

    if counts["UNCERTAIN"] > 0:
        print(f"\n  ⚠ {counts['UNCERTAIN']} case(s) need RADIOLOGIST REVIEW:\n")
        for r in results:
            if r["level_key"] == "UNCERTAIN":
                print(f"    → {r['filename']}  (P_pneumonia={r['pneumonia_prob']:.1f}%)")

    # ── Save JSON report ─────────────────────
    if save_report:
        report = {
            "timestamp":  datetime.now().isoformat(),
            "total":      len(results),
            "summary":    counts,
            "cases":      [
                {
                    "filename":      r["filename"],
                    "triage_level":  r["level_key"],
                    "pneumonia_prob": r["pneumonia_prob"],
                    "normal_prob":   r["normal_prob"],
                    "action":        r["action"],
                }
                for r in results
            ],
        }
        report_path = os.path.join(config.RESULTS_DIR, "triage_report.json")
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  ✔ Triage report saved → {report_path}")

        # ── Summary plot ──────────────────────
        _plot_triage_summary(results, counts)

    return results


def _plot_triage_summary(results: list, counts: dict):
    """Bar chart + pie chart summary of batch triage results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#1a1a2e")

    colors = ["#d63031", "#fdcb6e", "#00b894"]
    labels = ["HIGH RISK", "UNCERTAIN", "LOW RISK"]
    values = [counts["HIGH_RISK"], counts["UNCERTAIN"], counts["LOW_RISK"]]

    # Bar chart
    axes[0].set_facecolor("#16213e")
    bars = axes[0].bar(labels, values, color=colors, width=0.5, edgecolor="white")
    axes[0].set_title("Triage Distribution", color="white", fontsize=13)
    axes[0].tick_params(colors="white")
    axes[0].set_ylabel("Number of Cases", color="white")
    for spine in axes[0].spines.values():
        spine.set_edgecolor("#444")
    for bar, val in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.2,
                     str(val), ha="center",
                     color="white", fontweight="bold", fontsize=12)

    # Pie chart
    axes[1].set_facecolor("#16213e")
    non_zero = [(v, l, c) for v, l, c in zip(values, labels, colors) if v > 0]
    if non_zero:
        v, l, c = zip(*non_zero)
        wedges, texts, autotexts = axes[1].pie(
            v, labels=l, colors=c,
            autopct="%1.1f%%", startangle=90,
            textprops={"color": "white"},
            wedgeprops={"edgecolor": "#1a1a2e", "linewidth": 2},
        )
        for at in autotexts:
            at.set_fontsize(10)
            at.set_fontweight("bold")
    axes[1].set_title("Triage Breakdown", color="white", fontsize=13)

    plt.suptitle("Batch Triage Summary", color="white",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    summary_path = os.path.join(config.RESULTS_DIR, "triage_summary.png")
    plt.savefig(summary_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  ✔ Summary plot saved → {summary_path}")
    plt.show()
    plt.close()


# ─────────────────────────────────────────────
# Threshold sensitivity analysis
# ─────────────────────────────────────────────

def analyze_thresholds(model=None, save_path: str = None):
    """
    Show how the UNCERTAIN zone size changes with different confidence thresholds.
    Helps clinical staff choose the right threshold for their risk tolerance.
    """
    from dataset import get_dataloaders
    import numpy as np

    print("\nRunning threshold sensitivity analysis on test set …")

    if model is None:
        model = get_model()
        model, _, _, _ = load_checkpoint(config.MODEL_PATH, model)
    model.to(config.DEVICE)
    model.eval()

    _, _, test_loader = get_dataloaders(use_weighted_sampler=False)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="  Collecting probs", ncols=70):
            images = images.to(config.DEVICE)
            probs  = torch.softmax(model(images), dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    thresholds = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    rows = []

    for high_t in thresholds:
        low_t        = 1 - high_t
        high_risk    = (all_probs >= high_t).sum()
        low_risk     = (all_probs <= low_t).sum()
        uncertain    = len(all_probs) - high_risk - low_risk

        # Among HIGH_RISK predictions, how many are truly pneumonia?
        hr_mask      = all_probs >= high_t
        hr_precision = all_labels[hr_mask].mean() * 100 if hr_mask.sum() > 0 else 0

        rows.append({
            "threshold": f">{high_t:.0%} / <{low_t:.0%}",
            "high_risk": high_risk,
            "uncertain": uncertain,
            "low_risk":  low_risk,
            "hr_precision": f"{hr_precision:.1f}%",
        })

    # Print table
    print(f"\n{'Threshold':>18}  {'HIGH RISK':>9}  {'UNCERTAIN':>9}  {'LOW RISK':>8}  {'HR Precision':>12}")
    print("─" * 70)
    for r in rows:
        print(f"  {r['threshold']:>16}  {r['high_risk']:>9}  {r['uncertain']:>9}  "
              f"{r['low_risk']:>8}  {r['hr_precision']:>12}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(thresholds))
    ax.bar([i - 0.25 for i in x], [r["high_risk"] for r in rows],
           width=0.25, label="HIGH RISK", color="#d63031")
    ax.bar([i       for i in x], [r["uncertain"] for r in rows],
           width=0.25, label="UNCERTAIN", color="#fdcb6e")
    ax.bar([i + 0.25 for i in x], [r["low_risk"] for r in rows],
           width=0.25, label="LOW RISK", color="#00b894")
    ax.set_xticks(list(x))
    ax.set_xticklabels([r["threshold"] for r in rows], rotation=15)
    ax.set_ylabel("Number of Cases")
    ax.set_title("Triage Distribution vs. Confidence Threshold\n(Test Set)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✔ Threshold analysis saved → {save_path}")
    plt.show()
    plt.close()


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AI-Assisted Pneumonia Triage System"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  help="Path to a single X-ray image")
    group.add_argument("--folder", help="Path to a folder of X-ray images")
    group.add_argument("--analyze-thresholds", action="store_true",
                       help="Run threshold sensitivity analysis on test set")

    parser.add_argument("--high-threshold", type=float, default=0.80,
                        help="Confidence above which → HIGH RISK (default 0.80)")
    parser.add_argument("--low-threshold",  type=float, default=0.20,
                        help="Confidence below which → LOW RISK (default 0.20)")
    parser.add_argument("--save", default=None,
                        help="Path to save output image/report")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    if args.image:
        triage_image(
            image_path          = args.image,
            high_conf_threshold = args.high_threshold,
            low_conf_threshold  = args.low_threshold,
            show_plot           = not args.no_plot,
            save_path           = args.save,
        )
    elif args.folder:
        triage_folder(
            folder_path         = args.folder,
            high_conf_threshold = args.high_threshold,
            low_conf_threshold  = args.low_threshold,
            save_report         = True,
        )
    elif args.analyze_thresholds:
        analyze_thresholds(
            save_path = args.save or os.path.join(config.RESULTS_DIR, "threshold_analysis.png")
        )
