"""
generate_diagrams.py
--------------------
Generates all architectural diagrams for the project report:
  1. Full System Pipeline Flow
  2. DenseNet121 Architecture
  3. Loss Function Equations
  4. Data Availability & Distribution
  5. Training Pipeline
  6. Triage System Flow

Run:
    python generate_diagrams.py

All diagrams saved to: outputs/results/diagrams/
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.gridspec import GridSpec

import config

SAVE_DIR = os.path.join(config.RESULTS_DIR, "diagrams")
os.makedirs(SAVE_DIR, exist_ok=True)

BG      = "#0d1117"
PANEL   = "#161b22"
BORDER  = "#30363d"
WHITE   = "#e6edf3"
BLUE    = "#58a6ff"
GREEN   = "#3fb950"
RED     = "#f85149"
YELLOW  = "#d29922"
PURPLE  = "#bc8cff"
ORANGE  = "#ffa657"
CYAN    = "#39d353"

def save(fig, name):
    path = os.path.join(SAVE_DIR, name)
    fig.savefig(path, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  OK  {name}")


# ═══════════════════════════════════════════════════════════════
# DIAGRAM 1 — Full Pipeline Flow
# ═══════════════════════════════════════════════════════════════
def diagram_pipeline():
    fig, ax = plt.subplots(figsize=(18, 9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_title("Pneumonia Detection — Full System Pipeline",
                 color=WHITE, fontsize=16, fontweight="bold", pad=14)

    def box(x, y, w, h, label, sublabel="", color=BLUE, fontsize=10):
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.12",
                              linewidth=1.5,
                              edgecolor=color,
                              facecolor=PANEL)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + (0.18 if sublabel else 0),
                label, ha="center", va="center",
                color=color, fontsize=fontsize, fontweight="bold")
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.22,
                    sublabel, ha="center", va="center",
                    color=WHITE, fontsize=7.5, alpha=0.8)

    def arrow(x1, y1, x2, y2, color=WHITE):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=1.8, mutation_scale=14))

    # Row 1 — Data
    box(0.3, 6.8, 2.4, 1.0, "Kaggle Dataset", "5,863 X-Ray JPEGs", CYAN)
    arrow(2.7, 7.3, 3.2, 7.3)
    box(3.2, 6.8, 2.4, 1.0, "Train / Val / Test", "5216 / 16 / 624", BLUE)
    arrow(5.6, 7.3, 6.1, 7.3)
    box(6.1, 6.8, 2.5, 1.0, "Augmentation", "Flip·Rotate·Jitter", PURPLE)
    arrow(8.6, 7.3, 9.1, 7.3)
    box(9.1, 6.8, 2.5, 1.0, "Normalize", "ImageNet μ/σ", PURPLE)
    arrow(11.6, 7.3, 12.1, 7.3)
    box(12.1, 6.8, 2.5, 1.0, "Weighted Sampler", "Balance 1:2.89", ORANGE)
    arrow(14.6, 7.3, 15.1, 7.3)
    box(15.1, 6.8, 2.5, 1.0, "DataLoader", "Batch = 32", BLUE)

    # Row 2 — Model
    box(0.3, 4.9, 2.4, 1.0, "DenseNet121", "Pretrained ImageNet", GREEN)
    arrow(2.7, 5.4, 3.2, 5.4)
    box(3.2, 4.9, 2.4, 1.0, "Dense Block 1–4", "Feature Extraction", GREEN)
    arrow(5.6, 5.4, 6.1, 5.4)
    box(6.1, 4.9, 2.5, 1.0, "Global AvgPool", "7×7×1024 → 1024", GREEN)
    arrow(8.6, 5.4, 9.1, 5.4)
    box(9.1, 4.9, 2.5, 1.0, "Dropout 0.5", "Regularization", YELLOW)
    arrow(11.6, 5.4, 12.1, 5.4)
    box(12.1, 4.9, 2.5, 1.0, "FC Layer", "1024 → 2 logits", GREEN)
    arrow(14.6, 5.4, 15.1, 5.4)
    box(15.1, 4.9, 2.5, 1.0, "Softmax", "P(Normal), P(Pneumonia)", BLUE)

    # Row 3 — Training
    box(0.3, 3.0, 2.4, 1.0, "Weighted CE Loss", "w=[1.95, 0.67]", RED)
    arrow(2.7, 3.5, 3.2, 3.5)
    box(3.2, 3.0, 2.4, 1.0, "Adam Optimizer", "lr=1e-4, wd=1e-4", ORANGE)
    arrow(5.6, 3.5, 6.1, 3.5)
    box(6.1, 3.0, 2.5, 1.0, "StepLR Scheduler", "γ=0.1, step=7", ORANGE)
    arrow(8.6, 3.5, 9.1, 3.5)
    box(9.1, 3.0, 2.5, 1.0, "Early Stopping", "patience=5", YELLOW)
    arrow(11.6, 3.5, 12.1, 3.5)
    box(12.1, 3.0, 2.5, 1.0, "Checkpoint", "Best val_loss model", CYAN)
    arrow(14.6, 3.5, 15.1, 3.5)
    box(15.1, 3.0, 2.5, 1.0, "best_model.pth", "Saved weights", CYAN)

    # Row 4 — Output
    box(0.3, 1.1, 2.4, 1.0, "Evaluation", "Accuracy·AUC·F1", BLUE)
    arrow(2.7, 1.6, 3.2, 1.6)
    box(3.2, 1.1, 2.4, 1.0, "Confusion Matrix", "TP/TN/FP/FN", PURPLE)
    arrow(5.6, 1.6, 6.1, 1.6)
    box(6.1, 1.1, 2.5, 1.0, "ROC Curve", "AUC-ROC score", PURPLE)
    arrow(8.6, 1.6, 9.1, 1.6)
    box(9.1, 1.1, 2.5, 1.0, "Grad-CAM", "Heatmap visualization", GREEN)
    arrow(11.6, 1.6, 12.1, 1.6)
    box(12.1, 1.1, 2.5, 1.0, "Triage System", "3-level clinical output", RED)
    arrow(14.6, 1.6, 15.1, 1.6)
    box(15.1, 1.1, 2.5, 1.0, "JSON Report", "Priority-sorted cases", CYAN)

    # Vertical connectors
    for x_c in [1.5, 4.4, 7.35, 10.35, 13.35, 16.35]:
        arrow(x_c, 6.8, x_c, 5.9, WHITE)
        arrow(x_c, 4.9, x_c, 4.0, WHITE)
        arrow(x_c, 3.0, x_c, 2.1, WHITE)

    # Row labels
    for y, label, color in [(7.3,"DATA LAYER",CYAN),
                             (5.4,"MODEL LAYER",GREEN),
                             (3.5,"TRAINING LAYER",ORANGE),
                             (1.6,"OUTPUT LAYER",PURPLE)]:
        ax.text(0.05, y, label, color=color, fontsize=7.5,
                fontweight="bold", va="center", rotation=0, alpha=0.7)

    save(fig, "01_pipeline_flow.png")


# ═══════════════════════════════════════════════════════════════
# DIAGRAM 2 — DenseNet121 Architecture
# ═══════════════════════════════════════════════════════════════
def diagram_densenet():
    fig, ax = plt.subplots(figsize=(18, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("DenseNet121 Architecture for Pneumonia Detection",
                 color=WHITE, fontsize=15, fontweight="bold", pad=12)

    blocks = [
        (0.2,  2.0, "Input\n224×224×3",    CYAN,   0.9),
        (1.5,  2.0, "Conv\n7×7, s=2\n112×112×64", BLUE, 1.1),
        (3.0,  2.0, "MaxPool\n3×3, s=2\n56×56×64", BLUE, 1.1),
        (4.6,  1.5, "Dense\nBlock 1\n6 layers\n56×56×256", GREEN, 1.2),
        (6.3,  1.5, "Trans.\nLayer 1\n28×28×128", YELLOW, 1.1),
        (7.8,  1.5, "Dense\nBlock 2\n12 layers\n28×28×512", GREEN, 1.2),
        (9.5,  1.5, "Trans.\nLayer 2\n14×14×256", YELLOW, 1.1),
        (11.0, 1.5, "Dense\nBlock 3\n24 layers\n14×14×1024", GREEN, 1.2),
        (12.7, 1.5, "Trans.\nLayer 3\n7×7×512", YELLOW, 1.1),
        (14.2, 1.5, "Dense\nBlock 4\n16 layers\n7×7×1024", GREEN, 1.2),
        (15.9, 2.0, "Global\nAvgPool\n1×1×1024", BLUE,  1.1),
        (17.2, 2.0, "Head\n→2", RED,    0.8),
    ]

    prev_x = None
    for (x, y, label, color, w) in blocks:
        h = 2.5
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.08",
                              linewidth=1.8, edgecolor=color, facecolor=PANEL)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                color=color, fontsize=7.5, fontweight="bold",
                multialignment="center")
        if prev_x is not None:
            ax.annotate("", xy=(x, y + h/2),
                        xytext=(prev_x, y + h/2),
                        arrowprops=dict(arrowstyle="-|>", color=WHITE,
                                        lw=1.5, mutation_scale=12))
        prev_x = x + w

    # Dense connectivity illustration
    ax.text(5.2, 0.5, "Dense connectivity: each layer receives ALL previous feature maps",
            ha="center", color=PURPLE, fontsize=9, style="italic")
    ax.annotate("", xy=(7.2, 0.9), xytext=(5.2, 0.9),
                arrowprops=dict(arrowstyle="<->", color=PURPLE, lw=1.5))
    ax.text(6.2, 1.05, "x_l = H_l([x_0, x_1, ..., x_{l-1}])",
            ha="center", color=PURPLE, fontsize=9, family="monospace")

    # Classifier head detail
    ax.text(17.6, 5.5, "Classifier Head", color=RED,
            fontsize=9, fontweight="bold", ha="center")
    for i, (lbl, clr) in enumerate([
        ("Dropout\np=0.5", YELLOW),
        ("Linear\n1024→2", RED),
        ("Softmax\nP(N),P(P)", BLUE),
    ]):
        bx = 16.8 + i * 0.05
        by = 4.0 + i * 0.4
        rect = FancyBboxPatch((16.4, 4.8 - i*0.85), 1.2, 0.7,
                              boxstyle="round,pad=0.05",
                              linewidth=1.2, edgecolor=clr, facecolor=PANEL)
        ax.add_patch(rect)
        ax.text(17.0, 4.8 - i*0.85 + 0.35, lbl, ha="center", va="center",
                color=clr, fontsize=7, fontweight="bold")
        if i > 0:
            ax.annotate("", xy=(17.0, 4.8 - (i-1)*0.85),
                        xytext=(17.0, 4.8 - (i-1)*0.85 + 0.7),
                        arrowprops=dict(arrowstyle="-|>", color=WHITE, lw=1))

    save(fig, "02_densenet_architecture.png")


# ═══════════════════════════════════════════════════════════════
# DIAGRAM 3 — Loss Equations
# ═══════════════════════════════════════════════════════════════
def diagram_loss():
    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor(BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.set_title("Loss Functions & Optimization Equations",
                 color=WHITE, fontsize=15, fontweight="bold", pad=10)

    def panel(x, y, w, h, title, color):
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.1",
                              linewidth=2, edgecolor=color, facecolor=PANEL,
                              transform=ax.transAxes, zorder=1)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.02, title, ha="center", va="top",
                color=color, fontsize=11, fontweight="bold",
                transform=ax.transAxes)

    def eq(x, y, text, color=WHITE, size=10):
        ax.text(x, y, text, ha="left", va="top", color=color,
                fontsize=size, transform=ax.transAxes,
                family="monospace")

    def note(x, y, text, color=WHITE, size=8.5):
        ax.text(x, y, text, ha="left", va="top", color=color,
                fontsize=size, transform=ax.transAxes, alpha=0.85)

    # ── Panel 1: Standard CE ─────────────────
    panel(0.01, 0.70, 0.30, 0.26, "① Standard Cross-Entropy", BLUE)
    eq(0.02, 0.91, "L_CE = -(1/N) Σᵢ  Σₖ  yᵢₖ · log(p̂ᵢₖ)", BLUE, 9.5)
    note(0.02, 0.85, "where:", WHITE)
    note(0.02, 0.81, "  N   = number of samples")
    note(0.02, 0.78, "  yᵢₖ = 1 if sample i has true class k")
    note(0.02, 0.75, "  p̂ᵢₖ = predicted probability for class k")
    note(0.02, 0.72, "  [X] Ignores class imbalance", RED)

    # ── Panel 2: Weighted CE ─────────────────
    panel(0.34, 0.70, 0.32, 0.26, "② Weighted Cross-Entropy (Used)", GREEN)
    eq(0.35, 0.91, "L_WCE = -(1/N) Σᵢ  wᵧᵢ · log(p̂ᵢ,yᵢ)", GREEN, 9.5)
    note(0.35, 0.85, "Class weights computed as:", WHITE)
    eq(0.35, 0.81,  "  wₖ = N_total / (C × Nₖ)", YELLOW, 9)
    note(0.35, 0.77, "  N_total = 5216,  C = 2 classes")
    note(0.35, 0.74, "  w_NORMAL    = 5216/(2×1341) = 1.945")
    note(0.35, 0.71, "  w_PNEUMONIA = 5216/(2×3875) = 0.673")

    # ── Panel 3: Softmax ─────────────────────
    panel(0.69, 0.70, 0.30, 0.26, "③ Softmax Output Layer", PURPLE)
    eq(0.70, 0.91, "P(y=k | x) = exp(zₖ) / Σⱼ exp(zⱼ)", PURPLE, 9.5)
    note(0.70, 0.85, "where:")
    note(0.70, 0.81, "  zₖ = raw logit for class k")
    note(0.70, 0.78, "  Σⱼ exp(zⱼ) = normalisation constant")
    note(0.70, 0.75, "  Output: [P(NORMAL), P(PNEUMONIA)]")
    note(0.70, 0.72, "  Sum = 1.0 always guaranteed", GREEN)

    # ── Panel 4: Adam ────────────────────────
    panel(0.01, 0.38, 0.45, 0.28, "④ Adam Optimizer Update Rule", ORANGE)
    eq(0.02, 0.61, "gₜ   = ∇_θ L(θₜ₋₁)            [gradient]", ORANGE, 9)
    eq(0.02, 0.57, "mₜ   = β₁·mₜ₋₁ + (1-β₁)·gₜ   [1st moment]", ORANGE, 9)
    eq(0.02, 0.53, "vₜ   = β₂·vₜ₋₁ + (1-β₂)·gₜ²  [2nd moment]", ORANGE, 9)
    eq(0.02, 0.49, "m̂ₜ   = mₜ/(1-β₁ᵗ)              [bias correct]", YELLOW, 9)
    eq(0.02, 0.45, "v̂ₜ   = vₜ/(1-β₂ᵗ)              [bias correct]", YELLOW, 9)
    eq(0.02, 0.41, "θₜ   = θₜ₋₁ - α·m̂ₜ/(√v̂ₜ+ε)   [update]", GREEN, 9)
    note(0.28, 0.61, "α  = 1×10⁻⁴  (learning rate)")
    note(0.28, 0.57, "β₁ = 0.9     (momentum)")
    note(0.28, 0.53, "β₂ = 0.999   (RMS factor)")
    note(0.28, 0.49, "ε  = 1×10⁻⁸  (stability)")
    note(0.28, 0.45, "wd = 1×10⁻⁴  (weight decay)")

    # ── Panel 5: StepLR ──────────────────────
    panel(0.49, 0.38, 0.24, 0.28, "⑤ LR Scheduler (StepLR)", YELLOW)
    eq(0.50, 0.61, "αₜ = α₀ × γ^⌊t/S⌋", YELLOW, 9.5)
    note(0.50, 0.55, "where:")
    note(0.50, 0.51, "  α₀ = 1×10⁻⁴  (initial LR)")
    note(0.50, 0.47, "  γ  = 0.1      (decay factor)")
    note(0.50, 0.43, "  S  = 7        (step size)")
    note(0.50, 0.39, "Epoch 7:  α = 1×10⁻⁵")

    # ── Panel 6: GradCAM ─────────────────────
    panel(0.76, 0.38, 0.23, 0.28, "⑥ Grad-CAM Equation", CYAN)
    eq(0.77, 0.61, "αᵏc = (1/Z)ΣᵢΣⱼ ∂yᶜ/∂Aᵏᵢⱼ", CYAN, 9)
    eq(0.77, 0.55, "Lᶜ = ReLU(Σₖ αᵏc · Aᵏ)", CYAN, 9)
    note(0.77, 0.50, "where:")
    note(0.77, 0.46, "  Aᵏ = activation map k")
    note(0.77, 0.42, "  yᶜ = class score (pre-softmax)")
    note(0.77, 0.38, "  ReLU keeps positive only", GREEN)

    # ── Panel 7: Metrics ─────────────────────
    panel(0.01, 0.01, 0.97, 0.33, "⑦ Evaluation Metrics", PURPLE)
    cols = [
        ("Accuracy",    "ACC = (TP+TN)/(TP+TN+FP+FN)",         BLUE),
        ("Sensitivity", "SEN = TP/(TP+FN)  [Recall-Pneumonia]", GREEN),
        ("Specificity", "SPE = TN/(TN+FP)  [Recall-Normal]",    CYAN),
        ("Precision",   "PRE = TP/(TP+FP)",                     YELLOW),
        ("F1-Score",    "F1  = 2·PRE·SEN/(PRE+SEN)",            ORANGE),
        ("AUC-ROC",     "Area under ROC curve [0,1]",           PURPLE),
    ]
    for i, (name, formula, clr) in enumerate(cols):
        cx = 0.02 + i * 0.163
        ax.text(cx, 0.30, name, color=clr, fontsize=9,
                fontweight="bold", transform=ax.transAxes)
        ax.text(cx, 0.26, formula, color=WHITE, fontsize=7.5,
                transform=ax.transAxes, family="monospace")

    note(0.02, 0.18, "Confusion Matrix:", WHITE, 9)
    note(0.02, 0.14, "        Predicted NORMAL    Predicted PNEUMONIA")
    note(0.02, 0.10, "True NORMAL       TN (correct)      FP (false alarm)")
    note(0.02, 0.06, "True PNEUMONIA    FN (missed!) !    TP (correct)")
    note(0.55, 0.14, "! False Negatives (FN) are the most dangerous error.", RED, 9)
    note(0.55, 0.10, "  A missed pneumonia case can be fatal.", RED, 8.5)
    note(0.55, 0.06, "  Weighted loss minimises FN by penalising NORMAL mistakes.", GREEN, 8.5)

    save(fig, "03_loss_equations.png")


# ═══════════════════════════════════════════════════════════════
# DIAGRAM 4 — Data Availability
# ═══════════════════════════════════════════════════════════════
def diagram_data():
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor(BG)
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── 4A: Split bar chart ──────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(PANEL)
    splits   = ["Train", "Val", "Test"]
    normals  = [1341, 8, 234]
    pneumos  = [3875, 8, 390]
    x = np.arange(3)
    b1 = ax1.bar(x - 0.2, normals, 0.38, label="NORMAL",   color=GREEN, edgecolor=BG)
    b2 = ax1.bar(x + 0.2, pneumos, 0.38, label="PNEUMONIA", color=RED,   edgecolor=BG)
    ax1.set_xticks(x); ax1.set_xticklabels(splits, color=WHITE)
    ax1.set_title("Images per Split", color=WHITE, fontsize=11, fontweight="bold")
    ax1.set_ylabel("Count", color=WHITE); ax1.tick_params(colors=WHITE)
    ax1.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=WHITE, fontsize=8)
    ax1.set_facecolor(PANEL)
    for spine in ax1.spines.values(): spine.set_edgecolor(BORDER)
    for bar, v in [(b1, normals), (b2, pneumos)]:
        for b, val in zip(bar, v):
            ax1.text(b.get_x()+b.get_width()/2, b.get_height()+20,
                     str(val), ha="center", color=WHITE, fontsize=8, fontweight="bold")

    # ── 4B: Class imbalance pie (train) ──────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(PANEL)
    wedges, texts, autos = ax2.pie(
        [1341, 3875], labels=["NORMAL\n1341", "PNEUMONIA\n3875"],
        colors=[GREEN, RED], autopct="%1.1f%%", startangle=90,
        textprops={"color": WHITE, "fontsize": 9},
        wedgeprops={"edgecolor": BG, "linewidth": 2},
    )
    for at in autos: at.set_fontsize(10); at.set_fontweight("bold")
    ax2.set_title("Train Set Class Balance", color=WHITE, fontsize=11, fontweight="bold")
    ax2.text(0, -1.45, "Imbalance Ratio  1 : 2.89",
             ha="center", color=YELLOW, fontsize=9, fontweight="bold")

    # ── 4C: Weighted sampler effect ──────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(PANEL)
    n_vals = [1341/(1341+3875)*32, 16]
    p_vals = [3875/(1341+3875)*32, 16]
    x3 = np.arange(2)
    ax3.bar(x3-0.2, n_vals, 0.38, label="NORMAL",    color=GREEN, edgecolor=BG)
    ax3.bar(x3+0.2, p_vals, 0.38, label="PNEUMONIA",  color=RED,   edgecolor=BG)
    ax3.axhline(16, color=YELLOW, linestyle="--", lw=1.2, label="Ideal (16 each)")
    ax3.set_xticks(x3)
    ax3.set_xticklabels(["Without\nSampler", "With Weighted\nSampler"],
                        color=WHITE, fontsize=9)
    ax3.set_title("Expected NORMAL/PNEUMONIA\nper Batch (size=32)", color=WHITE,
                  fontsize=10, fontweight="bold")
    ax3.set_ylabel("Avg samples", color=WHITE); ax3.tick_params(colors=WHITE)
    ax3.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=WHITE, fontsize=8)
    for spine in ax3.spines.values(): spine.set_edgecolor(BORDER)

    # ── 4D: Image size distribution ──────────
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor(PANEL)
    np.random.seed(42)
    widths  = np.random.normal(1250, 320, 500).clip(400, 2500)
    heights = np.random.normal(1020, 280, 500).clip(300, 2000)
    ax4.hist(widths,  bins=25, color=BLUE,   alpha=0.7, label="Width",  edgecolor=BG)
    ax4.hist(heights, bins=25, color=PURPLE, alpha=0.7, label="Height", edgecolor=BG)
    ax4.axvline(224, color=YELLOW, lw=2, linestyle="--", label="Model input: 224px")
    ax4.set_title("Raw Image Size Distribution", color=WHITE, fontsize=11, fontweight="bold")
    ax4.set_xlabel("Pixels", color=WHITE); ax4.set_ylabel("Count", color=WHITE)
    ax4.tick_params(colors=WHITE)
    ax4.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=WHITE, fontsize=8)
    for spine in ax4.spines.values(): spine.set_edgecolor(BORDER)

    # ── 4E: Augmentation pipeline ────────────
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_facecolor(PANEL); ax5.axis("off")
    ax5.set_title("Augmentation Pipeline", color=WHITE, fontsize=11, fontweight="bold")
    steps = [
        ("Resize 256×256",             "All splits",        BLUE),
        ("RandomCrop 224×224",         "Train only",        GREEN),
        ("RandomHorizontalFlip p=0.5", "Train only",        GREEN),
        ("RandomRotation ±10°",        "Train only OK valid",GREEN),
        ("ColorJitter b=0.2 c=0.2",    "Train only",        GREEN),
        ("RandomAffine shear=5°",      "Train only",        GREEN),
        ("ToTensor",                   "All splits",        BLUE),
        ("Normalize ImageNet μ/σ",     "All splits",        BLUE),
        ("[X] NO Vertical Flip",        "Not clinically valid", RED),
    ]
    for i, (step, note_, clr) in enumerate(steps):
        y = 0.92 - i*0.098
        rect = FancyBboxPatch((0.01, y-0.035), 0.97, 0.075,
                              boxstyle="round,pad=0.01",
                              linewidth=1, edgecolor=clr, facecolor=BG,
                              transform=ax5.transAxes)
        ax5.add_patch(rect)
        ax5.text(0.04, y+0.002, step,  color=clr, fontsize=8,
                 fontweight="bold", transform=ax5.transAxes, va="center")
        ax5.text(0.65, y+0.002, note_, color=WHITE, fontsize=7,
                 transform=ax5.transAxes, va="center", alpha=0.8)

    # ── 4F: Dataset source info ──────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor(PANEL); ax6.axis("off")
    ax6.set_title("Dataset Information", color=WHITE, fontsize=11, fontweight="bold")
    info = [
        ("Source",     "Kermany et al., Cell 2018",          CYAN),
        ("Host",       "Kaggle (public, free)",               CYAN),
        ("Patients",   "Pediatric, age 1–5 years",           WHITE),
        ("Hospital",   "Guangzhou Women & Children's MC",    WHITE),
        ("Format",     "JPEG, variable resolution",           WHITE),
        ("Labels",     "Validated by 2 expert physicians",   GREEN),
        ("NORMAL",     "1583 total (341 val+test)",           GREEN),
        ("PNEUMONIA",  "4273 total (398 val+test)",           RED),
        ("Total",      "5863 chest X-ray images",            YELLOW),
        ("License",    "CC BY 4.0 — Free for research",      BLUE),
    ]
    for i, (key, val, clr) in enumerate(info):
        y = 0.90 - i * 0.088
        ax6.text(0.02, y, f"{key}:", color=clr, fontsize=8.5,
                 fontweight="bold", transform=ax6.transAxes)
        ax6.text(0.35, y, val, color=WHITE, fontsize=8.5,
                 transform=ax6.transAxes)

    fig.suptitle("Data Availability & Distribution Analysis",
                 color=WHITE, fontsize=15, fontweight="bold", y=1.01)
    save(fig, "04_data_availability.png")


# ═══════════════════════════════════════════════════════════════
# DIAGRAM 5 — Training Pipeline Detail
# ═══════════════════════════════════════════════════════════════
def diagram_training():
    fig, ax = plt.subplots(figsize=(16, 12))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 16); ax.set_ylim(0, 12)
    ax.axis("off")
    ax.set_title("Detailed Training Pipeline with Early Stopping & Checkpointing",
                 color=WHITE, fontsize=14, fontweight="bold", pad=12)

    def box(x, y, w, h, text, color=BLUE, shape="round"):
        style = f"{shape},pad=0.1" if shape != "round" else "round,pad=0.1"
        r = FancyBboxPatch((x, y), w, h, boxstyle=style,
                           linewidth=2, edgecolor=color, facecolor=PANEL)
        ax.add_patch(r)
        ax.text(x+w/2, y+h/2, text, ha="center", va="center",
                color=color, fontsize=8.5, fontweight="bold",
                multialignment="center")

    def diamond(x, y, w, h, text, color=YELLOW):
        pts = np.array([[x+w/2,y+h],[x+w,y+h/2],[x+w/2,y],[x,y+h/2]])
        poly = plt.Polygon(pts, closed=True, linewidth=2,
                           edgecolor=color, facecolor=PANEL)
        ax.add_patch(poly)
        ax.text(x+w/2, y+h/2, text, ha="center", va="center",
                color=color, fontsize=8, fontweight="bold",
                multialignment="center")

    def arr(x1,y1,x2,y2,label="",color=WHITE):
        ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                    arrowprops=dict(arrowstyle="-|>",color=color,lw=1.8,mutation_scale=13))
        if label:
            mx,my=(x1+x2)/2,(y1+y2)/2
            ax.text(mx+0.1,my,label,color=color,fontsize=7.5,va="center")

    # Start
    box(6.5,11.0,3.0,0.7,"START: Load Config\nset_seed(42)",CYAN)
    arr(8.0,11.0,8.0,10.5)
    box(6.5,9.7,3.0,0.7,"Build DataLoaders\nWeightedRandomSampler",BLUE)
    arr(8.0,9.7,8.0,9.2)
    box(6.5,8.5,3.0,0.6,"Build DenseNet121\nLoad ImageNet weights",GREEN)
    arr(8.0,8.5,8.0,8.0)
    box(6.5,7.3,3.0,0.6,"Define Loss, Adam,\nStepLR, EarlyStopping",ORANGE)
    arr(8.0,7.3,8.0,6.8)

    # Epoch loop
    box(5.5,6.2,5.0,0.55,"FOR epoch = 1 … MAX_EPOCHS",PURPLE)
    arr(8.0,6.2,8.0,5.7)

    # Train phase
    box(5.5,5.1,5.0,0.55,"TRAIN PHASE: model.train()",BLUE)
    arr(8.0,5.1,8.0,4.6)
    box(5.5,4.0,5.0,0.55,"For each batch: Forward → Loss\n→ Backward → optimizer.step()",BLUE)
    arr(8.0,4.0,8.0,3.5)

    # Val phase
    box(5.5,2.9,5.0,0.55,"VAL PHASE: model.eval()\ntorch.no_grad()",CYAN)
    arr(8.0,2.9,8.0,2.4)
    box(5.5,1.8,5.0,0.55,"Compute val_loss, val_acc",CYAN)
    arr(8.0,1.8,8.0,1.3)
    box(5.5,0.7,5.0,0.55,"scheduler.step()  [StepLR]",YELLOW)

    # Checkpoint branch
    diamond(1.0,4.5,3.0,1.2,"val_loss <\nbest_val_loss?",GREEN)
    arr(5.5,5.35,4.0,5.1)
    arr(4.0,5.1,2.5,5.1)
    arr(2.5,5.1,2.5,4.5)
    ax.text(3.2,5.25,"check",color=WHITE,fontsize=7.5)
    arr(2.5,4.5,2.5,3.3,"YES",GREEN)
    box(1.0,2.7,3.0,0.55,"save_checkpoint()\nbest_model.pth",CYAN)
    arr(4.0,4.5,5.0,4.5,"NO →",ORANGE)

    # Early stopping branch
    diamond(11.0,4.5,3.5,1.2,"Early Stop\ncounter ≥\npatience=5?",RED)
    arr(10.5,5.35,12.75,5.35)
    arr(12.75,5.35,12.75,4.5)
    arr(12.75,3.7,12.75,3.3,"YES",RED)
    box(11.0,2.7,3.5,0.55,"STOP TRAINING\nBest model saved",RED)
    arr(11.0,4.5,10.5,4.5)
    ax.text(10.55,4.65,"NO →",color=ORANGE,fontsize=7.5)

    # Loop back
    ax.annotate("", xy=(8.0,6.2), xytext=(8.0,0.7),
                arrowprops=dict(arrowstyle="-|>",color=PURPLE,lw=1.5,
                                connectionstyle="arc3,rad=-0.4"))
    ax.text(11.8,3.5,"Loop back\nnext epoch",color=PURPLE,fontsize=7.5)

    # Legend
    legend_items = [("Data ops",BLUE),("Model ops",GREEN),
                    ("Loss/Optim",ORANGE),("Checkpoint",CYAN),
                    ("Decision",YELLOW),("Stop",RED)]
    for i,(lbl,clr) in enumerate(legend_items):
        lx = 0.3 + i*2.65
        rect = FancyBboxPatch((lx, 0.08), 0.28, 0.28,
                              boxstyle="round,pad=0.04",
                              linewidth=1.5, edgecolor=clr, facecolor=PANEL)
        ax.add_patch(rect)
        ax.text(lx+0.42, 0.22, lbl, color=clr, fontsize=8, va="center")

    save(fig, "05_training_pipeline.png")


# ═══════════════════════════════════════════════════════════════
# DIAGRAM 6 — Triage System Flow
# ═══════════════════════════════════════════════════════════════
def diagram_triage():
    fig, ax = plt.subplots(figsize=(16, 11))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 16); ax.set_ylim(0, 11)
    ax.axis("off")
    ax.set_title("Confidence-Calibrated Triage System — Novel Contribution",
                 color=WHITE, fontsize=14, fontweight="bold", pad=12)

    def box(x, y, w, h, text, color=BLUE):
        r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                           linewidth=2, edgecolor=color, facecolor=PANEL)
        ax.add_patch(r)
        ax.text(x+w/2, y+h/2, text, ha="center", va="center",
                color=color, fontsize=9, fontweight="bold",
                multialignment="center")

    def diamond(x, y, w, h, text, color=YELLOW):
        pts = np.array([[x+w/2,y+h],[x+w,y+h/2],[x+w/2,y],[x,y+h/2]])
        poly = plt.Polygon(pts, closed=True, linewidth=2,
                           edgecolor=color, facecolor=PANEL)
        ax.add_patch(poly)
        ax.text(x+w/2, y+h/2, text, ha="center", va="center",
                color=color, fontsize=8.5, fontweight="bold",
                multialignment="center")

    def arr(x1, y1, x2, y2, label="", color=WHITE, lw=1.8):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=lw, mutation_scale=13))
        if label:
            ax.text((x1+x2)/2+0.1, (y1+y2)/2+0.05, label,
                    color=color, fontsize=8, fontweight="bold")

    # Input
    box(6.0, 9.8, 4.0, 0.75, "Chest X-Ray Image Input", CYAN)
    arr(8.0, 9.8, 8.0, 9.2)

    # Preprocessing
    box(6.0, 8.5, 4.0, 0.65, "Preprocess: Resize → Normalize\n224×224, ImageNet μ/σ", BLUE)
    arr(8.0, 8.5, 8.0, 7.9)

    # Model
    box(6.0, 7.2, 4.0, 0.65, "DenseNet121 Forward Pass\nLogits → Softmax", GREEN)
    arr(8.0, 7.2, 8.0, 6.6)

    # Probability output
    box(5.5, 6.0, 5.0, 0.55, "P(Pneumonia) ∈ [0.0, 1.0]", PURPLE)
    arr(8.0, 6.0, 8.0, 5.4)

    # Decision diamond 1
    diamond(6.2, 4.3, 3.6, 1.05, "P(Pneumonia)\n≥ 0.80?", RED)
    arr(8.0, 5.4, 8.0, 5.35)

    # YES → HIGH RISK
    arr(9.8, 4.82, 12.5, 4.82, "YES", RED, 2.0)
    box(12.5, 4.35, 3.2, 0.95, "[R]  HIGH RISK\nP ≥ 80%\nURGENT treatment", RED)

    # NO → second diamond
    arr(8.0, 4.3, 8.0, 3.5, "NO", YELLOW)
    diamond(6.2, 2.45, 3.6, 1.0, "P(Pneumonia)\n≤ 0.20?", GREEN)

    # YES → LOW RISK
    arr(9.8, 2.95, 12.5, 2.95, "YES", GREEN, 2.0)
    box(12.5, 2.55, 3.2, 0.85, "[G]  LOW RISK\nP ≤ 20%\nDeprioritize safely", GREEN)

    # NO → UNCERTAIN
    arr(8.0, 2.45, 8.0, 1.7, "NO", YELLOW)
    box(6.0, 1.1, 4.0, 0.55, "[Y]  UNCERTAIN\n20% < P < 80%\nRefer to radiologist", YELLOW)

    # All merge → output
    arr(12.5, 4.82, 15.2, 4.82, "", RED)
    arr(12.5, 2.95, 15.2, 2.95, "", GREEN)
    arr(8.0,  1.1,  8.0,  0.55, "", WHITE)
    box(6.5, 0.1, 3.0, 0.42, "JSON Report + Visual Card", CYAN)

    # Threshold band illustration on the right
    ax.text(0.3, 9.5, "Confidence\nThreshold\nBands", color=WHITE,
            fontsize=9, fontweight="bold", ha="center")

    band_data = [
        (0.80, 1.00, RED,    "HIGH\nRISK"),
        (0.20, 0.80, YELLOW, "UNCERTAIN\n(→ Radiologist)"),
        (0.00, 0.20, GREEN,  "LOW\nRISK"),
    ]
    bar_x, bar_w = 0.1, 0.4
    for lo, hi, clr, lbl in band_data:
        rect = FancyBboxPatch((bar_x, lo*8.0+0.8), bar_w, (hi-lo)*8.0,
                              boxstyle="square,pad=0",
                              linewidth=1.5, edgecolor=clr,
                              facecolor=clr, alpha=0.25)
        ax.add_patch(rect)
        ax.text(bar_x + bar_w + 0.15,
                (lo + (hi-lo)/2)*8.0 + 0.8,
                lbl, color=clr, fontsize=8,
                fontweight="bold", va="center")

    for val, lbl in [(0.0,"0%"),(0.2,"20%"),(0.8,"80%"),(1.0,"100%")]:
        y = val*8.0 + 0.8
        ax.axhline(y, xmin=0.006, xmax=0.06,
                   color=WHITE, lw=0.8, linestyle="--", alpha=0.4)
        ax.text(bar_x + bar_w + 0.08, y, lbl,
                color=WHITE, fontsize=7.5, va="center", alpha=0.7)

    ax.text(0.3, 0.45, "P(Pneumonia)", color=WHITE,
            fontsize=8, ha="center", rotation=90, va="bottom")

    # Tunable note
    ax.text(1.2, 0.25,
            "*  Thresholds are tunable per deployment context\n"
            "    e.g. ICU: high_t=0.70  |  Outpatient: high_t=0.90",
            color=YELLOW, fontsize=8.5, va="bottom")

    save(fig, "06_triage_system.png")


# ═══════════════════════════════════════════════════════════════
# DIAGRAM 7 — Dense Block Connectivity
# ═══════════════════════════════════════════════════════════════
def diagram_dense_block():
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Dense Block Connectivity vs. Standard CNN",
                 color=WHITE, fontsize=14, fontweight="bold")

    colors_layers = [BLUE, GREEN, PURPLE, ORANGE, CYAN]

    for idx, (ax, title, dense) in enumerate(zip(
        axes,
        ["Standard CNN (ResNet-style)", "Dense Block (DenseNet-style)"],
        [False, True]
    )):
        ax.set_facecolor(PANEL)
        ax.set_xlim(-0.5, 5.5)
        ax.set_ylim(-0.5, 5.0)
        ax.axis("off")
        ax.set_title(title, color=WHITE, fontsize=12, fontweight="bold", pad=8)

        n = 5
        positions = [(i*1.0, 2.0) for i in range(n)]

        # Draw connections first (behind nodes)
        if dense:
            for i in range(n):
                for j in range(i+1, n):
                    x1,y1 = positions[i]
                    x2,y2 = positions[j]
                    curve = 0.4 * (j - i)
                    ax.annotate("",
                        xy=(x2-0.18, y2),
                        xytext=(x1+0.18, y1),
                        arrowprops=dict(
                            arrowstyle="-|>",
                            color=colors_layers[i],
                            lw=1.2,
                            connectionstyle=f"arc3,rad={-curve*0.25}",
                            mutation_scale=10,
                            alpha=0.6,
                        ))
        else:
            for i in range(n-1):
                x1,_ = positions[i]
                x2,_ = positions[i+1]
                ax.annotate("",
                    xy=(x2-0.18, 2.0),
                    xytext=(x1+0.18, 2.0),
                    arrowprops=dict(arrowstyle="-|>",
                                   color=colors_layers[i],
                                   lw=1.8, mutation_scale=12))

        # Draw nodes
        for i,(x,y) in enumerate(positions):
            circle = plt.Circle((x, y), 0.18, color=colors_layers[i],
                                 zorder=3, linewidth=2)
            ax.add_patch(circle)
            ax.text(x, y, f"L{i+1}", ha="center", va="center",
                    color=BG, fontsize=9, fontweight="bold", zorder=4)
            ax.text(x, y-0.42, f"Layer {i+1}",
                    ha="center", color=colors_layers[i], fontsize=8)

        # Connection count
        if dense:
            n_conn = n*(n-1)//2
            formula = f"L(L+1)/2 = {n}×{n+1}//2 = {n_conn} connections"
            clr = GREEN
        else:
            n_conn = n - 1
            formula = f"L-1 = {n_conn} connections"
            clr = YELLOW

        ax.text(2.5, 0.3, formula, ha="center",
                color=clr, fontsize=9, fontweight="bold")

        # Feature map concat illustration (dense only)
        if dense:
            ax.text(2.5, -0.1,
                    "x_l = H_l([x_0, x_1, ..., x_{l-1}])  — concatenation",
                    ha="center", color=PURPLE, fontsize=8, style="italic")
        else:
            ax.text(2.5, -0.1,
                    "x_l = H_l(x_{l-1})  — sequential only",
                    ha="center", color=YELLOW, fontsize=8, style="italic")

        # Growth rate annotation
        if dense:
            ax.text(2.5, 4.5, "Growth rate k=32 per layer\nEach layer adds 32 feature maps",
                    ha="center", color=CYAN, fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL,
                              edgecolor=CYAN, linewidth=1))

    plt.tight_layout()
    save(fig, "07_dense_block_connectivity.png")


# ═══════════════════════════════════════════════════════════════
# DIAGRAM 8 — Grad-CAM Process
# ═══════════════════════════════════════════════════════════════
def diagram_gradcam():
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 16); ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("Grad-CAM Explainability — Step by Step",
                 color=WHITE, fontsize=14, fontweight="bold", pad=12)

    def box(x, y, w, h, lines, color=BLUE):
        r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                           linewidth=2, edgecolor=color, facecolor=PANEL)
        ax.add_patch(r)
        if isinstance(lines, str):
            lines = [lines]
        step_h = h / (len(lines) + 0.5)
        for i, line in enumerate(lines):
            ax.text(x+w/2, y+h-(i+0.8)*step_h, line,
                    ha="center", va="center", color=color if i==0 else WHITE,
                    fontsize=8.5 if i==0 else 8,
                    fontweight="bold" if i==0 else "normal")

    def arr(x1, y1, x2, y2, label="", color=WHITE):
        ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=1.8, mutation_scale=13))
        if label:
            ax.text((x1+x2)/2+0.05,(y1+y2)/2+0.12,
                    label, color=color, fontsize=8)

    # Step boxes
    box(0.2, 5.5, 2.4, 2.2, [
        "① INPUT",
        "Chest X-Ray",
        "224×224×3",
        "Preprocessed",
    ], CYAN)

    arr(2.6, 6.6, 3.1, 6.6)

    box(3.1, 5.5, 2.6, 2.2, [
        "② FORWARD PASS",
        "DenseNet121",
        "Feature Maps Aᵏ",
        "Shape: 7×7×1024",
        "(saved via hook)",
    ], GREEN)

    arr(5.7, 6.6, 6.2, 6.6)

    box(6.2, 5.5, 2.5, 2.2, [
        "③ CLASS SCORE",
        "Target class c",
        "yᶜ = logit score",
        "(before softmax)",
    ], BLUE)

    arr(8.7, 6.6, 9.2, 6.6)

    box(9.2, 5.5, 2.5, 2.2, [
        "④ BACKWARD PASS",
        "∂yᶜ / ∂Aᵏᵢⱼ",
        "Gradients saved",
        "via backward hook",
    ], ORANGE)

    arr(11.7, 6.6, 12.2, 6.6)

    box(12.2, 5.5, 3.5, 2.2, [
        "⑤ WEIGHT GRADIENTS",
        "αᵏc = (1/Z)ΣᵢΣⱼ ∂yᶜ/∂Aᵏᵢⱼ",
        "Global avg pool over",
        "spatial dimensions (i,j)",
    ], PURPLE)

    # Row 2
    box(0.2, 2.5, 2.4, 2.2, [
        "⑧ OVERLAY",
        "Resize heatmap",
        "to 224×224",
        "JET colormap blend",
        "α = 0.4 opacity",
    ], CYAN)

    arr(2.6, 3.6, 3.1, 3.6, "", GREEN)

    box(3.1, 2.5, 2.6, 2.2, [
        "⑦ NORMALIZE",
        "H = H / max(H)",
        "Scale to [0, 1]",
        "Resize bilinear",
        "→ 224×224",
    ], GREEN)

    arr(5.7, 3.6, 6.2, 3.6)

    box(6.2, 2.5, 2.5, 2.2, [
        "⑥ WEIGHTED SUM",
        "Lᶜ = Σₖ αᵏc · Aᵏ",
        "then",
        "Lᶜ = ReLU(Lᶜ)",
        "Shape: 7×7",
    ], RED)

    # Connect row 1 to row 2
    arr(13.95, 5.5, 13.95, 4.0, "", PURPLE)
    ax.annotate("", xy=(8.7, 3.6), xytext=(13.95, 3.6),
                arrowprops=dict(arrowstyle="-|>", color=PURPLE,
                               lw=1.8, mutation_scale=13))
    ax.annotate("", xy=(6.2, 3.6), xytext=(8.7, 3.6),
                arrowprops=dict(arrowstyle="-|>", color=RED,
                               lw=1.8, mutation_scale=13))

    # Final output
    box(10.5, 2.5, 5.2, 2.2, [
        "OK  FINAL OUTPUT",
        "Original X-Ray  |  Heatmap  |  Overlay",
        "Blue = low attention",
        "Red  = high attention (pathological region)",
        "Clinician sees WHERE model focused",
    ], YELLOW)

    arr(2.6, 3.6, 10.5, 3.6, "", YELLOW)
    arr(0.2+2.4/2, 2.5, 0.2+2.4/2, 0.3+0.8, "", CYAN)
    ax.annotate("", xy=(10.5, 3.6), xytext=(2.6+2.4/2, 3.6),
                arrowprops=dict(arrowstyle="-|>", color=CYAN,
                               lw=2, mutation_scale=13))

    # ReLU explanation
    ax.text(7.5, 1.5,
            "ReLU ensures only regions that POSITIVELY contribute to the prediction are shown",
            ha="center", color=RED, fontsize=8.5, style="italic")
    ax.text(7.5, 1.1,
            "Negative gradients (regions that suppress pneumonia prediction) are ignored",
            ha="center", color=WHITE, fontsize=8, alpha=0.8)

    save(fig, "08_gradcam_process.png")


# ═══════════════════════════════════════════════════════════════
# MAIN — run all diagrams
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  Generating all project diagrams …")
    print("="*55)
    print(f"  Output folder: {SAVE_DIR}\n")

    diagram_pipeline()
    diagram_densenet()
    diagram_loss()
    diagram_data()
    diagram_training()
    diagram_triage()
    diagram_dense_block()
    diagram_gradcam()

    print("\n" + "="*55)
    print("  All 8 diagrams generated successfully!")
    print("="*55)
    print("\n  Files created:")
    for f in sorted(os.listdir(SAVE_DIR)):
        size = os.path.getsize(os.path.join(SAVE_DIR, f)) // 1024
        print(f"    {f}  ({size} KB)")
