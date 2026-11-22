"""
predict.py
----------
Single-image inference script.

Usage (CLI):
    python predict.py --image path/to/xray.jpeg

Usage (Python):
    from predict import predict_image
    result = predict_image("path/to/xray.jpeg")
    print(result)
"""

import argparse
import os
import json
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import config
from model import get_model
from utils import load_checkpoint
from dataset import get_transforms


# ─────────────────────────────────────────────
# Core prediction function
# ─────────────────────────────────────────────

@torch.no_grad()
def predict_image(image_path: str,
                  model=None,
                  show_plot: bool = True,
                  save_path: str = None) -> dict:
    """
    Run inference on a single chest X-ray image.

    Parameters
    ----------
    image_path : path to the image file
    model      : pre-loaded model (loads from checkpoint if None)
    show_plot  : display the prediction visualization
    save_path  : optional path to save the plot

    Returns
    -------
    dict with keys: predicted_class, confidence, probabilities
    """
    # ── Load model ──────────────────────────
    if model is None:
        model = get_model()
        model, _, _, _ = load_checkpoint(config.MODEL_PATH, model)
    model.to(config.DEVICE)
    model.eval()

    # ── Preprocess ──────────────────────────
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    transform    = get_transforms("test")
    pil_image    = Image.open(image_path).convert("RGB")
    input_tensor = transform(pil_image).unsqueeze(0).to(config.DEVICE)

    # ── Inference ───────────────────────────
    logits = model(input_tensor)
    probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred_idx   = int(np.argmax(probs))
    confidence = float(probs[pred_idx]) * 100

    predicted_class = config.CLASSES[pred_idx]
    probabilities   = {cls: float(p) * 100
                       for cls, p in zip(config.CLASSES, probs)}

    result = {
        "image_path":     image_path,
        "predicted_class": predicted_class,
        "confidence":      round(confidence, 2),
        "probabilities":   {k: round(v, 2) for k, v in probabilities.items()},
    }

    # ── Print ───────────────────────────────
    print("\n" + "─" * 40)
    print(f"  Image     : {os.path.basename(image_path)}")
    print(f"  Prediction: {predicted_class}")
    print(f"  Confidence: {confidence:.2f}%")
    print("  Class probabilities:")
    for cls, prob in probabilities.items():
        bar = "█" * int(prob / 5)
        print(f"    {cls:12s} {prob:5.1f}%  {bar}")
    print("─" * 40)

    # ── Visualize ───────────────────────────
    if show_plot or save_path:
        _plot_prediction(pil_image, result, save_path)

    return result


def _plot_prediction(pil_image: Image.Image, result: dict, save_path: str = None):
    """Create a clean prediction summary figure."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5),
                             gridspec_kw={"width_ratios": [1.5, 1]})
    fig.patch.set_facecolor("#1a1a2e")

    # ── Left: X-ray image ───────────────────
    axes[0].imshow(pil_image, cmap="gray")
    axes[0].set_title("Chest X-Ray", color="white", fontsize=13)
    axes[0].axis("off")

    pred   = result["predicted_class"]
    conf   = result["confidence"]
    colors = {"NORMAL": "#00b894", "PNEUMONIA": "#d63031"}
    color  = colors.get(pred, "#fdcb6e")

    axes[0].text(
        0.5, -0.04,
        f"Prediction: {pred}  ({conf:.1f}%)",
        transform=axes[0].transAxes,
        ha="center", fontsize=12, fontweight="bold",
        color=color,
    )

    # ── Right: Probability bar chart ────────
    classes = list(result["probabilities"].keys())
    probs   = list(result["probabilities"].values())
    bar_clrs = [colors.get(c, "#636e72") for c in classes]

    bars = axes[1].barh(classes, probs, color=bar_clrs, height=0.5)
    axes[1].set_xlim(0, 105)
    axes[1].set_xlabel("Probability (%)", color="white")
    axes[1].set_title("Class Probabilities", color="white", fontsize=13)
    axes[1].tick_params(colors="white")
    axes[1].set_facecolor("#16213e")
    for spine in axes[1].spines.values():
        spine.set_edgecolor("#444")

    for bar, p in zip(bars, probs):
        axes[1].text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                     f"{p:.1f}%", va="center", color="white", fontsize=11)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  ✔ Prediction plot saved → {save_path}")

    plt.show()
    plt.close()


# ─────────────────────────────────────────────
# Batch prediction helper
# ─────────────────────────────────────────────

@torch.no_grad()
def predict_batch(image_paths: list, model=None) -> list:
    """
    Run inference on a list of image paths.

    Returns a list of result dicts (same format as predict_image).
    """
    if model is None:
        model = get_model()
        model, _, _, _ = load_checkpoint(config.MODEL_PATH, model)
    model.to(config.DEVICE)
    model.eval()

    transform = get_transforms("test")
    results   = []

    for path in image_paths:
        result = predict_image(path, model=model, show_plot=False)
        results.append(result)

    return results


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pneumonia Prediction")
    parser.add_argument("--image", required=True,
                        help="Path to chest X-ray image")
    parser.add_argument("--save", default=None,
                        help="Optional path to save the result plot")
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable the plot display")
    args = parser.parse_args()

    result = predict_image(
        image_path = args.image,
        show_plot  = not args.no_plot,
        save_path  = args.save,
    )

    # Optionally dump JSON
    print("\nJSON output:")
    print(json.dumps(result, indent=2))
