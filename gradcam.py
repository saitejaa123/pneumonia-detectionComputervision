"""
gradcam.py
----------
Grad-CAM (Gradient-weighted Class Activation Mapping) visualization.

Highlights the regions of a chest X-ray that most influenced the model's
prediction — essential for clinical interpretability.

Usage (CLI):
    python gradcam.py --image path/to/xray.jpeg [--layer features.denseblock4]

Usage (Python):
    from gradcam import run_gradcam
    run_gradcam("path/to/xray.jpeg")
"""

import argparse
import os
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import config
from model import get_model
from utils import load_checkpoint
from dataset import get_transforms


# ─────────────────────────────────────────────
# Grad-CAM implementation
# ─────────────────────────────────────────────

class GradCAM:
    """
    Vanilla Grad-CAM for any CNN.

    Parameters
    ----------
    model      : trained nn.Module
    target_layer: the convolutional layer to hook (e.g. model.features.denseblock4)
    """

    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None):
        """
        Compute the Grad-CAM heatmap for a single image tensor.

        Parameters
        ----------
        input_tensor : (1, C, H, W) tensor on the correct device
        class_idx    : target class (defaults to predicted class)

        Returns
        -------
        heatmap : (H, W) numpy array in [0, 1]
        pred_idx: int predicted class index
        pred_prob: float predicted probability
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)
        probs  = torch.softmax(output, dim=1)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        pred_prob = probs[0, class_idx].item()

        # Backward pass w.r.t. target class
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()

        # Pool gradients over spatial dimensions
        pooled_grads = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Weight activations
        weighted_act = (pooled_grads * self.activations).squeeze(0)   # (C, H, W)
        heatmap      = weighted_act.sum(dim=0).cpu().numpy()           # (H, W)

        # ReLU + normalize
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap, class_idx, pred_prob


# ─────────────────────────────────────────────
# Overlay heatmap on image
# ─────────────────────────────────────────────

def overlay_heatmap(heatmap: np.ndarray, original_img: np.ndarray,
                    alpha: float = 0.4) -> np.ndarray:
    """
    Resize heatmap to match the original image and blend as a colour overlay.

    Parameters
    ----------
    heatmap      : (H, W) float array in [0, 1]
    original_img : (H, W, 3) uint8 RGB array
    alpha        : heatmap opacity

    Returns
    -------
    blended : (H, W, 3) uint8 RGB array
    """
    h, w = original_img.shape[:2]

    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Apply colormap (JET: blue=low, red=high)
    colormap = cm.get_cmap("jet")
    colored  = colormap(heatmap_resized)[:, :, :3]           # (H, W, 3) float
    colored  = (colored * 255).astype(np.uint8)              # → uint8 RGB

    # Blend
    blended = cv2.addWeighted(original_img, 1 - alpha, colored, alpha, 0)
    return blended


# ─────────────────────────────────────────────
# Target layer helper
# ─────────────────────────────────────────────

def get_target_layer(model, layer_name: str = None):
    """
    Return the appropriate last convolutional layer for Grad-CAM.
    Falls back to sensible defaults per architecture.
    """
    if layer_name:
        # Navigate dotted path e.g. "features.denseblock4"
        parts = layer_name.split(".")
        layer = model
        for p in parts:
            layer = getattr(layer, p)
        return layer

    # Auto-detect
    name = config.MODEL_NAME.lower()
    if "densenet" in name:
        return model.features.denseblock4
    elif "resnet" in name:
        return model.layer4
    elif "efficientnet" in name:
        return model.features[-1]
    elif "vgg" in name:
        return model.features[-1]          # last MaxPool2d after conv block
    elif "mobilenet" in name:
        return model.features[-1]          # last Conv2dNormActivation block
    else:
        raise ValueError(
            f"Cannot auto-detect target layer for '{config.MODEL_NAME}'. "
            "Pass --layer explicitly."
        )


# ─────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────

def run_gradcam(image_path: str, layer_name: str = None,
                save_dir: str = None):
    """
    Generate and display Grad-CAM for a single X-ray image.

    Parameters
    ----------
    image_path : path to the JPEG/PNG chest X-ray
    layer_name : dotted layer name override (optional)
    save_dir   : folder to save the output image (optional)
    """
    # ── Load model ──────────────────────────
    model = get_model()
    model, _, _, _ = load_checkpoint(config.MODEL_PATH, model)
    model.eval()

    # ── Target layer ────────────────────────
    target_layer = get_target_layer(model, layer_name)
    grad_cam     = GradCAM(model, target_layer)

    # ── Preprocess image ────────────────────
    transform   = get_transforms("test")
    pil_image   = Image.open(image_path).convert("RGB")
    input_tensor = transform(pil_image).unsqueeze(0).to(config.DEVICE)

    # Keep original for overlay
    orig_np = np.array(pil_image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE)))

    # ── Generate CAM ────────────────────────
    heatmap, pred_idx, pred_prob = grad_cam.generate(input_tensor)

    label_name = config.CLASSES[pred_idx]
    print(f"\n  Prediction : {label_name}  (confidence {pred_prob * 100:.1f}%)")

    # ── Overlay ─────────────────────────────
    blended = overlay_heatmap(heatmap, orig_np)

    # ── Plot ────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        f"Grad-CAM  |  Prediction: {label_name}  ({pred_prob * 100:.1f}%)",
        fontsize=14, fontweight="bold"
    )

    axes[0].imshow(orig_np, cmap="gray" if orig_np.ndim == 2 else None)
    axes[0].set_title("Original X-Ray")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(blended)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.splitext(os.path.basename(image_path))[0]
        out   = os.path.join(save_dir, f"gradcam_{fname}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  ✔ Grad-CAM saved → {out}")

    plt.show()


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grad-CAM for Chest X-Ray")
    parser.add_argument("--image", required=True,
                        help="Path to the X-ray image file")
    parser.add_argument("--layer", default=None,
                        help="Dotted path to target layer (optional)")
    parser.add_argument("--save-dir", default=config.RESULTS_DIR,
                        help="Directory to save the Grad-CAM image")
    args = parser.parse_args()

    run_gradcam(args.image, layer_name=args.layer, save_dir=args.save_dir)
