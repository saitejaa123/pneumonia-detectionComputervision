"""
run_all_models.py
-----------------
Trains, evaluates, and saves results for all 5 models sequentially.
Results are saved to outputs/comparison/

Usage:
    python run_all_models.py
"""

import os
import sys
import json
import time
import torch
import importlib
import config
from utils import set_seed, print_device_info

MODELS = ["densenet121", "resnet50", "efficientnet_b0", "vgg16", "mobilenet_v3"]

COMPARISON_DIR = os.path.join(config.OUTPUT_DIR, "comparison")
os.makedirs(COMPARISON_DIR, exist_ok=True)


def run_model(model_name: str) -> dict:
    print("\n" + "=" * 60)
    print(f"  MODEL: {model_name.upper()}")
    print("=" * 60)

    # Override model name in config at runtime
    config.MODEL_NAME = model_name
    config.MODEL_PATH = os.path.join(config.OUTPUT_DIR, f"best_model_{model_name}.pth")

    set_seed()

    # ── Train (skip if checkpoint already exists) ───
    if os.path.exists(config.MODEL_PATH):
        print(f"\n[1/3] Skipping training — checkpoint already exists: {config.MODEL_PATH}")
        # Force-reload model module so it picks up the new config.MODEL_NAME
        import model as model_mod
        importlib.reload(model_mod)
        from utils import load_checkpoint
        m = model_mod.get_model(name=model_name)
        m, _, _, _ = load_checkpoint(config.MODEL_PATH, m)
        tracker = None
    else:
        print(f"\n[1/3] Training {model_name} …\n")
        # Force-reload train + model modules so they pick up new config.MODEL_NAME
        import model as model_mod
        import train as train_mod
        importlib.reload(model_mod)
        importlib.reload(train_mod)
        m, tracker = train_mod.train()

    # ── Evaluate ───────────────────────────
    print(f"\n[2/3] Evaluating {model_name} …\n")
    import evaluate as eval_mod
    importlib.reload(eval_mod)
    metrics = eval_mod.evaluate(model=m)

    # ── Grad-CAM sample ────────────────────
    print(f"\n[3/3] Grad-CAM for {model_name} …\n")
    import gradcam as gradcam_mod
    importlib.reload(gradcam_mod)
    for cls in config.CLASSES:
        cls_dir = os.path.join(config.TEST_DIR, cls)
        if not os.path.isdir(cls_dir):
            continue
        files = [f for f in os.listdir(cls_dir)
                 if f.lower().endswith((".jpeg", ".jpg", ".png"))]
        if files:
            sample_path = os.path.join(cls_dir, files[0])
            save_dir = os.path.join(COMPARISON_DIR, model_name)
            os.makedirs(save_dir, exist_ok=True)
            gradcam_mod.run_gradcam(sample_path, save_dir=save_dir)

    return metrics


def main():
    print_device_info()
    all_results = {}
    times = {}

    for model_name in MODELS:
        start = time.time()
        try:
            metrics = run_model(model_name)
            all_results[model_name] = metrics
        except Exception as e:
            print(f"\n  ERROR training {model_name}: {e}")
            all_results[model_name] = {"error": str(e)}
        times[model_name] = round(time.time() - start, 1)

    # ── Print comparison table ──────────────
    print("\n" + "=" * 70)
    print("  MODEL COMPARISON RESULTS")
    print("=" * 70)
    header = f"{'Model':<20} {'Accuracy':>10} {'AUC-ROC':>10} {'Sensitivity':>12} {'Specificity':>12} {'Time(s)':>9}"
    print(header)
    print("-" * 70)
    for name, metrics in all_results.items():
        if "error" in metrics:
            print(f"  {name:<18}  ERROR: {metrics['error']}")
        else:
            print(
                f"  {name:<18}"
                f"  {metrics.get('accuracy', 0):>9.2f}%"
                f"  {metrics.get('auc_roc', 0):>9.2f}%"
                f"  {metrics.get('sensitivity', 0):>11.2f}%"
                f"  {metrics.get('specificity', 0):>11.2f}%"
                f"  {times[name]:>8.1f}s"
            )

    # ── Save JSON summary ───────────────────
    summary = {
        "models": MODELS,
        "results": all_results,
        "training_time_seconds": times,
    }
    summary_path = os.path.join(COMPARISON_DIR, "comparison_results.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved → {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
