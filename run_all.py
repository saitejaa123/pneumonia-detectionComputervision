"""
run_all.py
----------
End-to-end pipeline runner.
Runs: Train → Evaluate → Grad-CAM on a sample image.

Usage:
    python run_all.py
    python run_all.py --skip-train   (if model already trained)
"""

import argparse
import os
import sys

import config
from utils import set_seed, print_device_info


def main(skip_train: bool = False):
    print("\n" + "=" * 60)
    print("  Pneumonia Detection — Full Pipeline")
    print("=" * 60)

    set_seed()
    print_device_info()

    # ── Step 1: Train ───────────────────────
    if not skip_train:
        print("\n[STEP 1/3] Training …\n")
        from train import train
        model, tracker = train()
    else:
        print("\n[STEP 1/3] Skipping training (--skip-train)")
        model = None

    # ── Step 2: Evaluate ────────────────────
    print("\n[STEP 2/3] Evaluating on test set …\n")
    from evaluate import evaluate
    metrics = evaluate(model=model)

    print("\nResults summary:")
    for k, v in metrics.items():
        print(f"  {k:15s}: {v:.2f}%")

    # ── Step 3: Grad-CAM sample ──────────────
    print("\n[STEP 3/3] Generating Grad-CAM for sample images …\n")
    from gradcam import run_gradcam

    for cls in config.CLASSES:
        cls_dir = os.path.join(config.TEST_DIR, cls)
        if not os.path.isdir(cls_dir):
            print(f"  Skipping {cls} — directory not found")
            continue
        files = [f for f in os.listdir(cls_dir)
                 if f.lower().endswith((".jpeg", ".jpg", ".png"))]
        if not files:
            continue
        sample_path = os.path.join(cls_dir, files[0])
        print(f"  Grad-CAM: {cls}")
        run_gradcam(sample_path, save_dir=config.RESULTS_DIR)

    print("\n" + "=" * 60)
    print("  Pipeline complete. Results in:", config.RESULTS_DIR)
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training and load saved checkpoint")
    args = parser.parse_args()
    main(skip_train=args.skip_train)
