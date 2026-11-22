"""
train.py
--------
Full training loop with:
  - Weighted cross-entropy loss (class imbalance)
  - Adam optimizer + StepLR scheduler
  - Early stopping
  - Best-model checkpointing
  - Live progress bars (tqdm)
  - Training curve plots saved to outputs/results/
"""

import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

import config
from dataset import get_dataloaders
from model import get_model
from utils import (
    set_seed, print_device_info,
    MetricTracker, EarlyStopping,
    save_checkpoint
)


# ─────────────────────────────────────────────
# One epoch helpers
# ─────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, is_train: bool):
    """Run one full pass over the data loader and return (loss, accuracy)."""
    model.train() if is_train else model.eval()

    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    context = torch.enable_grad if is_train else torch.no_grad
    desc    = "  Train" if is_train else "  Val  "

    with context():
        for images, labels in tqdm(loader, desc=desc, leave=False,
                                   unit="batch", ncols=80):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Forward
            logits = model(images)
            loss   = criterion(logits, labels)

            # Backward (train only)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Accumulate stats
            total_loss    += loss.item() * images.size(0)
            preds          = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

    avg_loss = total_loss    / total_samples
    accuracy = total_correct / total_samples * 100.0
    return avg_loss, accuracy


# ─────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────

def train():
    print("\n" + "=" * 55)
    print("  Pneumonia Detection — Training")
    print("=" * 55)

    set_seed()
    print_device_info()

    # ── Data ────────────────────────────────
    print("\nLoading data …")
    train_loader, val_loader, _ = get_dataloaders(use_weighted_sampler=True)

    # ── Model ───────────────────────────────
    print("\nBuilding model …")
    model = get_model()

    # ── Loss (weighted to penalise NORMAL misses less) ──────
    class_weights = torch.tensor(config.CLASS_WEIGHTS,
                                 dtype=torch.float32).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Optimizer & Scheduler ───────────────
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = config.LEARNING_RATE,
        weight_decay = config.WEIGHT_DECAY,
    )
    scheduler = StepLR(optimizer,
                       step_size = config.LR_STEP_SIZE,
                       gamma     = config.LR_GAMMA)

    # ── Trackers ────────────────────────────
    tracker      = MetricTracker()
    early_stop   = EarlyStopping(patience=config.PATIENCE)
    best_val_loss = float("inf")

    # ── Training Loop ───────────────────────
    print(f"\nStarting training for up to {config.NUM_EPOCHS} epochs …\n")
    start_time = time.time()

    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"Epoch [{epoch:02d}/{config.NUM_EPOCHS}]")

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer,
            config.DEVICE, is_train=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer,
            config.DEVICE, is_train=False
        )
        scheduler.step()

        # Log
        tracker.update(train_loss, train_acc, val_loss, val_acc)
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"  Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}%  |  "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.2f}%  |  "
            f"LR: {current_lr:.2e}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                state={
                    "epoch":               epoch,
                    "model_state_dict":    model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss":       best_val_loss,
                    "val_acc":             val_acc,
                    "model_name":          config.MODEL_NAME,
                },
                filepath=config.MODEL_PATH,
            )

        # Early stopping
        if early_stop(val_loss):
            print(f"\n  Early stopping triggered at epoch {epoch}.")
            break

        print()

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed / 60:.1f} min.")
    print(f"Best Val Loss : {best_val_loss:.4f}")

    # ── Save training curves ─────────────────
    curves_path = os.path.join(config.RESULTS_DIR, "training_curves.png")
    tracker.plot(save_path=curves_path)

    return model, tracker


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    train()
