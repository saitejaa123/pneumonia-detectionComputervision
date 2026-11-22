"""
dataset.py
----------
PyTorch Dataset and DataLoader setup for the chest X-ray pneumonia dataset.

Augmentation strategy
  Train : RandomHorizontalFlip, RandomRotation, ColorJitter, RandomAffine
  Val/Test : Resize + CenterCrop + Normalize only

Class imbalance is handled via a WeightedRandomSampler so every training
batch sees a balanced mix of NORMAL and PNEUMONIA images.
"""

import os
from typing import Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

import config
from utils import set_seed


# ─────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────

def get_transforms(split: str = "train") -> transforms.Compose:
    """
    Return the appropriate transform pipeline for a given split.

    Parameters
    ----------
    split : "train" | "val" | "test"
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE + 32, config.IMAGE_SIZE + 32)),
            transforms.RandomCrop(config.IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, shear=5, scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.MEAN, std=config.STD),
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.MEAN, std=config.STD),
        ])


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class ChestXRayDataset(Dataset):
    """
    Loads chest X-ray images from a directory structured as:

        root/
            NORMAL/    *.jpeg
            PNEUMONIA/ *.jpeg
    """

    def __init__(self, root_dir: str, split: str = "train"):
        """
        Parameters
        ----------
        root_dir : path to train / val / test folder
        split    : "train" | "val" | "test"
        """
        self.root_dir  = root_dir
        self.split     = split
        self.transform = get_transforms(split)
        self.samples   = []   # list of (image_path, label_idx)
        self.labels    = []   # parallel list of label indices (for sampler)

        self._load_samples()

    def _load_samples(self):
        """Walk the directory and build the sample list."""
        supported = {".jpg", ".jpeg", ".png"}

        for label_idx, class_name in enumerate(config.CLASSES):
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                raise FileNotFoundError(
                    f"Class folder not found: {class_dir}\n"
                    f"Make sure you downloaded the dataset to {config.DATA_DIR}"
                )
            for fname in os.listdir(class_dir):
                if os.path.splitext(fname)[1].lower() in supported:
                    self.samples.append(
                        (os.path.join(class_dir, fname), label_idx)
                    )
                    self.labels.append(label_idx)

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {self.root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Load and convert to RGB (handles grayscale DICOM exports too)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, label

    def class_counts(self):
        """Return a dict {class_name: count}."""
        counts = {}
        for label_idx, name in enumerate(config.CLASSES):
            counts[name] = self.labels.count(label_idx)
        return counts


# ─────────────────────────────────────────────
# Weighted sampler (combat class imbalance)
# ─────────────────────────────────────────────

def make_weighted_sampler(dataset: ChestXRayDataset) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler that gives each class equal expected
    frequency within a batch.
    """
    counts    = np.bincount(dataset.labels)          # [n_normal, n_pneumonia]
    n_total   = len(dataset.labels)
    weights   = n_total / (len(counts) * counts)     # inverse frequency
    sample_wt = [weights[lbl] for lbl in dataset.labels]

    return WeightedRandomSampler(
        weights     = sample_wt,
        num_samples = len(sample_wt),
        replacement = True,
    )


# ─────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────

def get_dataloaders(
    use_weighted_sampler: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build and return (train_loader, val_loader, test_loader).

    Parameters
    ----------
    use_weighted_sampler : if True the training loader uses a balanced sampler
    """
    set_seed()

    train_dataset = ChestXRayDataset(config.TRAIN_DIR, split="train")
    val_dataset   = ChestXRayDataset(config.VAL_DIR,   split="val")
    test_dataset  = ChestXRayDataset(config.TEST_DIR,  split="test")

    # Print dataset stats
    for split_name, ds in [("Train", train_dataset),
                            ("Val",   val_dataset),
                            ("Test",  test_dataset)]:
        counts = ds.class_counts()
        print(f"  {split_name:5s}: {len(ds):5d} images  |  "
              + "  ".join(f"{k}: {v}" for k, v in counts.items()))

    # Sampler only for training
    sampler = make_weighted_sampler(train_dataset) if use_weighted_sampler else None

    train_loader = DataLoader(
        train_dataset,
        batch_size  = config.BATCH_SIZE,
        sampler     = sampler,
        shuffle     = sampler is None,   # mutually exclusive with sampler
        num_workers = config.NUM_WORKERS,
        pin_memory  = config.PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = config.BATCH_SIZE,
        shuffle     = False,
        num_workers = config.NUM_WORKERS,
        pin_memory  = config.PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = config.BATCH_SIZE,
        shuffle     = False,
        num_workers = config.NUM_WORKERS,
        pin_memory  = config.PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading datasets …")
    train_loader, val_loader, test_loader = get_dataloaders()
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape : {images.shape}")
    print(f"Labels      : {labels.tolist()[:10]}")
    print(f"Min pixel   : {images.min():.3f}  Max pixel: {images.max():.3f}")
    print("\nDataset check passed ✔")
