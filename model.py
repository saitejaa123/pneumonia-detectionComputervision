"""
model.py
--------
Model factory that supports DenseNet121, ResNet50, and EfficientNet-B0
with custom classification heads for binary pneumonia detection.

Default: DenseNet121 (inspired by CheXNet — the clinical benchmark for
         chest X-ray classification).
"""

import torch
import torch.nn as nn
from torchvision import models

import config
from utils import count_parameters


# ─────────────────────────────────────────────
# Custom classification head
# ─────────────────────────────────────────────

class ClassifierHead(nn.Module):
    """
    Fully-connected head attached after the backbone feature extractor.

    Architecture: GlobalAvgPool → Dropout → FC → output logits
    """

    def __init__(self, in_features: int, num_classes: int = config.NUM_CLASSES,
                 dropout: float = config.DROPOUT_RATE):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),          # → (B, in_features, 1, 1)
            nn.Flatten(),                      # → (B, in_features)
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ─────────────────────────────────────────────
# Model builders
# ─────────────────────────────────────────────

def build_densenet121(pretrained: bool = True,
                      freeze: bool = False) -> nn.Module:
    """DenseNet121 — recommended backbone for chest X-ray tasks."""
    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    model   = models.densenet121(weights=weights)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the classifier
    in_features = model.classifier.in_features   # 1024
    model.classifier = nn.Sequential(
        nn.Dropout(p=config.DROPOUT_RATE),
        nn.Linear(in_features, config.NUM_CLASSES),
    )
    return model


def build_resnet50(pretrained: bool = True,
                   freeze: bool = False) -> nn.Module:
    """ResNet50 backbone."""
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model   = models.resnet50(weights=weights)

    if freeze:
        for name, param in model.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False

    in_features = model.fc.in_features          # 2048
    model.fc = nn.Sequential(
        nn.Dropout(p=config.DROPOUT_RATE),
        nn.Linear(in_features, config.NUM_CLASSES),
    )
    return model


def build_efficientnet_b0(pretrained: bool = True,
                           freeze: bool = False) -> nn.Module:
    """EfficientNet-B0 backbone."""
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model   = models.efficientnet_b0(weights=weights)

    if freeze:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=config.DROPOUT_RATE),
        nn.Linear(in_features, config.NUM_CLASSES),
    )
    return model


def build_vgg16(pretrained: bool = True,
                freeze: bool = False) -> nn.Module:
    """VGG16 backbone — strong baseline with deep conv features."""
    weights = models.VGG16_Weights.DEFAULT if pretrained else None
    model   = models.vgg16(weights=weights)

    if freeze:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace the classifier (in_features = 4096 after VGG's AdaptiveAvgPool)
    in_features = model.classifier[0].in_features  # 25088
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(p=config.DROPOUT_RATE),
        nn.Linear(4096, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=config.DROPOUT_RATE / 2),
        nn.Linear(512, config.NUM_CLASSES),
    )
    return model


def build_mobilenet_v3(pretrained: bool = True,
                       freeze: bool = False) -> nn.Module:
    """MobileNetV3-Large — lightweight and fast, good for deployment."""
    weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model   = models.mobilenet_v3_large(weights=weights)

    if freeze:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[0].in_features  # 960
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 1280),
        nn.Hardswish(inplace=True),
        nn.Dropout(p=config.DROPOUT_RATE),
        nn.Linear(1280, config.NUM_CLASSES),
    )
    return model


# ─────────────────────────────────────────────
# Public factory function
# ─────────────────────────────────────────────

_BUILDERS = {
    "densenet121":     build_densenet121,
    "resnet50":        build_resnet50,
    "efficientnet_b0": build_efficientnet_b0,
    "vgg16":           build_vgg16,
    "mobilenet_v3":    build_mobilenet_v3,
}


def get_model(
    name: str      = config.MODEL_NAME,
    pretrained: bool = config.PRETRAINED,
    freeze: bool   = config.FREEZE_BACKBONE,
) -> nn.Module:
    """
    Build and return a model ready for training.

    Parameters
    ----------
    name       : model name key (see _BUILDERS)
    pretrained : load ImageNet weights
    freeze     : freeze backbone, train head only

    Returns
    -------
    model moved to config.DEVICE
    """
    if name not in _BUILDERS:
        raise ValueError(
            f"Unknown model '{name}'. Choose from: {list(_BUILDERS.keys())}"
        )

    model = _BUILDERS[name](pretrained=pretrained, freeze=freeze)
    model = model.to(config.DEVICE)

    trainable = count_parameters(model)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Model        : {name}")
    print(f"  Parameters   : {trainable:,} trainable / {total:,} total")
    print(f"  Device       : {config.DEVICE}")

    return model


# ─────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import config
    model = get_model()
    dummy = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).to(config.DEVICE)
    out   = model(dummy)
    print(f"\nInput  shape : {dummy.shape}")
    print(f"Output shape : {out.shape}")
    assert out.shape == (2, config.NUM_CLASSES), "Output shape mismatch!"
    print("Model check passed ✔")
