"""Model definitions for CelebA facial-attribute recognition."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


NUM_ATTRIBUTES = 40


class SimpleNN(nn.Module):
    """Two-layer fully connected baseline for flattened face images."""

    def __init__(self, input_size: int, hidden_size: int, num_classes: int = NUM_ATTRIBUTES):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class SimpleCNN(nn.Module):
    """Lightweight CNN baseline with resolution-independent pooling.

    Adaptive average pooling removes the original hard-coded fully connected
    input size and makes the model robust to small changes in image resolution.
    The network returns raw logits for use with BCEWithLogitsLoss.
    """

    def __init__(self, num_classes: int = NUM_ATTRIBUTES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(
    name: str,
    num_classes: int = NUM_ATTRIBUTES,
    pretrained: bool = True,
) -> nn.Module:
    """Construct one of the models used in the project."""
    normalized = name.lower()

    if normalized == "simplenn":
        # Default project transform is 3 x 156 x 128.
        return SimpleNN(3 * 156 * 128, hidden_size=300, num_classes=num_classes)

    if normalized == "simplecnn":
        return SimpleCNN(num_classes=num_classes)

    if normalized == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    raise ValueError("Unknown model. Choose from: SimpleNN, SimpleCNN, ResNet50")
