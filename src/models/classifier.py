"""
Brain Tumor Classifier using pretrained models.

Supports multiple backbones (ResNet-18, ResNet-50, EfficientNet-B0, DenseNet-121)
with transfer learning.
"""

import torch.nn as nn
from torchvision import models


class BrainTumorClassifier(nn.Module):
    """
    Image classifier for brain tumor detection.

    Uses a pretrained backbone with a custom classification head.
    The backbone extracts visual features, and the head classifies
    them into one of 4 classes.

    Args:
        num_classes:   Number of output classes (default: 4)
        backbone_name: Which pretrained model to use
        freeze_backbone: If True, only train the classification head
    """

    # Supported backbones and their feature dimensions
    BACKBONES = {
        "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT, 512),
        "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, 2048),
        "densenet121": (models.densenet121, models.DenseNet121_Weights.DEFAULT, 1024),
        "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT, 1280),
    }

    def __init__(self, num_classes=4, backbone_name="resnet18", freeze_backbone=False):
        super().__init__()

        if backbone_name not in self.BACKBONES:
            raise ValueError(
                f"Unknown backbone: {backbone_name}. "
                f"Choose from: {list(self.BACKBONES.keys())}"
            )

        model_fn, weights, num_features = self.BACKBONES[backbone_name]
        self.backbone_name = backbone_name

        # Load pretrained backbone
        backbone = model_fn(weights=weights)

        # Remove the original classification head
        if backbone_name.startswith("resnet"):
            self.features = nn.Sequential(*list(backbone.children())[:-1])
        elif backbone_name == "densenet121":
            self.features = backbone.features
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif backbone_name == "efficientnet_b0":
            self.features = backbone.features
            self.pool = nn.AdaptiveAvgPool2d(1)

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        # Our custom classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Batch of images, shape (batch_size, 3, 224, 224)

        Returns:
            Logits, shape (batch_size, num_classes)
        """
        x = self.features(x)

        if self.backbone_name in ["densenet121", "efficientnet_b0"]:
            x = self.pool(x)

        x = self.classifier(x)
        return x