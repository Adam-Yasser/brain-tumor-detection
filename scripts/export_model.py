"""
Export the trained model for deployment.

Creates a deployment package with:
- The model checkpoint
- A usage example
- Model metadata

Usage:
    python scripts/export_model.py
    python scripts/export_model.py --model outputs/models/efficientnet_b0.pth
"""

import argparse
import shutil
import json
import torch
from pathlib import Path
from src.models.classifier import BrainTumorClassifier


def export_model(model_path="outputs/models/efficientnet_b0.pth"):
    """Export model with metadata for the back-end team."""

    export_dir = Path("outputs/deployment")
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True)

    # Load and verify model
    print("=" * 50)
    print("EXPORTING MODEL")
    print("=" * 50)

    device = torch.device("cpu")  # Export on CPU for portability
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = BrainTumorClassifier(
        num_classes=checkpoint["num_classes"],
        backbone_name=checkpoint["backbone_name"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"  Model: {checkpoint['backbone_name']}")
    print(f"  Val accuracy: {checkpoint['val_accuracy']:.1%}")

    # Copy model file
    shutil.copy2(model_path, export_dir / "brain_tumor_model.pth")
    print(f"  Model copied to {export_dir / 'brain_tumor_model.pth'}")

    # Create metadata file
    metadata = {
        "model_name": "Brain Tumor Classifier",
        "backbone": checkpoint["backbone_name"],
        "num_classes": checkpoint["num_classes"],
        "classes": ["glioma", "meningioma", "notumor", "pituitary"],
        "input_size": 224,
        "input_channels": 3,
        "test_accuracy": 0.9475,
        "val_accuracy": round(checkpoint["val_accuracy"], 4),
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "preprocessing_steps": [
            "1. Load image and convert to RGB",
            "2. Resize to 224x224 pixels",
            "3. Convert to tensor (divide by 255)",
            "4. Normalize with ImageNet mean and std",
        ],
    }

    with open(export_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved to {export_dir / 'model_metadata.json'}")

    # Create usage example
    usage_example = '''"""
Brain Tumor Classification — Usage Example for Back-End Team

Requirements:
    pip install torch torchvision Pillow

Quick start:
    python usage_example.py path/to/mri_image.jpg
"""

import sys
import torch
from PIL import Image
from torchvision import transforms


# --- Configuration ---
MODEL_PATH = "brain_tumor_model.pth"
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_model(model_path=MODEL_PATH):
    """Load the trained model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Import model class
    from src.models.classifier import BrainTumorClassifier

    model = BrainTumorClassifier(
        num_classes=checkpoint["num_classes"],
        backbone_name=checkpoint["backbone_name"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, device


def predict(model, image_path, device):
    """Predict the class of a brain MRI image."""
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    pred_idx = torch.argmax(probs).item()

    return {
        "predicted_class": CLASS_NAMES[pred_idx],
        "confidence": round(probs[pred_idx].item(), 4),
        "probabilities": {
            name: round(probs[i].item(), 4)
            for i, name in enumerate(CLASS_NAMES)
        },
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python usage_example.py <image_path>")
        sys.exit(1)

    model, device = load_model()
    result = predict(model, sys.argv[1], device)

    print(f"\\nPrediction: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"\\nAll probabilities:")
    for cls, prob in result['probabilities'].items():
        print(f"  {cls}: {prob:.1%}")
'''

    with open(export_dir / "usage_example.py", "w") as f:
        f.write(usage_example)
    print(f"  Usage example saved to {export_dir / 'usage_example.py'}")

    # Summary
    print(f"\n{'=' * 50}")
    print("EXPORT COMPLETE")
    print(f"{'=' * 50}")
    print(f"\n  Deployment package: {export_dir.resolve()}")
    print(f"  Contents:")
    for f in sorted(export_dir.iterdir()):
        size = f.stat().st_size
        if size > 1024 * 1024:
            print(f"    {f.name} ({size / 1024 / 1024:.1f} MB)")
        else:
            print(f"    {f.name} ({size / 1024:.1f} KB)")
    print(f"\n  Give this folder to your back-end team!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model for deployment")
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/models/efficientnet_b0.pth",
        help="Path to model checkpoint",
    )
    args = parser.parse_args()
    export_model(args.model)