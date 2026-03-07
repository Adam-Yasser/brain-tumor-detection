"""
Evaluate the trained Brain Tumor Classifier on the test set.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --model outputs/models/best_model.pth
"""

import argparse
import yaml
import torch
from pathlib import Path

from src.models.classifier import BrainTumorClassifier
from src.data.dataset import get_datasets, get_dataloaders
from src.evaluation.metrics import (
    get_predictions,
    print_classification_report,
    plot_confusion_matrix,
    plot_per_class_accuracy,
    find_misclassified,
    plot_misclassified,
)


def evaluate(model_path="outputs/models/best_model.pth"):
    """Load best model and evaluate on test set."""

    # Create output directories if they don't exist
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Rebuild model from saved config
    model = BrainTumorClassifier(
        num_classes=checkpoint["num_classes"],
        backbone_name=checkpoint["backbone_name"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    print(f"  Backbone: {checkpoint['backbone_name']}")
    print(f"  Val accuracy (during training): {checkpoint['val_accuracy']:.4f}")

    # Load test data
    config = checkpoint["config"]
    datasets = get_datasets(config["data"]["data_dir"])
    loaders = get_dataloaders(
        data_dir=config["data"]["data_dir"],
        batch_size=config["data"]["batch_size"],
        num_workers=0,
    )

    # Get predictions on test set
    print("\nRunning inference on test set...")
    labels, preds, probs = get_predictions(model, loaders["test"], device)

    # Print classification report
    accuracy = print_classification_report(labels, preds)

    # Save confusion matrix
    figure_dir = config["paths"]["figure_dir"]
    plot_confusion_matrix(labels, preds, f"{figure_dir}/confusion_matrix.png")

    # Save per-class accuracy
    plot_per_class_accuracy(labels, preds, f"{figure_dir}/per_class_accuracy.png")

    # Error analysis
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)

    misclassified = find_misclassified(datasets["test"], labels, preds, probs)
    print(f"\nTotal misclassified: {(labels != preds).sum()} / {len(labels)}")
    print(f"\nTop confidently wrong predictions:")
    for item in misclassified[:8]:
        print(
            f"  True: {item['true_label']:12s} → "
            f"Pred: {item['pred_label']:12s} "
            f"(confidence: {item['confidence']:.1%})  "
            f"{item['path'].split('/')[-1]}"
        )

    plot_misclassified(misclassified, f"{figure_dir}/misclassified.png")

    print(f"\n{'=' * 60}")
    print(f"TEST ACCURACY: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Brain Tumor Classifier")
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/models/best_model.pth",
        help="Path to model checkpoint",
    )
    args = parser.parse_args()
    evaluate(args.model)