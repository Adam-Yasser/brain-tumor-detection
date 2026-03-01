"""
Evaluation metrics and visualization for Brain Tumor Classification.

Generates:
- Classification report (precision, recall, F1 per class)
- Confusion matrix
- Per-class accuracy breakdown
- Misclassified image analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from src.data.dataset import CLASS_NAMES


@torch.no_grad()
def get_predictions(model, dataloader, device):
    """
    Run the model on an entire dataloader and collect all predictions.

    Args:
        model:      Trained model
        dataloader: DataLoader to evaluate
        device:     'cuda' or 'cpu'

    Returns:
        all_labels:  True labels (numpy array)
        all_preds:   Predicted labels (numpy array)
        all_probs:   Prediction probabilities (numpy array)
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    for images, labels in dataloader:
        images = images.to(device)
        outputs = model(images)

        # Convert logits to probabilities
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def print_classification_report(labels, preds):
    """Print detailed per-class metrics."""
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    report = classification_report(
        labels, preds,
        target_names=CLASS_NAMES,
        digits=4,
    )
    print(report)

    accuracy = accuracy_score(labels, preds)
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

    return accuracy


def plot_confusion_matrix(labels, preds, save_path):
    """
    Generate and save a confusion matrix heatmap.

    The confusion matrix shows:
    - Rows = true class
    - Columns = predicted class
    - Diagonal = correct predictions
    - Off-diagonal = mistakes
    """
    cm = confusion_matrix(labels, preds)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=axes[0],
    )
    axes[0].set_title("Confusion Matrix (Counts)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")

    # Normalized (percentages per row)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(
        cm_normalized, annot=True, fmt=".2%", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=axes[1],
    )
    axes[1].set_title("Confusion Matrix (Normalized)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nConfusion matrix saved to {save_path}")


def plot_per_class_accuracy(labels, preds, save_path):
    """Bar chart showing accuracy for each class."""
    cm = confusion_matrix(labels, preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(CLASS_NAMES, per_class_acc, color=colors, edgecolor="black", alpha=0.8)

    for bar, acc in zip(bars, per_class_acc):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{acc:.1%}", ha="center", fontweight="bold", fontsize=12,
        )

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Per-Class Accuracy on Test Set", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Per-class accuracy saved to {save_path}")


def find_misclassified(dataset, labels, preds, probs, max_samples=16):
    """
    Find misclassified images for error analysis.

    Returns list of dicts with image path, true label, predicted label,
    and confidence score.
    """
    misclassified = []

    for idx in range(len(labels)):
        if labels[idx] != preds[idx]:
            img_path, _ = dataset.samples[idx]
            confidence = probs[idx][preds[idx]]

            misclassified.append({
                "path": img_path,
                "true_label": CLASS_NAMES[labels[idx]],
                "pred_label": CLASS_NAMES[preds[idx]],
                "confidence": confidence,
            })

    # Sort by confidence (highest first — these are the most "confidently wrong")
    misclassified.sort(key=lambda x: x["confidence"], reverse=True)

    return misclassified[:max_samples]


def plot_misclassified(misclassified, save_path):
    """Display misclassified images with their true and predicted labels."""
    n = len(misclassified)
    if n == 0:
        print("No misclassified images found!")
        return

    cols = 4
    rows = min((n + cols - 1) // cols, 4)
    show = min(n, rows * cols)

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    fig.suptitle("Misclassified Images (sorted by confidence)", fontsize=14, fontweight="bold")

    if rows == 1:
        axes = [axes]

    for idx in range(show):
        row, col = idx // cols, idx % cols
        item = misclassified[idx]

        from PIL import Image
        img = Image.open(item["path"]).convert("RGB")

        axes[row][col].imshow(img)
        axes[row][col].set_title(
            f"True: {item['true_label']}\n"
            f"Pred: {item['pred_label']}\n"
            f"Conf: {item['confidence']:.1%}",
            fontsize=10,
            color="red",
        )
        axes[row][col].axis("off")

    # Hide empty subplots
    for idx in range(show, rows * cols):
        row, col = idx // cols, idx % cols
        axes[row][col].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Misclassified images saved to {save_path}")