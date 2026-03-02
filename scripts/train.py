"""
Train the Brain Tumor Classifier.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/default.yaml
"""

import argparse
import random
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import mlflow

from src.models.classifier import BrainTumorClassifier
from src.data.dataset import get_dataloaders
from src.training.trainer import Trainer


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_training_history(history, save_dir):
    """Save training curves as plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss plot
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    axes[0].plot(epochs, history["val_loss"], "r-o", label="Val Loss", markersize=4)
    axes[0].set_title("Loss over Epochs", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train Acc", markersize=4)
    axes[1].plot(epochs, history["val_acc"], "r-o", label="Val Acc", markersize=4)
    axes[1].set_title("Accuracy over Epochs", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = f"{save_dir}/training_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training curves saved to {save_path}")


def train(config_path="configs/default.yaml"):
    """Main training function."""

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Set seed
    set_seed(config["experiment"]["seed"])

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    strong_aug = config["data"].get("strong_augmentation", False)
    loaders = get_dataloaders(
        data_dir=config["data"]["data_dir"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        strong_augmentation=strong_aug,
    )

    # Model
    model = BrainTumorClassifier(
        num_classes=config["model"]["num_classes"],
        backbone_name=config["model"]["backbone"],
        freeze_backbone=config["model"]["freeze_backbone"],
    )

    # MLflow experiment tracking
    mlflow.set_tracking_uri(f"sqlite:///{config['paths']['log_dir']}/mlflow.db")
    mlflow.set_experiment(config["experiment"]["name"])

    with mlflow.start_run():
        # Log all config parameters
        mlflow.log_params({
            "backbone": config["model"]["backbone"],
            "freeze_backbone": config["model"]["freeze_backbone"],
            "batch_size": config["data"]["batch_size"],
            "learning_rate": config["training"]["learning_rate"],
            "weight_decay": config["training"]["weight_decay"],
            "epochs": config["training"]["epochs"],
            "scheduler_type": config["training"]["scheduler"]["type"],
            "scheduler_step_size": config["training"]["scheduler"]["step_size"],
            "scheduler_gamma": config["training"]["scheduler"]["gamma"],
        })

        # Train
        trainer = Trainer(model, loaders, config, device)
        history = trainer.fit()

        # Log final metrics
        mlflow.log_metrics({
            "best_val_accuracy": trainer.best_val_acc,
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
            "final_train_accuracy": history["train_acc"][-1],
            "final_val_accuracy": history["val_acc"][-1],
        })

        # Plot and save training curves
        plot_training_history(history, config["paths"]["figure_dir"])

        # Log plots to MLflow
        mlflow.log_artifact(f"{config['paths']['figure_dir']}/training_curves.png")

    print("\nDone! View experiments with: mlflow ui --backend-store-uri outputs/logs/mlruns")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Brain Tumor Classifier")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()
    train(args.config)