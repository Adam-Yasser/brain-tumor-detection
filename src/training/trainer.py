"""
Training engine for Brain Tumor Classification.

Handles the training loop, validation, checkpointing, and logging.
"""

import time
import copy
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from pathlib import Path


class Trainer:
    """
    Manages the full training lifecycle.

    Args:
        model:       The neural network to train
        loaders:     Dict with 'train' and 'val' DataLoaders
        config:      Training configuration dictionary
        device:      'cuda' or 'cpu'
    """

    def __init__(self, model, loaders, config, device):
        self.model = model.to(device)
        self.loaders = loaders
        self.config = config
        self.device = device

        # Loss function — CrossEntropyLoss is standard for classification
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer — AdamW is Adam with proper weight decay
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )

        # Learning rate scheduler — reduces LR over time
        scheduler_cfg = config["training"]["scheduler"]
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=scheduler_cfg["step_size"],
            gamma=scheduler_cfg["gamma"],
        )

        # Track the best model
        self.best_val_acc = 0.0
        self.best_model_weights = None

        # Training history for plotting
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
        }

        # Mixed precision training — saves GPU memory
        self.scaler = GradScaler("cuda")

    def train_one_epoch(self):
        """Run one complete pass through the training data."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.loaders["train"]:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass with mixed precision
            with autocast("cuda"):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            # Backward pass with scaled gradients
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Track metrics
            running_loss += loss.item() * images.size(0)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate(self):
        """Evaluate the model on the validation set."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.loaders["val"]:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def save_checkpoint(self, path):
        """Save the best model weights."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.best_model_weights,
            "backbone_name": self.config["model"]["backbone"],
            "num_classes": self.config["model"]["num_classes"],
            "val_accuracy": self.best_val_acc,
            "config": self.config,
        }, path)
        print(f"  Model saved to {path}")

    def fit(self):
        """
        Full training loop.

        Returns:
            Training history dictionary
        """
        num_epochs = self.config["training"]["epochs"]
        model_dir = self.config["paths"]["model_dir"]

        print("=" * 60)
        print(f"TRAINING — {num_epochs} epochs")
        print(f"  Backbone:       {self.config['model']['backbone']}")
        print(f"  Learning rate:  {self.config['training']['learning_rate']}")
        print(f"  Batch size:     {self.config['data']['batch_size']}")
        print(f"  Device:         {self.device}")
        print("=" * 60)

        total_start = time.time()

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Train
            train_loss, train_acc = self.train_one_epoch()

            # Validate
            val_loss, val_acc = self.validate()

            # Step scheduler
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            # Save history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            # Track best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_weights = copy.deepcopy(self.model.state_dict())
                marker = " ★ BEST"
            else:
                marker = ""

            # Print progress
            epoch_time = time.time() - epoch_start
            print(
                f"  Epoch {epoch+1:2d}/{num_epochs} │ "
                f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} │ "
                f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} │ "
                f"LR: {current_lr:.6f} │ "
                f"{epoch_time:.1f}s{marker}"
            )

        # Save best model
        total_time = time.time() - total_start
        print(f"\n{'=' * 60}")
        print(f"TRAINING COMPLETE — {total_time:.1f}s total")
        print(f"  Best validation accuracy: {self.best_val_acc:.4f}")

        save_path = f"{model_dir}/best_model.pth"
        self.save_checkpoint(save_path)

        return self.history