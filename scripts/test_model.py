"""Quick test to verify the model works with our data pipeline."""

import torch
from src.models.classifier import BrainTumorClassifier
from src.data.dataset import get_dataloaders, CLASS_NAMES


def test_model():
    print("=" * 50)
    print("TESTING MODEL")
    print("=" * 50)

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Create model
    model = BrainTumorClassifier(num_classes=4, backbone_name="resnet18")
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test with one batch from our data
    print("\n--- Testing with real data ---")
    loaders = get_dataloaders(batch_size=16)
    images, labels = next(iter(loaders["train"]))
    images, labels = images.to(device), labels.to(device)

    print(f"Input shape:  {images.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(images)

    print(f"Output shape: {output.shape}")
    print(f"Output sample: {output[0].cpu()}")

    # Get predictions
    predictions = torch.argmax(output, dim=1)
    print(f"\nPredictions: {[CLASS_NAMES[p] for p in predictions.cpu()]}")
    print(f"Actual:      {[CLASS_NAMES[l] for l in labels.cpu()]}")

    print(f"\n{'=' * 50}")
    print("MODEL TEST PASSED ✓")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    test_model()