"""Quick test to verify the data pipeline works end-to-end."""

from src.data.dataset import get_datasets, get_dataloaders, CLASS_NAMES


def test_pipeline():
    print("=" * 50)
    print("TESTING DATA PIPELINE")
    print("=" * 50)

    # Test datasets
    print("\n--- Loading Datasets ---")
    datasets = get_datasets()
    for split_name, dataset in datasets.items():
        print(f"  {split_name:5s} → {len(dataset)} images | classes: {dataset.classes}")

    # Test dataloaders
    print("\n--- Loading Batches ---")
    loaders = get_dataloaders(batch_size=16)
    for split_name, loader in loaders.items():
        batch = next(iter(loader))
        images, labels = batch
        print(f"  {split_name:5s} → batch shape: {images.shape} | labels shape: {labels.shape}")

    # Verify a single sample
    print("\n--- Single Sample Check ---")
    img, label = datasets["train"][0]
    print(f"  Image shape: {img.shape}")
    print(f"  Image dtype: {img.dtype}")
    print(f"  Pixel range: [{img.min():.2f}, {img.max():.2f}]")
    print(f"  Label: {label} ({CLASS_NAMES[label]})")

    print(f"\n{'=' * 50}")
    print("ALL TESTS PASSED ✓")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    test_pipeline()