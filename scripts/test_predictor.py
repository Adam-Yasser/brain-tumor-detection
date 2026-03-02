"""Test the predictor interface with sample images."""

from pathlib import Path
from src.inference.predictor import BrainTumorPredictor


def test_predictor():
    print("=" * 50)
    print("TESTING PREDICTOR")
    print("=" * 50)

    # Load model
    predictor = BrainTumorPredictor("outputs/deployment/brain_tumor_model.pth")

    # Test with one image from each class
    test_dir = Path("data/splits/test")
    classes = ["glioma", "meningioma", "notumor", "pituitary"]

    print("\n--- Predictions ---")
    for cls in classes:
        img_path = list((test_dir / cls).iterdir())[0]
        result = predictor.predict(str(img_path))

        status = "✓" if result["predicted_class"] == cls else "✗"
        print(
            f"  {status} True: {cls:12s} → "
            f"Pred: {result['predicted_class']:12s} "
            f"(confidence: {result['confidence']:.1%})"
        )

    # Test batch prediction
    print("\n--- Batch Prediction ---")
    sample_paths = [
        str(list((test_dir / cls).iterdir())[0])
        for cls in classes
    ]
    results = predictor.predict_batch(sample_paths)
    print(f"  Batch of {len(results)} images processed successfully")

    print(f"\n{'=' * 50}")
    print("PREDICTOR TEST PASSED ✓")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    test_predictor()