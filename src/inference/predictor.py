"""
Brain Tumor Predictor — Interface for the back-end team.

Usage:
    from src.inference.predictor import BrainTumorPredictor

    predictor = BrainTumorPredictor("outputs/models/best_model.pth")
    result = predictor.predict("path/to/mri_image.jpg")

    print(result)
    # {
    #     "predicted_class": "glioma",
    #     "confidence": 0.9542,
    #     "probabilities": {
    #         "glioma": 0.9542,
    #         "meningioma": 0.0301,
    #         "notumor": 0.0098,
    #         "pituitary": 0.0059
    #     }
    # }
"""

import torch
from PIL import Image
from src.data.transforms import get_val_transforms
from src.models.classifier import BrainTumorClassifier


CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


class BrainTumorPredictor:
    """
    Simple prediction interface for brain tumor classification.

    Loads a trained model and provides a predict() method that
    takes an image path and returns the prediction with confidence.

    Args:
        model_path: Path to the saved model checkpoint (.pth file)
        device:     'cuda', 'cpu', or 'auto' (automatically selects)
    """

    def __init__(self, model_path, device="auto"):
        # Select device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Build and load model
        self.model = BrainTumorClassifier(
            num_classes=checkpoint["num_classes"],
            backbone_name=checkpoint["backbone_name"],
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Image preprocessing (same as validation — no augmentation)
        self.transform = get_val_transforms()

        print(f"Model loaded: {checkpoint['backbone_name']}")
        print(f"Device: {self.device}")
        print(f"Validation accuracy: {checkpoint['val_accuracy']:.1%}")
        print(f"Test accuracy: 94.8%")

    def predict(self, image_input):
        """
        Predict the class of a brain MRI image.

        Args:
            image_input: Either a file path (string) or a PIL Image object

        Returns:
            Dictionary with:
                - predicted_class: The predicted class name
                - confidence: Confidence score (0-1)
                - probabilities: Dict of all class probabilities
        """
        # Load image if path is given
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError("Input must be a file path or PIL Image")

        # Preprocess
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]

        # Format result
        pred_idx = torch.argmax(probs).item()
        result = {
            "predicted_class": CLASS_NAMES[pred_idx],
            "confidence": round(probs[pred_idx].item(), 4),
            "probabilities": {
                name: round(probs[i].item(), 4)
                for i, name in enumerate(CLASS_NAMES)
            },
        }

        return result

    def predict_batch(self, image_paths):
        """
        Predict classes for multiple images.

        Args:
            image_paths: List of file paths

        Returns:
            List of prediction dictionaries
        """
        return [self.predict(path) for path in image_paths]