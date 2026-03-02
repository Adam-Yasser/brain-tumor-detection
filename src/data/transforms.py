"""
Image transforms for training, validation, and testing.

Training:   Augmentation + preprocessing (makes model more robust)
Val/Test:   Preprocessing only (we want consistent evaluation)
"""

from torchvision import transforms


# ImageNet statistics — used for normalization because our pretrained
# model was trained on ImageNet with these exact values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Standard input size for pretrained models
IMAGE_SIZE = 224


def get_train_transforms():
    """
    Transforms for training data.

    Includes augmentation to make the model more robust.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_strong_train_transforms():
    """
    Stronger augmentation for optimization experiments.

    More aggressive transforms to reduce overfitting and help
    the model generalize better on confusing cases like
    glioma vs meningioma.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 64, IMAGE_SIZE + 64)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms():
    """
    Transforms for validation and testing data.

    No augmentation — we want consistent, reproducible evaluation.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])