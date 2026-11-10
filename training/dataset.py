import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import List, Tuple, Optional, Callable


def get_training_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Data augmentation pipeline tailored for deepfake detection.
    Includes subtle blur, color jitter, affine rotation, and standard normalization.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_validation_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Deterministic validation / evaluation transform.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


class DeepfakeDataset(Dataset):
    """
    PyTorch Dataset for Real / Fake image pairs.
    Expects directory structure:
      root_dir/
        ├── real/  (label 0)
        └── fake/  (label 1)
    """
    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []

        real_dir = os.path.join(root_dir, "real")
        fake_dir = os.path.join(root_dir, "fake")

        extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

        if os.path.exists(real_dir):
            for ext in extensions:
                for img_path in glob.glob(os.path.join(real_dir, ext)):
                    self.samples.append((img_path, 0))  # 0 = Real

        if os.path.exists(fake_dir):
            for ext in extensions:
                for img_path in glob.glob(os.path.join(fake_dir, ext)):
                    self.samples.append((img_path, 1))  # 1 = Fake

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback black image if corrupt
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform is not None:
            image = self.transform(image)

        return image, label
