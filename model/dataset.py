"""
Dataset loaders for the Adversarial Face Detection pipeline.

Supports:
  - FaceDataset      : loads clean face images from a directory tree
  - AdversarialDataset: loads a mixed clean+adversarial set with type labels
  - DatasetBuilder   : builds train/val/test DataLoaders in one call
"""

import os
import csv
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torchvision import transforms
from PIL import Image, ImageFile

# Allow loading of truncated/incomplete JPEG files (common in large face datasets)
ImageFile.LOAD_TRUNCATED_IMAGES = True

from model.config import Config


# ────────────────────────────────────────────────────────────────────
#  Standard transforms
# ────────────────────────────────────────────────────────────────────

def get_train_transform() -> transforms.Compose:
    """Augmentation-heavy transform for training set."""
    return transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE + 16, Config.IMAGE_SIZE + 16)),
        transforms.RandomCrop(Config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.1, hue=0.05),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.NORMALIZE_MEAN,
                             std=Config.NORMALIZE_STD),
    ])


def get_eval_transform() -> transforms.Compose:
    """Minimal transform for validation / test / inference."""
    return transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.NORMALIZE_MEAN,
                             std=Config.NORMALIZE_STD),
    ])


# ────────────────────────────────────────────────────────────────────
#  1. FaceDataset  (clean images, identity-aware)
# ────────────────────────────────────────────────────────────────────

class FaceDataset(Dataset):
    """
    Loads face images from a directory with the structure:
        root/
          person_001/
            img_01.jpg
            img_02.jpg
          person_002/
            ...

    Args:
        root      : path to directory with identity sub-folders
        transform : torchvision transform to apply
        label_type: "identity" → class = person ID
                    "binary"   → class = 0 (clean)
    """

    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, root: str, transform=None, label_type: str = "identity"):
        self.root = Path(root)
        self.transform = transform or get_eval_transform()
        self.label_type = label_type

        self.samples: list[Tuple[Path, int]] = []
        self.class_to_idx: dict[str, int] = {}
        self._scan()

    def _scan(self):
        identities = sorted(
            [d for d in self.root.iterdir() if d.is_dir()]
        )
        for idx, identity_dir in enumerate(identities):
            self.class_to_idx[identity_dir.name] = idx
            for img_path in identity_dir.rglob("*"):
                if img_path.suffix.lower() in self.VALID_EXTENSIONS:
                    label = idx if self.label_type == "identity" else 0
                    self.samples.append((img_path, label))

        if not self.samples:
            raise RuntimeError(f"No images found under: {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except (OSError, Exception):
            # Return a blank image for rare corrupt files (~0.7% of CASIA-WebFace)
            image = Image.new("RGB", (Config.IMAGE_SIZE, Config.IMAGE_SIZE), color=0)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


# ────────────────────────────────────────────────────────────────────
#  2. AdversarialDataset  (mixed clean + adversarial, 3-class)
# ────────────────────────────────────────────────────────────────────

class AdversarialDataset(Dataset):
    """
    Loads images from a CSV file produced by the adversarial generation
    pipeline (model/adversarial.py).

    CSV format:
        image_path, label, image_type
        data/clean/img.jpg, 0, clean
        data/adversarial/img_fgsm.jpg, 1, adversarial

    Label mapping (aligns with Config.CLASS_NAMES):
        0 → normal
        1 → adversarial
        2 → suspicious
    """

    LABEL_MAP = {
        "clean": 0,
        "normal": 0,
        "adversarial": 1,
        "suspicious": 2,
    }

    def __init__(self, csv_path: str, transform=None):
        self.transform = transform or get_eval_transform()
        self.samples: list[Tuple[str, int]] = []
        self._load_csv(csv_path)

    def _load_csv(self, csv_path: str):
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = int(row.get("label",
                            self.LABEL_MAP.get(row.get("image_type", "clean"), 0)))
                self.samples.append((row["image_path"], label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except (OSError, Exception):
            image = Image.new("RGB", (Config.IMAGE_SIZE, Config.IMAGE_SIZE), color=0)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


# ────────────────────────────────────────────────────────────────────
#  3. DatasetBuilder  (one-call DataLoader factory)
# ────────────────────────────────────────────────────────────────────

class DatasetBuilder:
    """
    Convenience class: builds train / val / test DataLoaders.

    Two modes:
      A) From directory (clean images only — for pre-adversarial training)
         builder = DatasetBuilder.from_directory(clean_dir)

      B) From CSV (mixed clean+adversarial — for final model training)
         builder = DatasetBuilder.from_csv(csv_path)
    """

    def __init__(self, dataset: Dataset,
                 val_split: float = Config.VALIDATION_SPLIT,
                 test_split: float = 0.1,
                 batch_size: int = Config.BATCH_SIZE,
                 num_workers: int = Config.NUM_WORKERS,
                 seed: int = 42):
        self.batch_size = batch_size
        self.num_workers = num_workers

        total = len(dataset)
        n_test = int(total * test_split)
        n_val = int(total * val_split)
        n_train = total - n_val - n_test

        generator = torch.Generator().manual_seed(seed)
        self.train_ds, self.val_ds, self.test_ds = random_split(
            dataset, [n_train, n_val, n_test], generator=generator
        )

    # ── DataLoader properties ──

    @property
    def train_loader(self) -> DataLoader:
        return DataLoader(self.train_ds,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          drop_last=True)

    @property
    def val_loader(self) -> DataLoader:
        return DataLoader(self.val_ds,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=True)

    @property
    def test_loader(self) -> DataLoader:
        return DataLoader(self.test_ds,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=True)

    # ── Factory methods ──

    @classmethod
    def from_directory(cls, clean_dir: str, **kwargs) -> "DatasetBuilder":
        """Build from a clean-image directory tree."""
        dataset = FaceDataset(
            root=clean_dir,
            transform=get_train_transform(),
            label_type="binary",
        )
        return cls(dataset, **kwargs)

    @classmethod
    def from_csv(cls, csv_path: str, **kwargs) -> "DatasetBuilder":
        """Build from a CSV describing a mixed clean+adversarial dataset."""
        dataset = AdversarialDataset(
            csv_path=csv_path,
            transform=get_train_transform(),
        )
        return cls(dataset, **kwargs)

    def __repr__(self) -> str:
        return (f"DatasetBuilder("
                f"train={len(self.train_ds)}, "
                f"val={len(self.val_ds)}, "
                f"test={len(self.test_ds)})")
