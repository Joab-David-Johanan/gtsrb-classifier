import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import albumentations as A
from gtsrb.utils import DatasetError, logger


class GTSRBDataset(Dataset):
    NUM_CLASSES = 43

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: A.Compose | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform

        # List of (image_path, class_id) — images loaded lazily in __getitem__
        self.samples: list[tuple[Path, int]] = []
        self._load_samples()

        logger.info(
            "Loaded GTSRB {split} split: {n} images",
            split=self.split,
            n=len(self.samples),
        )

    def _load_samples(self) -> None:
        if self.split == "train":
            self._load_train_samples()
        elif self.split == "test":
            self._load_test_samples()
        else:
            raise DatasetError(
                f"Unknown split '{self.split}'. Expected 'train' or 'test'."
            )

        if len(self.samples) == 0:
            raise DatasetError(
                f"No images found for split '{self.split}' in {self.root}. "
                "Run notebooks/01_explore_data.ipynb to download the dataset first."
            )

    def _load_train_samples(self) -> None:
        train_root = self.root / "GTSRB" / "Training"
        if not train_root.exists():
            raise DatasetError(
                f"Training directory not found: {train_root}. "
                "Run notebooks/01_explore_data.ipynb to download the dataset first."
            )
        for class_dir in sorted(train_root.iterdir()):
            if not class_dir.is_dir():
                continue
            class_id = int(class_dir.name)
            for img_path in sorted(class_dir.glob("*.ppm")):
                self.samples.append((img_path, class_id))

    def _load_test_samples(self) -> None:
        csv_path = self.root / "GTSRB" / "Final_Test" / "Images" / "GT-final_test.csv"
        test_root = csv_path.parent

        if not csv_path.exists():
            raise DatasetError(f"Test labels CSV not found: {csv_path}")

        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f, delimiter=";"):
                self.samples.append((test_root / row["Filename"], int(row["ClassId"])))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        try:
            image = np.array(Image.open(img_path).convert("RGB"))
        except Exception as e:
            raise DatasetError(f"Failed to load image {img_path}: {e}") from e

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label

    def get_class_counts(self) -> list[int]:
        counts = [0] * self.NUM_CLASSES
        for _, label in self.samples:
            counts[label] += 1
        return counts
