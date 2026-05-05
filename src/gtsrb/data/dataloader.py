import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from gtsrb.data.dataset import GTSRBDataset
from gtsrb.data.transforms import get_train_transforms, get_val_transforms
from gtsrb.utils import logger


def build_dataloaders(
    data_root: str,
    image_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = GTSRBDataset(
        root=data_root,
        split="train",
        transform=get_train_transforms(image_size),
    )
    val_dataset = GTSRBDataset(
        root=data_root,
        split="test",
        transform=get_val_transforms(image_size),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=_make_sampler(train_dataset),  # replaces shuffle=True
        num_workers=num_workers,
        pin_memory=True,  # pre-loads batch into pinned RAM for faster GPU transfer
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(
        "DataLoaders ready — train: {n_train} batches | val: {n_val} batches",
        n_train=len(train_loader),
        n_val=len(val_loader),
    )

    return train_loader, val_loader


def _make_sampler(dataset: GTSRBDataset) -> WeightedRandomSampler:
    class_counts = dataset.get_class_counts()

    # Rare classes get a higher weight so they appear as often as common ones
    class_weights = [1.0 / count if count > 0 else 0.0 for count in class_counts]
    sample_weights = torch.tensor(
        [class_weights[label] for _, label in dataset.samples]
    )

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
