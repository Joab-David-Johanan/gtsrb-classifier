from pathlib import Path

import torch
import yaml

from gtsrb.data.dataloader import build_dataloaders
from gtsrb.models.classifier import GTSRBClassifier
from gtsrb.training.trainer import Trainer
from gtsrb.utils import logger


def main() -> None:
    config_path = Path("configs/config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    torch.manual_seed(config["training"]["seed"])

    logger.info("Building dataloaders...")
    train_loader, val_loader = build_dataloaders(
        data_root=config["data"]["root"],
        image_size=config["data"]["image_size"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )

    logger.info("Building model...")
    model = GTSRBClassifier(
        num_classes=config["data"]["num_classes"],
        dropout=config["model"]["dropout"],
        pretrained=config["model"]["pretrained"],
    )

    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.train()


if __name__ == "__main__":
    main()
