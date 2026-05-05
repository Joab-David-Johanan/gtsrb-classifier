from pathlib import Path

import numpy as np
import torch
import yaml

from gtsrb.data.dataloader import build_dataloaders
from gtsrb.models.classifier import GTSRBClassifier
from gtsrb.utils import CheckpointError, logger

CLASS_NAMES = [
    "Speed 20", "Speed 30", "Speed 50", "Speed 60", "Speed 70",
    "Speed 80", "End Speed 80", "Speed 100", "Speed 120", "No passing",
    "No passing >3.5t", "Right-of-way", "Priority road", "Yield", "Stop",
    "No vehicles", "No vehicles >3.5t", "No entry", "General caution",
    "Danger curve left", "Danger curve right", "Double curve", "Bumpy road",
    "Slippery road", "Road narrows", "Road work", "Traffic signals",
    "Pedestrians", "Children crossing", "Bicycles crossing",
    "Beware ice/snow", "Wild animals", "End all limits", "Turn right ahead",
    "Turn left ahead", "Ahead only", "Straight or right", "Straight or left",
    "Keep right", "Keep left", "Roundabout", "End no passing",
    "End no passing >3.5t",
]


def main() -> None:
    config_path = Path("configs/config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(config["paths"]["checkpoints"]) / "best_model.pt"
    if not checkpoint_path.exists():
        raise CheckpointError(
            f"No checkpoint at {checkpoint_path}. Run scripts/train.py first."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    logger.info(
        "Loaded checkpoint — epoch: {epoch} | val_acc: {acc:.2f}%",
        epoch=checkpoint["epoch"],
        acc=checkpoint["val_acc"] * 100,
    )

    model = GTSRBClassifier(
        num_classes=config["data"]["num_classes"],
        dropout=config["model"]["dropout"],
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    _, val_loader = build_dataloaders(
        data_root=config["data"]["root"],
        image_size=config["data"]["image_size"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            preds = model(images.to(device)).argmax(1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    overall_acc = (all_preds == all_labels).mean()
    logger.info("Overall accuracy: {acc:.2f}%", acc=overall_acc * 100)

    print(f"\n{'Class':<30} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("-" * 60)
    for class_id in range(config["data"]["num_classes"]):
        mask = all_labels == class_id
        if mask.sum() == 0:
            continue
        correct = (all_preds[mask] == all_labels[mask]).sum()
        acc = correct / mask.sum()
        print(f"{CLASS_NAMES[class_id]:<30} {correct:>8} {mask.sum():>8} {acc * 100:>9.1f}%")


if __name__ == "__main__":
    main()
