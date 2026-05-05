from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from gtsrb.models.classifier import GTSRBClassifier
from gtsrb.training.losses import FocalLoss
from gtsrb.utils import CheckpointError, logger


class Trainer:
    def __init__(
        self,
        model: GTSRBClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.epochs = config["training"]["epochs"]
        self.grad_clip = config["training"]["grad_clip"]
        self.mixed_precision = config["training"]["mixed_precision"]

        self.checkpoint_dir = Path(config["paths"]["checkpoints"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = FocalLoss(gamma=2.0)
        self.scaler = GradScaler("cuda", enabled=self.mixed_precision)
        self.best_val_acc = 0.0

        logger.info("Trainer ready — device: {device}", device=self.device)

    def train(self) -> None:
        # Phase 1 — only head, backbone frozen (5 epochs, stable + fast)
        logger.info("Phase 1 — training head only (backbone frozen)")
        self.model.freeze_backbone()
        self._run_epochs(
            lr=self.config["training"]["learning_rate"],
            num_epochs=5,
            phase=1,
        )

        # Phase 2 — all layers, lower lr (remaining epochs, squeezes accuracy)
        logger.info("Phase 2 — fine-tuning all layers (backbone unfrozen)")
        self.model.unfreeze_backbone()
        self._run_epochs(
            lr=self.config["training"]["learning_rate"] / 10,
            num_epochs=self.epochs - 5,
            phase=2,
        )

        logger.info(
            "Training complete — best val_acc: {acc:.2f}%",
            acc=self.best_val_acc * 100,
        )

    def _run_epochs(self, lr: float, num_epochs: int, phase: int) -> None:
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=self.config["training"]["weight_decay"],
        )
        scheduler = OneCycleLR(
            optimizer,
            max_lr=lr,
            steps_per_epoch=len(self.train_loader),
            epochs=num_epochs,
        )

        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = self._train_epoch(optimizer, scheduler)
            val_loss, val_acc = self._val_epoch()

            logger.info(
                "Phase {phase} | Epoch {epoch}/{total} | "
                "train_loss: {tl:.4f} | train_acc: {ta:.2f}% | "
                "val_loss: {vl:.4f} | val_acc: {va:.2f}%",
                phase=phase,
                epoch=epoch,
                total=num_epochs,
                tl=train_loss,
                ta=train_acc * 100,
                vl=val_loss,
                va=val_acc * 100,
            )

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_checkpoint(epoch, val_acc)

    def _train_epoch(
        self, optimizer: AdamW, scheduler: OneCycleLR
    ) -> tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            optimizer.zero_grad()
            with autocast("cuda", enabled=self.mixed_precision):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(optimizer)
            self.scaler.update()
            scheduler.step()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)

        return total_loss / total, correct / total

    @torch.no_grad()
    def _val_epoch(self) -> tuple[float, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in self.val_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            with autocast("cuda", enabled=self.mixed_precision):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)

        return total_loss / total, correct / total

    def _save_checkpoint(self, epoch: int, val_acc: float) -> None:
        path = self.checkpoint_dir / "best_model.pt"
        try:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "val_acc": val_acc,
                },
                path,
            )
            logger.info(
                "New best checkpoint saved — epoch: {epoch} | val_acc: {acc:.2f}%",
                epoch=epoch,
                acc=val_acc * 100,
            )
        except Exception as e:
            raise CheckpointError(f"Failed to save checkpoint to {path}: {e}") from e
