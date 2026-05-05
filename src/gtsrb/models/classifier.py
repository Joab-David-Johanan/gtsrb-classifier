import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

from gtsrb.utils import ModelError, logger


class GTSRBClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 43,
        dropout: float = 0.3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        in_features = backbone.fc.in_features  # 2048 for ResNet50
        backbone.fc = nn.Identity()  # strip the original ImageNet head
        self.backbone = backbone

        # Custom head: 2048 → 512 → num_classes with regularisation
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

        logger.info(
            "GTSRBClassifier ready — backbone: ResNet50 | classes: {n} | pretrained: {p}",
            n=num_classes,
            p=pretrained,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ModelError(
                f"Expected 4D input (batch, channels, H, W), got {x.ndim}D"
            )
        features = self.backbone(x)
        return self.head(features)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen — only training classification head")

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen — fine-tuning all layers")
