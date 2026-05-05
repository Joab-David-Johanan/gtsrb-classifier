import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet stats — used because our backbone is pretrained on ImageNet
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(image_size: int = 224) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            # Lighting variation — signs appear in sun, shade, night
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            # Camera motion while driving
            A.MotionBlur(blur_limit=5, p=0.2),
            # Signs viewed at an angle from a moving vehicle
            A.Perspective(scale=(0.05, 0.1), p=0.3),
            # Slight rotation from imperfect mounting
            A.Rotate(limit=15, p=0.4),
            # Color shift from different weather and camera sensors
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3
            ),
            # Sensor noise
            A.GaussNoise(std_range=(0.02, 0.1), p=0.2),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_val_transforms(image_size: int = 224) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
