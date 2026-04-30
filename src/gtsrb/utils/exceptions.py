class GtsrbError(Exception):
    """Base exception for all GTSRB project errors."""


class DatasetError(GtsrbError):
    """Raised when dataset loading or processing fails."""


class ConfigError(GtsrbError):
    """Raised when config is missing required fields or has invalid values."""


class CheckpointError(GtsrbError):
    """Raised when saving or loading a model checkpoint fails."""


class ModelError(GtsrbError):
    """Raised when model architecture or forward pass fails."""


class AugmentationError(GtsrbError):
    """Raised when the image augmentation pipeline fails."""
