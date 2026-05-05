from gtsrb.utils import logger
from gtsrb.utils.exceptions import (
    AugmentationError,
    CheckpointError,
    ConfigError,
    DatasetError,
    GtsrbError,
    ModelError,
)


def test_logger_importable():
    assert logger is not None


def test_exception_hierarchy():
    assert issubclass(DatasetError, GtsrbError)
    assert issubclass(ConfigError, GtsrbError)
    assert issubclass(CheckpointError, GtsrbError)
    assert issubclass(ModelError, GtsrbError)
    assert issubclass(AugmentationError, GtsrbError)
