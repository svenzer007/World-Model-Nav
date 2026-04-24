"""Self-contained 2D simulator and dataset generator for local navigation."""

from .config import ACTIONS, DatasetConfig, ExpertConfig, LidarConfig, MapConfig, RobotConfig

__all__ = [
    "ACTIONS",
    "DatasetConfig",
    "ExpertConfig",
    "LidarConfig",
    "MapConfig",
    "RobotConfig",
]

__version__ = "0.1.0"
