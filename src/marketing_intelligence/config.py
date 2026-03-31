from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    project_root: Path

    @property
    def raw_dir(self) -> Path:
        return self.project_root / "data" / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.project_root / "data" / "processed"

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"


RANDOM_SEED = 42
TARGET_CONVERSION_WINDOW_DAYS = 30
RETENTION_REPEAT_PURCHASE_DAYS = 180
