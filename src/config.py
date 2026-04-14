"""Central configuration loaded from environment."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    data_dir: Path = Path(os.getenv("DATA_DIR", "./data"))
    model_dir: Path = Path(os.getenv("MODEL_DIR", "./models"))
    report_dir: Path = Path(os.getenv("REPORT_DIR", "./reports"))
    seasons_start: int = int(os.getenv("SEASONS_START", "2015"))
    seasons_end: int = int(os.getenv("SEASONS_END", "2024"))

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def features_path(self) -> Path:
        return self.processed_dir / "features.parquet"

    def ensure_dirs(self) -> None:
        for p in (self.raw_dir, self.processed_dir, self.model_dir, self.report_dir):
            p.mkdir(parents=True, exist_ok=True)


CFG = Config()
CFG.ensure_dirs()
