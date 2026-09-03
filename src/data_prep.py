"""Load the drive-level dataset for analysis and modeling."""

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def load_drives(name: str = "drives.parquet") -> pd.DataFrame:
    """Load the drive-level aggregated dataset."""
    return pd.read_parquet(PROCESSED_DIR / name)
