"""Data loading utilities for MovieLens dataset."""
from pathlib import Path
import pandas as pd

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


def load_raw_data() -> dict[str, pd.DataFrame]:
    """Load all raw MovieLens CSV files."""
    return {
        "movies": pd.read_csv(RAW_DIR / "movies.csv"),
        "ratings": pd.read_csv(RAW_DIR / "ratings.csv"),
        "links": pd.read_csv(RAW_DIR / "links.csv"),
        "tags": pd.read_csv(RAW_DIR / "tags.csv"),
    }


def save_processed(name: str, df: pd.DataFrame) -> None:
    """Save processed DataFrame as parquet."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)


def load_processed(name: str) -> pd.DataFrame:
    """Load processed DataFrame from parquet."""
    return pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")