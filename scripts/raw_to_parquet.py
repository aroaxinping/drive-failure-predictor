"""Convert raw Backblaze daily CSVs to a single parquet for the pipeline.

Reads all CSV files from data/raw/data_Q1_2025/, selects relevant columns
(serial_number, date, model, capacity_bytes, failure, SMART raw attributes),
and writes data/processed/q1_2025_selected.parquet.

Uses polars lazy evaluation to handle the ~28M rows without blowing memory.
"""

import glob
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "data_Q1_2025" / "data_Q1_2025"
OUT = ROOT / "data" / "processed" / "q1_2025_selected.parquet"

KEEP_COLS = [
    "date", "serial_number", "model", "capacity_bytes", "failure",
    "smart_1_raw", "smart_3_raw", "smart_4_raw", "smart_5_raw",
    "smart_7_raw", "smart_9_raw", "smart_10_raw", "smart_12_raw",
    "smart_187_raw", "smart_188_raw", "smart_190_raw", "smart_192_raw",
    "smart_193_raw", "smart_194_raw", "smart_197_raw", "smart_198_raw",
    "smart_199_raw", "smart_240_raw", "smart_241_raw", "smart_242_raw",
]


def main():
    csv_files = sorted(glob.glob(str(RAW_DIR / "*.csv")))
    if not csv_files:
        print(f"No CSV files found in {RAW_DIR}")
        print("Expected: data/raw/data_Q1_2025/2025-01-01.csv etc.")
        return

    print(f"Found {len(csv_files)} daily CSV files")

    dfs = []
    for i, f in enumerate(csv_files):
        df = pl.scan_csv(f)
        cols = df.collect_schema().names()
        select = [c for c in KEEP_COLS if c in cols]
        dfs.append(df.select(select))
        if (i + 1) % 15 == 0:
            print(f"  scanned {i + 1}/{len(csv_files)} files...")

    print("Concatenating and writing parquet...")
    combined = pl.concat(dfs)
    combined = combined.with_columns(pl.col("date").cast(pl.Date))
    combined.collect(streaming=True).write_parquet(OUT)

    size_mb = OUT.stat().st_size / 1e6
    row_count = pl.scan_parquet(OUT).select(pl.len()).collect().item()
    print(f"Saved {OUT.name}: {row_count:,} rows, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
