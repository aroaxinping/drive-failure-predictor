"""Build the drive-level dataset from the daily parquet.

Uses polars for the heavy lifting (28M rows, ~13 GB) — its lazy evaluation
reads data in streaming fashion without loading everything into RAM at once.
The output is a standard parquet that the notebooks read with pandas.
"""

from pathlib import Path

import numpy as np
import polars as pl

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"

SMART_RAW_COLS = [
    "smart_1_raw",
    "smart_3_raw",
    "smart_4_raw",
    "smart_5_raw",
    "smart_7_raw",
    "smart_9_raw",
    "smart_10_raw",
    "smart_12_raw",
    "smart_187_raw",
    "smart_188_raw",
    "smart_190_raw",
    "smart_192_raw",
    "smart_193_raw",
    "smart_194_raw",
    "smart_197_raw",
    "smart_198_raw",
    "smart_199_raw",
    "smart_240_raw",
    "smart_241_raw",
    "smart_242_raw",
]

CRITICAL_SMART = [
    "smart_5_raw",
    "smart_187_raw",
    "smart_188_raw",
    "smart_197_raw",
    "smart_198_raw",
]

MANUFACTURER_RULES = [
    (r"(?i)^st|seagate", "Seagate"),
    (r"(?i)^wdc|western", "WDC"),
    (r"(?i)hgst|^hus|^hms", "HGST"),
    (r"(?i)toshiba", "Toshiba"),
]


def _slope(series: pl.Series) -> float:
    """Compute linear regression slope over a polars Series."""
    y = series.drop_nulls().to_numpy()
    if len(y) < 3:
        return 0.0
    x = np.arange(len(y), dtype=np.float64)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - x_mean) * (y - y_mean)).sum() / denom)


def _build_slope_exprs(critical_cols: list[str]) -> list[pl.Expr]:
    """Build slope (linear trend) expressions for critical SMART columns."""
    exprs = []
    for col in critical_cols:
        exprs.append(
            pl.col(col)
            .map_batches(lambda s: pl.Series([_slope(s)]), return_dtype=pl.Float64)
            .first()
            .alias(f"{col}_slope")
        )
    return exprs


def main():
    src = PROCESSED_DIR / "q1_2025_selected.parquet"
    print(f"Reading {src.name}...")
    df = pl.scan_parquet(src)

    schema_cols = df.collect_schema().names()
    smart_cols = [c for c in SMART_RAW_COLS if c in schema_cols]
    critical_cols = [c for c in CRITICAL_SMART if c in schema_cols]
    print(f"  SMART columns: {len(smart_cols)}, critical: {len(critical_cols)}")

    # Drop SMART columns with >80% nulls
    null_check = df.select([pl.col(c).null_count().alias(c) for c in smart_cols]).collect()
    total_rows = df.select(pl.len()).collect().item()
    sparse = [c for c in smart_cols if null_check[c][0] / total_rows > 0.80]
    smart_cols = [c for c in smart_cols if c not in sparse]
    critical_cols = [c for c in critical_cols if c not in sparse]
    if sparse:
        print(f"  Dropped {len(sparse)} sparse columns: {sparse}")

    df = df.sort("serial_number", "date")

    # Aggregation: last, max, mean, std for each SMART column
    agg_exprs = []
    for col in smart_cols:
        agg_exprs.extend([
            pl.col(col).last().alias(f"{col}_last"),
            pl.col(col).max().alias(f"{col}_max"),
            pl.col(col).mean().alias(f"{col}_mean"),
            pl.col(col).std().alias(f"{col}_std"),
        ])

    agg_exprs.extend([
        pl.col("failure").max().alias("failure"),
        pl.col("capacity_bytes").last().alias("capacity_bytes"),
        pl.col("model").last().alias("model"),
        pl.col("date").count().alias("days_observed"),
        (pl.col("date").max() - pl.col("date").min()).dt.total_days().alias("drive_age_days"),
    ])

    print("Aggregating to drive level (stats)...")
    drives = df.group_by("serial_number").agg(agg_exprs).collect()
    print(f"  {len(drives):,} drives")

    # Slopes for critical SMART attributes
    print("Computing slopes for critical SMART attributes...")
    df_eager = df.select(["serial_number"] + critical_cols).collect()
    slopes = df_eager.group_by("serial_number").agg(_build_slope_exprs(critical_cols))
    drives = drives.join(slopes, on="serial_number", how="left")

    # Manufacturer
    expr = pl.lit("Other")
    for pattern, name in reversed(MANUFACTURER_RULES):
        expr = pl.when(pl.col("model").str.contains(pattern)).then(pl.lit(name)).otherwise(expr)
    drives = drives.with_columns(expr.alias("manufacturer"))

    drives = drives.fill_null(0).fill_nan(0)

    failures = drives["failure"].sum()
    pct = failures / len(drives) * 100
    print(f"  {failures} failures ({pct:.3f}%)")
    print(f"  {drives.shape[1]} features")

    out = PROCESSED_DIR / "drives.parquet"
    drives.write_parquet(out)
    print(f"Saved to {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
