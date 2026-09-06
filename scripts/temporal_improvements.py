"""Test three improvements to temporal validation:

1. Normalize by window length — divide std/slope features by days_observed
2. Rolling window 30d — aggregate only the last 30 days per drive per period
3. Combined — rolling window + normalization

Compares all variants against the baseline temporal split (F1 ≈ 0, AUC-ROC = 0.82).
"""

import json, warnings
import numpy as np
import pandas as pd
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, f1_score,
    roc_auc_score, average_precision_score, precision_recall_curve,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"

BLUE = "2a78d6"
GREEN = "1a7f37"
RED = "cf222e"
GRAY = "57606a"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "font.size": 11,
})

SMART_RAW_COLS = [
    "smart_1_raw", "smart_3_raw", "smart_4_raw", "smart_5_raw",
    "smart_7_raw", "smart_9_raw", "smart_10_raw", "smart_12_raw",
    "smart_187_raw", "smart_188_raw", "smart_190_raw", "smart_192_raw",
    "smart_193_raw", "smart_194_raw", "smart_197_raw", "smart_198_raw",
    "smart_199_raw", "smart_240_raw", "smart_241_raw", "smart_242_raw",
]

CRITICAL_SMART = [
    "smart_5_raw", "smart_187_raw", "smart_188_raw",
    "smart_197_raw", "smart_198_raw",
]

MANUFACTURER_RULES = [
    (r"(?i)^st|seagate", "Seagate"),
    (r"(?i)^wdc|western", "WDC"),
    (r"(?i)hgst|^hus|^hms", "HGST"),
    (r"(?i)toshiba", "Toshiba"),
]


def _slope(series: pl.Series) -> float:
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


def aggregate_period(daily_df: pl.LazyFrame, smart_cols: list, critical_cols: list) -> pl.DataFrame:
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

    daily_sorted = daily_df.sort("serial_number", "date")
    drives = daily_sorted.group_by("serial_number").agg(agg_exprs).collect()

    slope_exprs = []
    for col in critical_cols:
        slope_exprs.append(
            pl.col(col)
            .map_batches(lambda s: pl.Series([_slope(s)]), return_dtype=pl.Float64)
            .first()
            .alias(f"{col}_slope")
        )
    df_eager = daily_sorted.select(["serial_number"] + critical_cols).collect()
    slopes = df_eager.group_by("serial_number").agg(slope_exprs)
    drives = drives.join(slopes, on="serial_number", how="left")

    expr = pl.lit("Other")
    for pattern, name in reversed(MANUFACTURER_RULES):
        expr = pl.when(pl.col("model").str.contains(pattern)).then(pl.lit(name)).otherwise(expr)
    drives = drives.with_columns(expr.alias("manufacturer"))
    drives = drives.fill_null(0).fill_nan(0)
    return drives


def keep_last_n_days(daily_df: pl.LazyFrame, n: int) -> pl.LazyFrame:
    """Keep only the last N days of observations per drive."""
    return (
        daily_df
        .sort("serial_number", "date")
        .with_columns(
            pl.col("date").rank("ordinal", descending=True)
            .over("serial_number")
            .alias("_rank")
        )
        .filter(pl.col("_rank") <= n)
        .drop("_rank")
    )


def prepare_features(df_pl: pl.DataFrame, feature_cols: list, normalize: bool = False) -> tuple:
    df = df_pl.to_pandas()
    df = df.drop(columns=["serial_number", "model"], errors="ignore")
    df = pd.get_dummies(df, columns=["manufacturer"], drop_first=True, dtype=int)

    if normalize:
        days = df["days_observed"].replace(0, 1)
        std_cols = [c for c in df.columns if c.endswith("_std")]
        slope_cols = [c for c in df.columns if c.endswith("_slope")]
        for col in std_cols + slope_cols:
            df[col] = df[col] / days

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    return df[feature_cols], df["failure"]


def train_and_eval(X_train, y_train, X_test, y_test, label: str) -> dict:
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    spw = n_neg / n_pos if n_pos > 0 else 1.0

    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        scale_pos_weight=spw, eval_metric="aucpr",
        random_state=42, n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    y_proba = xgb.predict_proba(X_test)[:, 1]

    auc_roc = roc_auc_score(y_test, y_proba)
    auc_pr = average_precision_score(y_test, y_proba)

    # Find best F1 threshold
    thresholds = np.arange(0.001, 1.0, 0.005)
    f1s = [f1_score(y_test, (y_proba >= t).astype(int)) for t in thresholds]
    best_idx = np.argmax(f1s)
    best_t = thresholds[best_idx]
    best_f1 = f1s[best_idx]

    y_pred_best = (y_proba >= best_t).astype(int)

    print(f"\n--- {label} ---")
    print(f"AUC-ROC: {auc_roc:.4f}  AUC-PR: {auc_pr:.4f}")
    print(f"Best threshold: {best_t:.3f} → F1 = {best_f1:.4f}")
    print(classification_report(y_test, y_pred_best, target_names=["healthy", "failure"]))

    return {
        "label": label,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "best_f1": best_f1,
        "best_threshold": best_t,
        "y_proba": y_proba,
    }


def main():
    src = DATA / "q1_2025_selected.parquet"
    if not src.exists():
        print(f"ERROR: {src} not found")
        return

    daily = pl.scan_parquet(src)
    schema_cols = daily.collect_schema().names()
    smart_cols = [c for c in SMART_RAW_COLS if c in schema_cols]
    critical_cols = [c for c in CRITICAL_SMART if c in schema_cols]

    cutoff = pl.lit("2025-03-01").cast(pl.Date)
    train_daily = daily.filter(pl.col("date") < cutoff)
    test_daily = daily.filter(pl.col("date") >= cutoff)

    with open(DATA / "feature_selection.json") as f:
        feature_cols = json.load(f)["feature_cols"]

    results = []

    # === Variant 0: Baseline (no fix) ===
    print("=" * 60)
    print("BASELINE: full-period aggregation, no normalization")
    print("=" * 60)
    train_drives = aggregate_period(train_daily, smart_cols, critical_cols)
    test_drives = aggregate_period(test_daily, smart_cols, critical_cols)
    print(f"Train: {len(train_drives):,} drives, {train_drives['failure'].sum()} failures")
    print(f"Test: {len(test_drives):,} drives, {test_drives['failure'].sum()} failures")

    X_tr, y_tr = prepare_features(train_drives, feature_cols, normalize=False)
    X_te, y_te = prepare_features(test_drives, feature_cols, normalize=False)
    results.append(train_and_eval(X_tr, y_tr, X_te, y_te, "Baseline"))

    # === Variant 1: Normalize by window length ===
    print("\n" + "=" * 60)
    print("VARIANT 1: Normalize std/slope by days_observed")
    print("=" * 60)
    X_tr_n, y_tr_n = prepare_features(train_drives, feature_cols, normalize=True)
    X_te_n, y_te_n = prepare_features(test_drives, feature_cols, normalize=True)
    results.append(train_and_eval(X_tr_n, y_tr_n, X_te_n, y_te_n, "Normalized"))

    # === Variant 2: Rolling window 30 days ===
    print("\n" + "=" * 60)
    print("VARIANT 2: Rolling window (last 30 days per drive)")
    print("=" * 60)
    train_30d = keep_last_n_days(train_daily, 30)
    test_30d = keep_last_n_days(test_daily, 30)

    train_drives_30 = aggregate_period(train_30d, smart_cols, critical_cols)
    test_drives_30 = aggregate_period(test_30d, smart_cols, critical_cols)
    print(f"Train: {len(train_drives_30):,} drives, {train_drives_30['failure'].sum()} failures")
    print(f"Test: {len(test_drives_30):,} drives, {test_drives_30['failure'].sum()} failures")

    X_tr_30, y_tr_30 = prepare_features(train_drives_30, feature_cols, normalize=False)
    X_te_30, y_te_30 = prepare_features(test_drives_30, feature_cols, normalize=False)
    results.append(train_and_eval(X_tr_30, y_tr_30, X_te_30, y_te_30, "Rolling 30d"))

    # === Variant 3: Rolling 30d + normalize ===
    print("\n" + "=" * 60)
    print("VARIANT 3: Rolling 30d + normalize")
    print("=" * 60)
    X_tr_30n, y_tr_30n = prepare_features(train_drives_30, feature_cols, normalize=True)
    X_te_30n, y_te_30n = prepare_features(test_drives_30, feature_cols, normalize=True)
    results.append(train_and_eval(X_tr_30n, y_tr_30n, X_te_30n, y_te_30n, "Rolling 30d + norm"))

    # === Summary table ===
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Variant':<25} {'AUC-ROC':>8} {'AUC-PR':>8} {'Best F1':>8} {'Threshold':>10}")
    print("-" * 65)
    for r in results:
        print(f"{r['label']:<25} {r['auc_roc']:>8.4f} {r['auc_pr']:>8.4f} {r['best_f1']:>8.4f} {r['best_threshold']:>10.3f}")

    # === Figure ===
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = [f"#{GRAY}", f"#{BLUE}", f"#{GREEN}", f"#{GREEN}"]

    # Panel 1: AUC-ROC and Best F1
    labels = [r["label"] for r in results]
    x = np.arange(len(results))
    w = 0.35
    bars1 = axes[0].bar(x - w/2, [r["auc_roc"] for r in results], w,
                         label="AUC-ROC", color=colors, alpha=0.7, edgecolor="white")
    bars2 = axes[0].bar(x + w/2, [r["best_f1"] for r in results], w,
                         label="Best F1", color=colors, edgecolor="white")

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.005:
                axes[0].text(bar.get_x() + bar.get_width()/2, h + 0.01,
                             f"{h:.3f}", ha="center", fontsize=8, fontweight="bold")

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    axes[0].set_ylim(0, 1.1)
    axes[0].set_title("Temporal validation improvements", fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Panel 2: PR curves
    for r, c in zip(results, colors):
        precision, recall, _ = precision_recall_curve(y_te, r["y_proba"])
        axes[1].plot(recall, precision, color=c, linewidth=2, label=r["label"])

    baseline_rate = y_te.mean()
    axes[1].axhline(y=baseline_rate, color="#d0d7de", linestyle="--", label=f"baseline ({baseline_rate:.4f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall curves (temporal)", fontweight="bold")
    axes[1].legend(fontsize=8)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(FIGURES / "temporal_improvements.png")
    plt.close(fig)
    print(f"\nSaved figures/temporal_improvements.png")


if __name__ == "__main__":
    main()
