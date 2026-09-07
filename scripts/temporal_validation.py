"""Temporal validation: train on Jan-Feb aggregation, test on March aggregation.

Build TWO separate aggregated datasets from the raw daily data:
  - Train: aggregate daily readings from Jan 1 - Feb 28 → one row per drive
  - Test:  aggregate daily readings from Mar 1 - Mar 31 → one row per drive

Key finding: F1 drops from 0.96 (random split) to near zero (temporal).
BUT AUC-ROC = 0.82 — the model HAS learned signal. The problem is
feature distribution shift: aggregating over 59 days (Jan-Feb) vs 31 days
(March) produces different scales for days_observed, std, and slope features.
The decision boundary calibrated on Jan-Feb doesn't transfer.

This validates the need for:
  1. Time-invariant feature engineering (fixed rolling windows)
  2. Multi-quarter training data
  3. Periodic model retraining in production
"""

import json, warnings
import numpy as np
import pandas as pd
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    roc_auc_score, average_precision_score, precision_recall_curve,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
MODELS = ROOT / "models"

BLUE = "#2a78d6"
GREEN = "#1a7f37"
RED = "#cf222e"
GRAY = "#57606a"
LIGHT_GRAY = "#d0d7de"

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


def prepare_features(df_pl: pl.DataFrame, feature_cols: list) -> tuple:
    df = df_pl.to_pandas()
    df = df.drop(columns=["serial_number", "model"], errors="ignore")
    df = pd.get_dummies(df, columns=["manufacturer"], drop_first=True, dtype=int)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    return df[feature_cols], df["failure"]


def main():
    src = DATA / "q1_2025_selected.parquet"
    if not src.exists():
        print(f"ERROR: {src} not found. Run scripts/raw_to_parquet.py first.")
        return

    daily = pl.scan_parquet(src)
    schema_cols = daily.collect_schema().names()
    smart_cols = [c for c in SMART_RAW_COLS if c in schema_cols]
    critical_cols = [c for c in CRITICAL_SMART if c in schema_cols]

    cutoff = pl.lit("2025-03-01").cast(pl.Date)
    train_daily = daily.filter(pl.col("date") < cutoff)
    test_daily = daily.filter(pl.col("date") >= cutoff)

    train_rows = train_daily.select(pl.len()).collect().item()
    test_rows = test_daily.select(pl.len()).collect().item()
    print(f"Daily rows — train (Jan-Feb): {train_rows:,}  test (March): {test_rows:,}")

    print("\nAggregating Jan-Feb (train)...")
    train_drives = aggregate_period(train_daily, smart_cols, critical_cols)
    train_fail = train_drives["failure"].sum()
    print(f"  {len(train_drives):,} drives, {train_fail} failures ({train_fail/len(train_drives)*100:.3f}%)")

    print("Aggregating March (test)...")
    test_drives = aggregate_period(test_daily, smart_cols, critical_cols)
    test_fail = test_drives["failure"].sum()
    print(f"  {len(test_drives):,} drives, {test_fail} failures ({test_fail/len(test_drives)*100:.3f}%)")

    with open(DATA / "feature_selection.json") as f:
        artifacts = json.load(f)
    feature_cols = artifacts["feature_cols"]

    X_train, y_train = prepare_features(train_drives, feature_cols)
    X_test, y_test = prepare_features(test_drives, feature_cols)

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    spw = n_neg / n_pos if n_pos > 0 else 1.0

    # --- XGBoost (balanced) ---
    print("\n--- XGBoost (temporal, balanced) ---")
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                         scale_pos_weight=spw, eval_metric="aucpr",
                         random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    y_proba_xgb = xgb.predict_proba(X_test)[:, 1]
    auc_roc = roc_auc_score(y_test, y_proba_xgb)
    auc_pr = average_precision_score(y_test, y_proba_xgb)
    y_pred_xgb = xgb.predict(X_test)
    f1_xgb = f1_score(y_test, y_pred_xgb)
    cm_xgb = confusion_matrix(y_test, y_pred_xgb)
    print(f"F1: {f1_xgb:.4f}  AUC-ROC: {auc_roc:.4f}  AUC-PR: {auc_pr:.4f}")
    print(classification_report(y_test, y_pred_xgb, target_names=["healthy", "failure"]))

    # --- Feature distribution shift analysis ---
    print("=== Feature distribution shift ===")
    shift_cols = ["days_observed", "drive_age_days"]
    for col in shift_cols:
        if col in X_train.columns:
            tr_med = X_train[col].median()
            te_med = X_test[col].median()
            print(f"  {col}: train median={tr_med:.0f}, test median={te_med:.0f}")

    std_cols = [c for c in feature_cols if c.endswith("_std")]
    if std_cols:
        tr_stds = X_train[std_cols].median().mean()
        te_stds = X_test[std_cols].median().mean()
        print(f"  *_std features median-of-medians: train={tr_stds:.2f}, test={te_stds:.2f}")

    # --- Comparison with random split ---
    print("\n=== Temporal vs Random split ===\n")
    xgb_rand = joblib.load(MODELS / "final_model.joblib")
    df_full = pd.read_parquet(DATA / "drives.parquet")
    df_full = df_full.drop(columns=["serial_number", "model", "last_date"], errors="ignore")
    df_full = pd.get_dummies(df_full, columns=["manufacturer"], drop_first=True, dtype=int)
    for col in feature_cols:
        if col not in df_full.columns:
            df_full[col] = 0

    rand_idx = artifacts["test_idx"]
    X_rand = df_full.loc[rand_idx, feature_cols]
    y_rand = df_full.loc[rand_idx, "failure"]

    y_proba_rand = xgb_rand.predict_proba(X_rand)[:, 1]
    f1_rand = f1_score(y_rand, xgb_rand.predict(X_rand))
    auc_roc_rand = roc_auc_score(y_rand, y_proba_rand)
    auc_pr_rand = average_precision_score(y_rand, y_proba_rand)

    print(f"{'Metric':<20} {'Random':>10} {'Temporal':>10} {'Delta':>10}")
    print("-" * 55)
    print(f"{'F1 Score':<20} {f1_rand:>10.4f} {f1_xgb:>10.4f} {f1_xgb - f1_rand:>+10.4f}")
    print(f"{'AUC-ROC':<20} {auc_roc_rand:>10.4f} {auc_roc:>10.4f} {auc_roc - auc_roc_rand:>+10.4f}")
    print(f"{'AUC-PR':<20} {auc_pr_rand:>10.4f} {auc_pr:>10.4f} {auc_pr - auc_pr_rand:>+10.4f}")

    # --- Figure ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: F1 and AUC comparison
    metrics = ["F1", "AUC-ROC", "AUC-PR"]
    rand_vals = [f1_rand, auc_roc_rand, auc_pr_rand]
    temp_vals = [f1_xgb, auc_roc, auc_pr]

    x = np.arange(len(metrics))
    w = 0.3
    bars1 = axes[0].bar(x - w/2, rand_vals, w, label="Random split", color=GRAY, edgecolor="white")
    bars2 = axes[0].bar(x + w/2, temp_vals, w, label="Temporal split", color=BLUE, edgecolor="white")

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                axes[0].text(bar.get_x() + bar.get_width()/2, h + 0.02,
                             f"{h:.2f}", ha="center", fontsize=9, fontweight="bold")

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    axes[0].set_ylim(0, 1.15)
    axes[0].set_title("XGBoost: random vs temporal", fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Panel 2: Precision-Recall curve (temporal)
    precision, recall, _ = precision_recall_curve(y_test, y_proba_xgb)
    axes[1].plot(recall, precision, color=BLUE, linewidth=2)
    axes[1].axhline(y=y_test.mean(), color=LIGHT_GRAY, linestyle="--", label=f"baseline ({y_test.mean():.4f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Temporal PR curve (AUC={auc_pr:.4f})", fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    # Panel 3: Feature shift — days_observed distribution
    ax3 = axes[2]
    train_days = X_train["days_observed"].values if "days_observed" in X_train.columns else np.array([])
    test_days = X_test["days_observed"].values if "days_observed" in X_test.columns else np.array([])
    if len(train_days) > 0 and len(test_days) > 0:
        bins = np.linspace(0, max(train_days.max(), test_days.max()) + 1, 40)
        ax3.hist(train_days, bins=bins, alpha=0.6, color=GRAY, label="Train (Jan-Feb)", density=True)
        ax3.hist(test_days, bins=bins, alpha=0.6, color=BLUE, label="Test (March)", density=True)
        ax3.set_xlabel("days_observed")
        ax3.set_ylabel("Density")
        ax3.set_title("Feature distribution shift", fontweight="bold")
        ax3.legend(fontsize=9)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(FIGURES / "temporal_validation.png")
    plt.close(fig)
    print(f"\nSaved figures/temporal_validation.png")

    # --- Takeaway ---
    print(f"\n{'='*55}")
    print("TAKEAWAY")
    print(f"{'='*55}")
    print(f"AUC-ROC = {auc_roc:.2f} → the model HAS learned discriminative signal.")
    print(f"F1 ≈ 0 → but the decision boundary doesn't transfer across time windows.")
    print(f"Root cause: feature distribution shift (59-day vs 31-day aggregation).")
    print(f"For production: use fixed-length rolling windows and multi-quarter data.")


if __name__ == "__main__":
    main()
