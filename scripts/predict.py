"""Score a single drive's SMART readings against the trained model.

Usage:
    uv run python scripts/predict.py <serial_number>

Loads the drive from data/processed/drives.parquet by serial number,
applies the same feature pipeline as training, and outputs a failure
probability with a risk classification.

In production this would read live SMART data from smartmontools
instead of the static parquet — the interface stays the same.
"""

import json, sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
DATA = ROOT / "data" / "processed"

COST_THRESHOLD = 0.040  # cost-optimal threshold (FN=$10K, FP=$50)


def load_model():
    model = joblib.load(MODELS / "final_model.joblib")
    with open(MODELS / "final_model_meta.json") as f:
        meta = json.load(f)
    return model, meta["feature_cols"]


def predict_drive(serial: str):
    model, feature_cols = load_model()

    df = pd.read_parquet(DATA / "drives.parquet")
    drive = df[df["serial_number"] == serial]

    if drive.empty:
        print(f"Drive {serial} not found in dataset.")
        print(f"Available: {len(df):,} drives. Example serials:")
        samples = df["serial_number"].sample(5, random_state=42).tolist()
        for s in samples:
            print(f"  {s}")
        sys.exit(1)

    row = drive.drop(columns=["serial_number", "model"]).copy()
    row = pd.get_dummies(row, columns=["manufacturer"], drop_first=True, dtype=int)

    for col in feature_cols:
        if col not in row.columns:
            row[col] = 0

    X = row[feature_cols]
    proba = model.predict_proba(X)[:, 1][0]
    actual = drive["failure"].values[0]

    if proba >= COST_THRESHOLD:
        risk = "HIGH — schedule replacement"
    elif proba >= 0.01:
        risk = "MEDIUM — monitor closely"
    else:
        risk = "LOW — healthy"

    print(f"Drive          : {serial}")
    print(f"Failure prob   : {proba:.4f} ({proba*100:.2f}%)")
    print(f"Risk level     : {risk}")
    print(f"Threshold used : {COST_THRESHOLD} (cost-optimised: FN=$10K, FP=$50)")
    if actual == 1:
        print(f"Actual outcome : FAILED")
    else:
        print(f"Actual outcome : healthy")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/predict.py <serial_number>")
        print("\nDemo mode — scoring 5 random drives:\n")
        df = pd.read_parquet(DATA / "drives.parquet")

        failed = df[df["failure"] == 1].sample(2, random_state=42)
        healthy = df[df["failure"] == 0].sample(3, random_state=42)
        demos = pd.concat([failed, healthy])

        for serial in demos["serial_number"]:
            predict_drive(serial)
            print()
    else:
        predict_drive(sys.argv[1])
