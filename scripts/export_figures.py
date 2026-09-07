"""Export presentation-ready figures from trained models."""

import json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, confusion_matrix,
)

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
MODELS  = ROOT / "models"
DATA    = ROOT / "data" / "processed"

GREEN  = "#1a7f37"
RED    = "#cf222e"
GRAY   = "#57606a"
PURPLE = "#6639a6"
ORANGE = "#d4830f"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "font.size": 11,
})


def load_data():
    df = pd.read_parquet(DATA / "drives.parquet")
    df = df.drop(columns=["serial_number", "model"])
    df = pd.get_dummies(df, columns=["manufacturer"], drop_first=True, dtype=int)

    with open(DATA / "feature_selection.json") as f:
        artifacts = json.load(f)

    feature_cols = artifacts["feature_cols"]
    train_idx = artifacts["train_idx"]
    test_idx = artifacts["test_idx"]

    X_train = df.loc[train_idx, feature_cols]
    X_test = df.loc[test_idx, feature_cols]
    y_train = df.loc[train_idx, "failure"]
    y_test = df.loc[test_idx, "failure"]
    return X_train, X_test, y_train, y_test, feature_cols


def fig_model_comparison(y_test, models_preds):
    """Horizontal bar chart comparing F1 across all models."""
    names = list(models_preds.keys())
    f1s = [models_preds[n]["f1"] for n in names]

    order = np.argsort(f1s)
    names = [names[i] for i in order]
    f1s = [f1s[i] for i in order]

    best_idx = len(f1s) - 1
    colors = [GREEN if i == best_idx else GRAY for i in range(len(f1s))]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(range(len(names)), f1s, color=colors, height=0.6, edgecolor="white")

    for i, (bar, val) in enumerate(zip(bars, f1s)):
        ax.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=10,
                fontweight="bold" if i == best_idx else "normal")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("F1 Score (failure class)")
    ax.set_title("Model comparison — F1 score", fontweight="bold", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(FIGURES / "model_comparison.png")
    plt.close(fig)
    print("  model_comparison.png")


def fig_confusion_matrix(y_test, y_pred, name, filename):
    """Confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Reds",
                xticklabels=["healthy", "failure"],
                yticklabels=["healthy", "failure"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix — {name}", fontweight="bold", fontsize=12)
    fig.savefig(FIGURES / filename)
    plt.close(fig)
    print(f"  {filename}")


def fig_roc_pr(y_test, models_probas):
    """ROC and Precision-Recall curves, all models overlaid."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for name, (proba, color) in models_probas.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc_val = roc_auc_score(y_test, proba)
        axes[0].plot(fpr, tpr, color=color, lw=2,
                     label=f"{name} ({auc_val:.4f})")

    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve", fontweight="bold", fontsize=12)
    axes[0].legend(fontsize=8)

    for name, (proba, color) in models_probas.items():
        prec, rec, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        axes[1].plot(rec, prec, color=color, lw=2,
                     label=f"{name} (AP={ap:.4f})")

    axes[1].axhline(y=y_test.mean(), color="k", lw=0.8, ls="--", alpha=0.4,
                    label=f"random ({y_test.mean():.4f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve", fontweight="bold", fontsize=12)
    axes[1].legend(fontsize=8, loc="upper right")

    fig.savefig(FIGURES / "roc_pr_curves.png")
    plt.close(fig)
    print("  roc_pr_curves.png")


def fig_feature_importance(model, top_n=15):
    """XGBoost feature importance by gain."""
    scores = model.get_booster().get_score(importance_type="gain")
    imp = pd.Series(scores).sort_values(ascending=False).head(top_n).sort_values()

    colors = [RED if "slope" in f else GREEN for f in imp.index]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(imp)), imp.values, color=colors, height=0.6, edgecolor="white")
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(imp.index, fontsize=9)
    ax.set_xlabel("Importance (gain)")
    ax.set_title("Top features — XGBoost (gain)", fontweight="bold", fontsize=13)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=RED, label="slope features"),
        Patch(color=GREEN, label="statistical aggregations"),
    ], loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(FIGURES / "feature_importance_xgb.png")
    plt.close(fig)
    print("  feature_importance_xgb.png")


def fig_imbalance_strategies(y_test, y_train, X_train, X_test, feature_cols):
    """Compare class imbalance handling strategies."""
    strategies = {
        "class_weight": 0.9549,
        "SMOTE": 0.9506,
        "Tomek links": 0.9551,
        "SMOTETomek": 0.9484,
        "RandomUnderSampler": 0.6055,
    }

    order = sorted(strategies, key=strategies.get)
    f1s = [strategies[k] for k in order]
    best_val = max(f1s)
    colors = [GREEN if v == best_val else GRAY for v in f1s]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(range(len(order)), f1s, color=colors, height=0.55, edgecolor="white")

    for i, (bar, val) in enumerate(zip(bars, f1s)):
        ax.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=10,
                fontweight="bold" if val == best_val else "normal")

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=10)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("F1 Score (failure class)")
    ax.set_title("Class imbalance strategies — RF comparison", fontweight="bold", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(FIGURES / "imbalance_strategies.png")
    plt.close(fig)
    print("  imbalance_strategies.png")


def fig_class_distribution(y_train, y_test):
    """Class distribution in train and test."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, y, title in [(axes[0], y_train, "Train"), (axes[1], y_test, "Test")]:
        counts = y.value_counts().sort_index()
        labels = ["Healthy", "Failure"]
        colors_bar = [GREEN, RED]
        bars = ax.bar(labels, counts.values, color=colors_bar, width=0.5, edgecolor="white")

        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + counts.max() * 0.02,
                    f"{val:,}", ha="center", fontsize=10, fontweight="bold")

        pct_fail = counts.iloc[1] / counts.sum() * 100
        ax.set_title(f"{title} — {pct_fail:.3f}% failures", fontweight="bold", fontsize=12)
        ax.set_ylabel("Drives")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Class distribution", fontweight="bold", fontsize=14, y=1.02)
    fig.savefig(FIGURES / "class_distribution.png")
    plt.close(fig)
    print("  class_distribution.png")


def main():
    print("Loading data...")
    X_train, X_test, y_train, y_test, feature_cols = load_data()

    print("Loading models...")
    scaler = joblib.load(MODELS / "scaler.joblib")
    lr = joblib.load(MODELS / "baseline_lr.joblib")
    rf = joblib.load(MODELS / "rf_best.joblib")
    xgb = joblib.load(MODELS / "final_model.joblib")

    X_ts = scaler.transform(X_test)
    X_ts = np.nan_to_num(X_ts, nan=0.0, posinf=0.0, neginf=0.0)

    y_proba_lr = lr.predict_proba(X_ts)[:, 1]
    y_pred_lr = lr.predict(X_ts)

    y_proba_rf = rf.predict_proba(X_test)[:, 1]
    y_pred_rf = rf.predict(X_test)

    y_proba_xgb = xgb.predict_proba(X_test)[:, 1]
    y_pred_xgb = xgb.predict(X_test)

    print("\nGenerating figures...")

    models_preds = {
        "LR baseline": {"f1": f1_score(y_test, y_pred_lr)},
        "RF balanced": {"f1": f1_score(y_test, y_pred_rf)},
        "XGBoost tuned": {"f1": f1_score(y_test, y_pred_xgb)},
    }
    fig_model_comparison(y_test, models_preds)

    fig_confusion_matrix(y_test, y_pred_xgb, "XGBoost tuned", "confusion_matrix_xgb.png")
    fig_confusion_matrix(y_test, y_pred_rf, "RF balanced", "confusion_matrix_rf.png")

    models_probas = {
        "LR baseline": (y_proba_lr, GRAY),
        "RF balanced": (y_proba_rf, PURPLE),
        "XGBoost tuned": (y_proba_xgb, GREEN),
    }
    fig_roc_pr(y_test, models_probas)

    fig_feature_importance(xgb, top_n=15)
    fig_imbalance_strategies(y_test, y_train, X_train, X_test, feature_cols)
    fig_class_distribution(y_train, y_test)

    print(f"\nDone — {len(list(FIGURES.glob('*.png')))} figures in {FIGURES}/")


if __name__ == "__main__":
    main()
