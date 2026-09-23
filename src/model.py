"""Model training and evaluation utilities."""

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series, name: str) -> dict:
    """Evaluate a fitted model and return a metrics dict."""
    y_pred = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )
    return {
        "model": name,
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else None,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred),
    }


def cross_val(model, X, y, cv: int = 5, scoring: str = "f1") -> float:
    """Stratified cross-validation score."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    return cross_val_score(model, X, y, cv=skf, scoring=scoring).mean()
