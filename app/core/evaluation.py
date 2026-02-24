"""Model evaluation with standard classification metrics."""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from typing import Any

from app.utils.logging import get_logger

logger = get_logger(__name__)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Compute classification metrics and return a dictionary.
    y_prob is optional; used for ROC-AUC when provided.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception as e:
            logger.warning("Could not compute ROC-AUC: %s", e)
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["confusion_matrix"] = {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }
    else:
        metrics["confusion_matrix"] = {"matrix": cm.tolist()}

    metrics["classification_report"] = classification_report(
        y_true, y_pred, zero_division=0, output_dict=True
    )

    return metrics
