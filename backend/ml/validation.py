"""Validation utilities used by the optimisation and training endpoints."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, KFold


def build_cv_meta(method: str, params: dict, y, is_classification: bool):
    """Return a cross-validation splitter and metadata.

    Parameters
    ----------
    method:
        Requested validation strategy name.  Case-insensitive.  Supported values
        are ``"LOO"``, ``"KFold"`` and ``"StratifiedKFold"``.  Any unknown
        value falls back to ``KFold``.
    params:
        Additional parameters such as ``n_splits`` or ``folds``.
    y:
        Target vector.
    is_classification:
        Whether the task is classification (enables ``StratifiedKFold``).
    """

    y = np.asarray(y).ravel()
    method = (method or "LOO").upper()
    params = params or {}

    if method == "LOO":
        cv = LeaveOneOut()
        splits = len(y)
        used = "LOO"
    elif method == "STRATIFIEDKFOLD" and is_classification:
        n_splits = int(params.get("n_splits") or params.get("folds") or 5)
        n_splits = max(2, min(n_splits, len(np.unique(y))))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = n_splits
        used = "StratifiedKFold"
    else:
        n_splits = int(params.get("n_splits") or params.get("folds") or 5)
        n_splits = max(2, min(n_splits, len(y)))
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = n_splits
        used = "KFold"

    return cv, {"method": used, "splits": splits}


__all__ = ["build_cv_meta"]

