import numpy as np
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, KFold


def build_cv_meta(method: str, params: dict, y):
    """Build a cross-validation splitter and metadata.

    Parameters
    ----------
    method: str
        Requested validation strategy. Supported values are ``"LOO"``,
        ``"KFold"`` and ``"StratifiedKFold"``. Any unknown value falls
        back to ``KFold``.
    params: dict
        Additional parameters like ``n_splits``. May be ``None``.
    y: array-like
        Target vector used to infer stratification when possible.
    """

    y = np.asarray(y).ravel() if y is not None else None
    params = params or {}

    if method == "LOO":
        cv = LeaveOneOut()
        meta = {"validation": {"method": "LOO", "splits": len(y)}}
        return cv, meta

    n_splits = int(params.get("n_splits", 5))
    if y is not None and len(np.unique(y)) > 1:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        method_used = "StratifiedKFold"
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        method_used = "KFold"

    return cv, {"validation": {"method": method_used, "splits": n_splits}}


__all__ = ["build_cv_meta"]
