from __future__ import annotations
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut


_LOO_ALIASES = {"LOO", "LEAVE-ONE-OUT", "LEAVE ONE OUT"}


def _safe_int(value: int | None, default: int = 5) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _cap_splits(requested: int, min_per_class: int | None) -> int:
    base = max(2, _safe_int(requested, 5))
    if min_per_class is None or min_per_class <= 0:
        return base
    return max(2, min(base, max(2, int(min_per_class))))


def build_cv(
    validation_method: str,
    y=None,
    n_splits: int = 5,
    stratified: bool = True,
    random_state: int = 42,
):
    method = (validation_method or "").upper().replace("-", " ").strip()

    if method in _LOO_ALIASES:
        if y is None:
            return LeaveOneOut()
        y_arr = np.asarray(y)
        if y_arr.size == 0:
            return LeaveOneOut()
        _, counts = np.unique(y_arr, return_counts=True)
        min_per_class = int(np.min(counts)) if counts.size else 0
        if min_per_class <= 1:
            safe = _cap_splits(n_splits, min_per_class)
            return StratifiedKFold(n_splits=safe, shuffle=True, random_state=random_state)
        return LeaveOneOut()

    if y is None or not stratified:
        safe = max(2, _safe_int(n_splits, 5))
        if y is not None:
            y_arr = np.asarray(y)
            safe = max(2, min(safe, int(y_arr.shape[0])))
        return KFold(n_splits=safe, shuffle=True, random_state=random_state)

    y_arr = np.asarray(y)
    if y_arr.size == 0:
        return KFold(n_splits=max(2, _safe_int(n_splits, 5)), shuffle=True, random_state=random_state)
    _, counts = np.unique(y_arr, return_counts=True)
    min_per_class = int(np.min(counts)) if counts.size else 0
    safe = _cap_splits(n_splits, min_per_class)
    return StratifiedKFold(n_splits=safe, shuffle=True, random_state=random_state)

