from __future__ import annotations
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut


def build_cv(validation_method: str, y=None, n_splits: int = 5, stratified: bool = True, random_state: int = 42):
    method = (validation_method or "").upper()
    if method in {"LOO", "LEAVE-ONE-OUT", "LEAVE ONE OUT"}:
        return LeaveOneOut()
    if y is None:
        return KFold(n_splits=max(2, int(n_splits or 5)), shuffle=True, random_state=random_state)
    y = np.asarray(y)
    if stratified:
        _, counts = np.unique(y, return_counts=True)
        safe = max(2, min(int(n_splits or 5), int(np.min(counts))))
        return StratifiedKFold(n_splits=safe, shuffle=True, random_state=random_state)
    return KFold(n_splits=max(2, min(int(n_splits or 5), y.shape[0])), shuffle=True, random_state=random_state)

