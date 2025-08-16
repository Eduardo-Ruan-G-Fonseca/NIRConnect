from sklearn.model_selection import LeaveOneOut, StratifiedKFold, KFold
import numpy as np


def build_cv(y, method: str, params: dict):
    method = (method or "").upper()
    y = np.array(y)

    if method == "LOO":
        cv = LeaveOneOut()
        val_name = "LOO"
        n_splits = len(y)
    elif method == "STRATIFIEDKFOLD":
        n_splits = int(params.get("n_splits", 5))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        val_name = "StratifiedKFold"
    else:
        n_splits = int(params.get("n_splits", 5))
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        val_name = "KFold"

    return cv, val_name, n_splits

