from sklearn.model_selection import LeaveOneOut, StratifiedKFold, KFold
import numpy as np
from sklearn.utils.multiclass import type_of_target


def is_classification(y: np.ndarray) -> bool:
    t = type_of_target(y)
    return t in ("binary", "multiclass")


def build_cv(method: str, params: dict, y: np.ndarray):
    method = (method or "KFold").upper()
    params = params or {}
    y = np.asarray(y)

    if method == "LOO":
        cv = LeaveOneOut()
        splits = len(y)
        return cv, {"method": "LOO", "splits": splits, "effective_splits": splits}
    elif method == "KFOld" or method == "KFOLD":
        n_splits = int(params.get("n_splits") or 5)
        if is_classification(y):
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        return cv, {"method": "KFold", "splits": n_splits, "effective_splits": n_splits}
    elif method == "STRATIFIEDKFOLD":
        n_splits = int(params.get("n_splits") or 5)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        return cv, {"method": "StratifiedKFold", "splits": n_splits, "effective_splits": n_splits}
    else:
        return build_cv("KFold", params, y)
