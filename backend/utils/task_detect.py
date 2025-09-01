from __future__ import annotations
import numpy as np
import pandas as pd

__all__ = ["detect_task_from_y", "is_numeric_like"]

_NUMERIC_KINDS = set("biufc")  # bool/int/unsigned/float/complex

def is_numeric_like(y: np.ndarray) -> bool:
    if y is None:
        return False
    y = np.asarray(y)
    if y.dtype.kind in _NUMERIC_KINDS:
        return True
    # tenta converter elementos individualmente; se muitos falharem, não é numérico
    s = pd.Series(y.ravel(), dtype="object")
    converted = pd.to_numeric(s, errors="coerce")
    frac_numeric = converted.notna().mean()
    return frac_numeric > 0.9  # heurística conservadora

def detect_task_from_y(y: np.ndarray, req_mode: str | None) -> str:
    """
    Retorna 'classification' se y for não numérico ou for claramente discreto com poucas classes.
    Caso contrário, 'regression'. Respeita req_mode quando já for explicitamente classificação.
    """
    if req_mode and str(req_mode).lower() in {"classification", "pls-da", "classificação (pls-da)"}:
        return "classification"
    if req_mode and str(req_mode).lower() in {"regression", "plsr", "regressão (plsr)"}:
        # pode ser sobrescrito se y for não numérico
        pass

    if not is_numeric_like(y):
        return "classification"

    # Se numérico mas com poucas classes distintas, pode ser classificação codificada
    y_arr = np.asarray(y).ravel()
    n = y_arr.size
    if n > 0:
        uniq = pd.unique(y_arr[~pd.isna(y_arr)])
        if 2 <= uniq.size <= min(10, max(2, n // 10)):
            return "classification"
    return "regression"
