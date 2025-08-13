import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple


def to_float_matrix(X_like: Any) -> np.ndarray:
    """Converte X em float, coerçando texto->NaN e ±inf->NaN."""
    df = pd.DataFrame(X_like)
    df = df.apply(pd.to_numeric, errors="coerce")
    X = df.to_numpy(dtype=float)
    X[~np.isfinite(X)] = np.nan
    return X


def encode_labels_if_needed(y_like: List[Any]) -> Tuple[np.ndarray, Dict[int, str], int]:
    """Se y for texto/categórico, codifica (0..K-1) e retorna mapping e K.
       Se já for numérico, retorna como float e mapping vazio."""
    s = pd.Series(y_like)
    if pd.api.types.is_numeric_dtype(s):
        y = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
        return y, {}, len(pd.unique(s.dropna()))
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_num = le.fit_transform(s.astype(str))
    mapping = {int(i): str(cls) for i, cls in enumerate(le.classes_)}
    return y_num.astype(float), mapping, len(le.classes_)
