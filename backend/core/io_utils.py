import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple


def to_float_matrix(X_like: Any) -> np.ndarray:
    """Converte X para float, coerçando texto -> NaN e trocando ±inf -> NaN.
    Mantém shape (n amostras, n colunas).
    """
    df = pd.DataFrame(X_like)
    df = df.apply(pd.to_numeric, errors="coerce")
    X = df.to_numpy(dtype=float)
    X[~np.isfinite(X)] = np.nan
    return X


def encode_labels_if_needed(y_like: List[Any]) -> Tuple[np.ndarray, Dict[int, str]]:
    """Se y for categórico/texto, codifica para 0..K-1 e retorna o mapping.
    Se já for numérico, apenas converte para float e mapping = {}.
    """
    s = pd.Series(y_like)
    if pd.api.types.is_numeric_dtype(s):
        return s.to_numpy(dtype=float), {}
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_num = le.fit_transform(s.astype(str))
    mapping = {int(i): str(cls) for i, cls in enumerate(le.classes_)}
    return y_num.astype(float), mapping
