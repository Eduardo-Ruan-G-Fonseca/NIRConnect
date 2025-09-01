from __future__ import annotations
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

__all__ = ["sanitize_X", "sanitize_y", "limit_n_components", "align_X_y"]

def sanitize_X(X: np.ndarray) -> np.ndarray:
    """float, ±Inf->NaN, remove colunas 100% NaN, imputa média."""
    X = np.asarray(X, dtype=float)
    X[np.isinf(X)] = np.nan
    # remove colunas inteiras NaN
    keep = ~np.all(np.isnan(X), axis=0)
    if np.any(~keep):
        X = X[:, keep]
    if X.size == 0:
        return X
    # imputa média por coluna (ignora colunas sem observações)
    imp = SimpleImputer(strategy="mean")
    X = imp.fit_transform(X)
    return X

def _coerce_float_or_nan(arr: np.ndarray) -> np.ndarray:
    out = np.empty(arr.shape, dtype=float)
    for i, v in enumerate(arr.ravel()):
        try:
            out.ravel()[i] = float(v)
        except Exception:
            out.ravel()[i] = np.nan
    return out

def sanitize_y(y: np.ndarray, task: str):
    """
    task: 'regression' ou 'classification'
    - regressão: tenta float, imputa média
    - classificação: aceita strings, label-encode; imputa moda antes do encode quando necessário
    Retorna (y_float_or_int, y_classes) onde y_classes é o mapping opcional (ou None)
    """
    y = np.asarray(y, dtype=object).reshape(-1, 1)
    # troca ±Inf por NaN quando possível
    try:
        y_float = y.astype(float)
        y = np.where(np.isinf(y_float), np.nan, y_float)
    except Exception:
        y = np.where(y == float("inf"), np.nan, y)
        y = np.where(y == float("-inf"), np.nan, y)

    if task == "classification":
        # imputação por moda em objeto
        imp = SimpleImputer(strategy="most_frequent")
        y_imp = imp.fit_transform(y).ravel().astype(object)

        le = LabelEncoder()
        y_encoded = le.fit_transform(y_imp)  # 0..K-1
        classes = list(le.classes_)
        return y_encoded.astype(int), classes
    else:
        # regressão: força float onde possível; inválidos -> NaN; imputa média
        yf = _coerce_float_or_nan(y.ravel()).reshape(-1, 1)
        imp = SimpleImputer(strategy="mean")
        yf = imp.fit_transform(yf).ravel()
        return yf, None

def limit_n_components(n_components: int, X: np.ndarray) -> int:
    if X is None or X.size == 0:
        return max(1, int(n_components))
    n_samples, n_features = X.shape
    hard_max = max(1, min(n_features, n_samples - 1))
    return int(max(1, min(n_components, hard_max)))

def align_X_y(X: np.ndarray, y: np.ndarray):
    """
    Remove linhas onde y é NaN (ou comprimento inconsistente), retornando X_alinhado, y_alinhado e máscara.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if y.ndim > 1:
        y = y.ravel()

    # Se y veio vazio, retorna como está (caller decide o erro)
    if y.size == 0:
        return X, y, np.ones(X.shape[0], dtype=bool)

    # constrói máscara de linhas válidas (y não-NaN)
    if np.issubdtype(y.dtype, np.floating):
        mask = ~np.isnan(y)
    else:
        # tipos inteiros/categóricos já devem estar sem NaN aqui
        mask = np.ones_like(y, dtype=bool)

    # alinha pelo tamanho mínimo
    n = min(X.shape[0], y.shape[0])
    Xn = X[:n]
    yn = y[:n]
    mask = mask[:n]

    Xn = Xn[mask]
    yn = yn[mask]
    return Xn, yn, mask
