import warnings
from collections.abc import Iterator
import numpy as np
from sklearn.model_selection import (
    LeaveOneOut, KFold, ShuffleSplit,
    StratifiedKFold, StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import Callable, Dict, Any, List, Optional

def make_cv(
    method: str,
    params: dict | None,
    n_samples: int,
    task: str = "regression",
    y: np.ndarray | None = None,
):
    """
    Retorna o esquema de validação adequado:
    - Classificação: usa versões estratificadas (KFold/Holdout)
    - LOO em dataset grande: troca por KFold(5)
    - Ajusta n_splits para não ultrapassar o mínimo de amostras por classe
    """
    params = params or {}

    if method == "LeaveOneOut":
        if n_samples >= 80:
            warnings.warn("LOO trocado automaticamente por KFold(5) para acelerar.")
            return (
                StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                if task == "classification"
                else KFold(n_splits=5, shuffle=True, random_state=42)
            )
        return LeaveOneOut()

    if method == "KFold":
        n = int(params.get("n_splits", 5))
        if task == "classification" and y is not None:
            _, counts = np.unique(y, return_counts=True)
            max_splits = int(counts.min())
            n = max(2, min(n, max_splits))
            return StratifiedKFold(n_splits=n, shuffle=True, random_state=42)
        return KFold(n_splits=max(2, n), shuffle=True, random_state=42)

    if method == "Holdout":
        test_size = float(params.get("test_size", 0.3))
        if task == "classification" and y is not None:
            return StratifiedShuffleSplit(
                n_splits=1, test_size=test_size, random_state=42
            )
        return ShuffleSplit(n_splits=1, test_size=test_size, random_state=42)

    # fallback
    return (
        StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        if task == "classification"
        else KFold(n_splits=5, shuffle=True, random_state=42)
    )


def safe_n_components(n_req: int, n_samples: int, n_features: int) -> int:
    """Garante 1 ≤ n_components ≤ min(n_samples-1, n_features-1)."""
    hard_max = max(1, min(n_samples - 1, n_features - 1))
    return max(1, min(int(n_req or 1), hard_max))



def build_cv(
    method: str,
    y: np.ndarray,
    classification: bool,
    params: dict | None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield train/test indices for the requested CV method.

    The helper caps ``n_splits`` by the smallest class frequency when
    ``classification`` is True and falls back to a simple ``KFold`` when the
    split would be degenerate (e.g. single-class folds).
    """

    method = (method or "").upper()
    params = params or {}

    if method in ("LOO", "LEAVE-ONE-OUT"):
        loo = LeaveOneOut()
        for tr, te in loo.split(np.zeros((len(y), 1)), y if classification else None):
            yield tr, te
        return

    strat_names = {
        "SKF",
        "STRATIFIEDK",
        "STRATIFIEDK_FOLD",
        "STRATIFIEDKFOLD",
        "STRATIFIEDK-FOLD",
        "STRATIFIED",
    }

    if classification and (method in strat_names or method == "KFold" or not method):
        folds = int(params.get("n_splits", 5))
        _, counts = np.unique(y, return_counts=True)
        max_splits = int(counts.min())
        if max_splits < 2:
            cv = KFold(n_splits=min(3, len(y)))
            for tr, te in cv.split(np.zeros((len(y), 1))):
                yield tr, te
            return
        n_splits = int(min(max(2, folds), max_splits))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for tr, te in cv.split(np.zeros((len(y), 1)), y):
            yield tr, te
        return

    folds = int(params.get("n_splits", 5))
    n = int(len(y))
    n_splits = int(min(max(2, folds), max(2, n)))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr, te in cv.split(np.zeros((len(y), 1))):
        yield tr, te


def evaluate_plsda_multiclass(model_factory: Callable[[], Any], X: np.ndarray, y: np.ndarray, method: str = "LOO", params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Perform cross-validation for multi-class PLS-DA models.

    Parameters
    ----------
    model_factory: callable
        Function returning an object with ``fit`` and ``predict`` methods.
    X: ndarray
        Feature matrix.
    y: ndarray
        Target labels (strings or numbers).
    method: str
        Validation method: ``LOO``, ``KFold`` or ``Holdout``.
    params: dict, optional
        Additional parameters for the validation splitter.
    """

    y = np.asarray(y)
    preds: List[str] = []
    trues: List[str] = []

    if method == "LOO":
        cv = LeaveOneOut()
        for tr, te in cv.split(X):
            m = model_factory()
            m.fit(X[tr], y[tr])
            p = m.predict(X[te])
            preds.extend(p.tolist())
            trues.extend(y[te].tolist())
    elif method == "KFold":
        n_splits = int((params or {}).get("n_splits", 5))
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        for tr, te in cv.split(X):
            m = model_factory()
            m.fit(X[tr], y[tr])
            p = m.predict(X[te])
            preds.extend(p.tolist())
            trues.extend(y[te].tolist())
    elif method == "Holdout":
        test_size = float((params or {}).get("test_size", 0.3))
        tr, te = train_test_split(np.arange(len(X)), test_size=test_size, random_state=42, stratify=y)
        m = model_factory()
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        preds = p.tolist()
        trues = y[te].tolist()
    else:
        raise ValueError("validation_method inválido")

    acc = accuracy_score(trues, preds)
    prec = precision_score(trues, preds, average="macro", zero_division=0)
    rec = recall_score(trues, preds, average="macro", zero_division=0)
    f1 = f1_score(trues, preds, average="macro", zero_division=0)
    cm = confusion_matrix(trues, preds).tolist()

    return {
        "Accuracy": float(acc),
        "MacroPrecision": float(prec),
        "MacroRecall": float(rec),
        "MacroF1": float(f1),
        "ConfusionMatrix": cm,
    }
