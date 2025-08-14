import numpy as np
import warnings
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    LeaveOneOut,
    ShuffleSplit,
    train_test_split,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import Callable, Dict, Any, List, Optional


def make_cv(method: str, params: dict | None, n_samples: int):
    """Create cross-validation scheme with safeguards for large datasets."""
    params = params or {}
    if method == "LeaveOneOut":
        if n_samples >= 80:
            warnings.warn("LOO trocado automaticamente por KFold(5) para acelerar.")
            return KFold(n_splits=5, shuffle=True, random_state=42)
        return LeaveOneOut()

    if method == "KFold":
        n = int(params.get("n_splits", 5))
        return KFold(n_splits=max(2, n), shuffle=True, random_state=42)

    if method == "Holdout":
        test_size = float(params.get("test_size", 0.3))
        return ShuffleSplit(n_splits=1, test_size=test_size, random_state=42)

    # fallback
    return KFold(n_splits=5, shuffle=True, random_state=42)


def safe_n_components(n_req: int, n_samples: int, n_features: int) -> int:
    """Ensure component count within valid bounds."""
    hard_max = max(1, min(n_samples - 1, n_features - 1))
    return max(1, min(int(n_req or 1), hard_max))



def build_cv(method: str, y: np.ndarray, classification: bool, params: dict):
    """Return generator of train/test indices for the given validation method."""
    params = params or {}
    indices = np.arange(len(y))

    if method == "LOO":
        if len(y) > 500:
            warnings.warn("Leave-One-Out may be slow for large datasets")
        cv = LeaveOneOut()
        return cv.split(indices)

    if method == "KFold":
        return KFold(**params).split(indices)

    if method == "StratifiedKFold":
        if not classification:
            raise ValueError("StratifiedKFold only valid for classification")
        return StratifiedKFold(**params).split(indices, y)

    if method == "Holdout":
        test_size = params.get("test_size", 0.2)
        shuffle = params.get("shuffle", True)
        random_state = params.get("random_state", 42)
        if classification:
            classes, counts = np.unique(y, return_counts=True)
            if np.any(counts < 2):
                raise ValueError("Holdout requires at least 2 samples per class")
            train_idx, test_idx = train_test_split(
                indices,
                test_size=test_size,
                shuffle=shuffle,
                random_state=random_state,
                stratify=y,
            )
        else:
            train_idx, test_idx = train_test_split(
                indices,
                test_size=test_size,
                shuffle=shuffle,
                random_state=random_state,
            )

        def _gen():
            yield train_idx, test_idx

        return _gen()

    raise ValueError(f"Unknown validation method: {method}")


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
