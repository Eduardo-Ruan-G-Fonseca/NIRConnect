import numpy as np
import warnings
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    LeaveOneOut,
    train_test_split,
)


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
