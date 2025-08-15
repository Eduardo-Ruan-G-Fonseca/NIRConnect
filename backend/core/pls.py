import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.utils.multiclass import type_of_target
from .metrics import regression_metrics, classification_metrics
from .logger import log_info
import warnings


@dataclass
class PLSDAOvR:
    """One-vs-Rest PLS-DA model using multiple ``PLSRegression`` estimators."""

    n_components: int
    pls_list: List[PLSRegression]
    classes_: np.ndarray
    label_encoder: LabelEncoder
    one_hot: OneHotEncoder

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = []
        for pls in self.pls_list:
            p = pls.predict(X).ravel()
            p = np.clip(p, 0.0, None)
            probs.append(p)
        P = np.vstack(probs).T
        row_sum = P.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return P / row_sum

    def predict(self, X: np.ndarray) -> np.ndarray:
        P = self.predict_proba(X)
        idx = np.argmax(P, axis=1)
        y_int = self.classes_[idx]
        return self.label_encoder.inverse_transform(y_int)


def fit_plsda_multiclass(X: np.ndarray, y: np.ndarray, n_components: int = 10, random_state: int = 42) -> PLSDAOvR:
    """Fit a multi-class PLS-DA model via One-vs-Rest strategy."""

    y = np.array(list(map(str, y)))
    le = LabelEncoder()
    y_int = le.fit_transform(y.ravel())
    classes = np.unique(y_int)

    ohe = OneHotEncoder(sparse_output=False, drop=None)
    Y = ohe.fit_transform(y_int.reshape(-1, 1))

    pls_list: List[PLSRegression] = []
    for c in range(Y.shape[1]):
        pls = PLSRegression(n_components=n_components, scale=False, copy=True)
        pls.random_state = random_state
        pls.fit(X, Y[:, c])
        pls_list.append(pls)

    return PLSDAOvR(
        n_components=n_components,
        pls_list=pls_list,
        classes_=classes,
        label_encoder=le,
        one_hot=ohe,
    )


def is_categorical(y: np.ndarray) -> bool:
    t = type_of_target(y)
    return t in ("binary", "multiclass")

def train_pls(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    n_components: int,
    classification: bool,
    validation_method: str = "none",
    validation_params: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    Xtr = np.asarray(Xtr); Xte = np.asarray(Xte)
    ytr = np.asarray(ytr); yte = np.asarray(yte)

    max_nc = int(min(Xtr.shape[1], max(1, Xtr.shape[0] - 1)))
    n_components = int(max(1, min(int(n_components), max_nc)))

    if classification:
        ytr = np.array(list(map(str, ytr)))
        yte = np.array(list(map(str, yte)))
        model = fit_plsda_multiclass(Xtr, ytr, n_components=n_components)
        preds = model.predict(Xte)
        metrics = classification_metrics(yte, preds, labels=model.label_encoder.classes_)
    else:
        model = PLSRegression(n_components=n_components)
        model.fit(Xtr, ytr)
        preds = model.predict(Xte).ravel()
        metrics = regression_metrics(yte, preds)

    return {"model": model, "metrics": metrics}
