import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Any
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils.multiclass import type_of_target
from .metrics import regression_metrics, classification_metrics, vip_scores
from .validation import build_cv
from .logger import log_info
import warnings

@dataclass
class PLSModelWrapper:
    model: PLSRegression
    classification: bool
    classes_: Optional[np.ndarray] = None
    lb_: Optional[dict] = None
    n_components: int = 2
    meta: Dict[str, Any] = None

    def predict(self, X: np.ndarray):
        if self.classification:
            Yhat = self.model.predict(X)
            if Yhat.ndim == 2 and Yhat.shape[1] > 1:
                idx = np.argmax(Yhat, axis=1)
            else:
                idx = (Yhat.ravel() > 0.5).astype(int)
            return np.array([self.classes_[i] for i in idx])
        else:
            return self.model.predict(X).ravel()

def is_categorical(y: np.ndarray) -> bool:
    t = type_of_target(y)
    return t in ("binary", "multiclass")

def train_pls(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int = 5,
    classification: Optional[bool] = None,
    validation_method: Optional[str] = None,
    validation_params: Optional[Dict[str, Any]] = None,
):
    if isinstance(y, pd.Series):
        y = y.values
        log_info(f"Convertendo y de Series para array: {y.shape}")
    elif not isinstance(y, np.ndarray) or y.ndim != 1:
        raise ValueError("y precisa ser uma série unidimensional")

    if classification is None:
        classification = is_categorical(y)

    if classification:
        y = y.astype(str)
    else:
        y = y.astype(float)

    if classification:
        y_series = np.array(list(map(str, y)))
        classes = sorted(np.unique(y_series))
        class_to_idx = {c: i for i, c in enumerate(classes)}
        idx = np.array([class_to_idx[val] for val in y_series])
        Ybin = np.eye(len(classes))[idx] if len(classes) > 2 else idx.reshape(-1, 1)
        pls = PLSRegression(n_components=n_components)
        pls.fit(X, Ybin)
        wrapper = PLSModelWrapper(pls, True, np.array(classes), None, n_components, {})
        Yhat = pls.predict(X)
        if Yhat.ndim > 1 and Yhat.shape[1] > 1:
            idx_pred = np.argmax(Yhat, axis=1)
        else:
            idx_pred = (Yhat.ravel() > 0.5).astype(int)
        y_pred = np.array([classes[i] for i in idx_pred])
        metrics = classification_metrics(y_series, y_pred, labels=classes)
    else:
        pls = PLSRegression(n_components=n_components)
        pls.fit(X, y)
        wrapper = PLSModelWrapper(pls, False, None, None, n_components, {})
        y_pred = wrapper.predict(X)
        metrics = regression_metrics(y, y_pred)

    vip = vip_scores(pls, X, y if y.ndim > 1 else y.reshape(-1, 1)).tolist()
    scores = pls.x_scores_.tolist()
    extra: Dict[str, Any] = {"vip": vip, "scores": scores}

    if validation_method == "none":
        return wrapper, metrics, extra

    if validation_method is None:
        validation_method = "StratifiedKFold" if classification else "KFold"
        validation_params = {
            "n_splits": 5,
            "shuffle": True,
            "random_state": 42,
        }
    validation_params = validation_params or {}

    try:
        cv = build_cv(validation_method, y, classification, validation_params)
        preds = np.empty(len(y), dtype=object if classification else float)
        for tr, te in cv:
            if classification:
                y_tr = np.array(list(map(str, y[tr])))
                classes_tr = sorted(np.unique(y_tr))
                class_to_idx_tr = {c: i for i, c in enumerate(classes_tr)}
                idx_tr = np.array([class_to_idx_tr[val] for val in y_tr])
                Ybin_tr = np.eye(len(classes_tr))[idx_tr] if len(classes_tr) > 2 else idx_tr.reshape(-1, 1)
                m = PLSRegression(n_components=n_components)
                m.fit(X[tr], Ybin_tr)
                pr = m.predict(X[te])
                if pr.ndim > 1 and pr.shape[1] > 1:
                    idx_pred = np.argmax(pr, axis=1)
                else:
                    idx_pred = (pr.ravel() > 0.5).astype(int)
                # map predictions back to original order of classes_tr
                preds[te] = np.array([classes_tr[i] for i in idx_pred])
            else:
                m = PLSRegression(n_components=n_components)
                m.fit(X[tr], y[tr])
                preds[te] = m.predict(X[te]).ravel()
        if classification:
            cv_metrics = classification_metrics(
                np.array(list(map(str, y))), preds.astype(str), labels=sorted(np.unique(np.array(list(map(str, y)))))
            )
        else:
            cv_metrics = regression_metrics(y, preds.astype(float))
        extra["cv_metrics"] = cv_metrics
    except Exception as exc:  # pragma: no cover - sanity
        warnings.warn(str(exc))
        extra["cv_metrics"] = None
    extra["validation"] = {"method": validation_method, "params": validation_params}

    return wrapper, metrics, extra
