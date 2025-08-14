import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.utils.multiclass import type_of_target
from sklearn.impute import SimpleImputer
from .metrics import regression_metrics, classification_metrics, vip_scores
from .validation import build_cv
from .logger import log_info
import warnings


def _sanitize_X_and_cap_components(X: np.ndarray, n_components: int) -> tuple[np.ndarray, int]:
    X = np.asarray(X)
    n_max = max(1, min(X.shape[1], X.shape[0] - 1))
    if X.dtype == float and np.isfinite(X).all():
        if n_components > n_max:
            log_info(
                f"n_components ajustado de {n_components} para {n_max} devido ao limite do rank."
            )
            return X, n_max
        return X, n_components

    X = X.astype(float, copy=False)
    X[~np.isfinite(X)] = np.nan
    col_ok = ~np.isnan(X).all(axis=0)
    if not col_ok.any():
        raise ValueError(
            "Matriz X inválida após pré-processamento (todas as variáveis são NaN)."
        )
    if not col_ok.all():
        X = X[:, col_ok]
    X = SimpleImputer(strategy="median").fit_transform(X)
    n_max = max(1, min(X.shape[1], X.shape[0] - 1))
    if n_components > n_max:
        log_info(
            f"n_components ajustado de {n_components} para {n_max} devido ao limite do rank."
        )
        n_components = n_max
    return X, n_components


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


@dataclass
class PLSModelWrapper:
    model: Any
    classification: bool
    classes_: Optional[np.ndarray] = None
    lb_: Optional[dict] = None
    n_components: int = 2
    meta: Dict[str, Any] = None

    def predict(self, X: np.ndarray):
        if self.classification:
            # ``PLSDAOvR`` already returns labels
            return self.model.predict(X)
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
        y = np.array(list(map(str, y)))
    else:
        y = y.astype(float)

    # === saneia X e limita n_components ===
    X, n_components = _sanitize_X_and_cap_components(X, n_components)

    if classification:
        pls_model = fit_plsda_multiclass(X, y, n_components=n_components)
        wrapper = PLSModelWrapper(
            pls_model, True, pls_model.label_encoder.classes_, None, n_components, {}
        )
        y_pred = pls_model.predict(X)
        metrics = classification_metrics(y, y_pred, labels=pls_model.label_encoder.classes_)
        # Compute VIP as mean of class models
        Y = pls_model.one_hot.transform(pls_model.label_encoder.transform(y).reshape(-1, 1))
        vips = [
            vip_scores(pls, X, Y[:, c].reshape(-1, 1))
            for c, pls in enumerate(pls_model.pls_list)
        ]
        vip_per_class = [v.tolist() for v in vips]
        vip = np.mean(vips, axis=0).tolist()
        scores = pls_model.pls_list[0].x_scores_.tolist()
    else:
        pls = PLSRegression(n_components=n_components)
        pls.fit(X, y)
        wrapper = PLSModelWrapper(pls, False, None, None, n_components, {})
        y_pred = wrapper.predict(X)
        metrics = regression_metrics(y, y_pred)
        vip = vip_scores(pls, X, y if y.ndim > 1 else y.reshape(-1, 1)).tolist()
        scores = pls.x_scores_.tolist()
    extra: Dict[str, Any] = {"vip": vip, "scores": scores}
    if classification:
        extra["vip_per_class"] = vip_per_class

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
                m = fit_plsda_multiclass(X[tr], y[tr], n_components=n_components)
                pr = m.predict(X[te])
                preds[te] = pr
            else:
                m = PLSRegression(n_components=n_components)
                m.fit(X[tr], y[tr])
                preds[te] = m.predict(X[te]).ravel()
        if classification:
            cv_metrics = classification_metrics(
                y, preds.astype(str), labels=sorted(np.unique(y))
            )
        else:
            cv_metrics = regression_metrics(y, preds.astype(float))
        extra["cv_metrics"] = cv_metrics
    except Exception as exc:  # pragma: no cover - sanity
        warnings.warn(str(exc))
        extra["cv_metrics"] = None
    extra["validation"] = {"method": validation_method, "params": validation_params}

    return wrapper, metrics, extra
