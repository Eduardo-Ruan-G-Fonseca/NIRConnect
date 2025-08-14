import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, r2_score
from .transformers import ReplaceInfWithNaN, DropAllNaNColumns


def build_pls_pipeline(n_components: int = 10) -> Pipeline:
    """Build a leak-proof PLS Regression pipeline."""
    return Pipeline(steps=[
        ("inf_to_nan", ReplaceInfWithNaN()),
        ("drop_all_nan_cols", DropAllNaNColumns()),
        ("impute", SimpleImputer(strategy="median")),
        ("var_thresh", VarianceThreshold(0.0)),
        ("pls", PLSRegression(n_components=n_components)),
    ])


def eval_pls_regression(X: np.ndarray, y: np.ndarray, n_components: int, cv) -> dict:
    """Evaluate PLS regression under a CV scheme."""
    preds = np.zeros(len(y), dtype=float)
    for tr, te in cv.split(X, y):
        pls = PLSRegression(n_components=n_components)
        pls.fit(X[tr], y[tr])
        preds[te] = pls.predict(X[te]).ravel()
    rmse = float(np.sqrt(np.mean((y - preds) ** 2)))
    r2 = float(r2_score(y, preds))
    return {"RMSECV": rmse, "R2": r2}


def eval_plsda_binary(X: np.ndarray, y: np.ndarray, n_components: int, cv) -> dict:
    """Evaluate binary PLS-DA using logistic regression on latent space."""
    accs = []
    cm_sum = None
    for tr, te in cv.split(X, y):
        pls = PLSRegression(n_components=n_components)
        pls.fit(X[tr], y[tr])
        Ttr = pls.transform(X[tr])
        Tte = pls.transform(X[te])
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Ttr, y[tr])
        ypred = clf.predict(Tte)
        accs.append(accuracy_score(y[te], ypred))
        cm = confusion_matrix(y[te], ypred, labels=np.unique(y))
        cm_sum = cm if cm_sum is None else (cm_sum + cm)
    return {
        "Accuracy": float(np.mean(accs)),
        "ConfusionMatrix": cm_sum.tolist() if cm_sum is not None else None,
    }


def eval_plsda_multiclass(X, y, n_components, cv):
    """PLS-DA multiclasse via OneVsRest."""
    accs, f1s = [], []
    cm_sum = None

    for train_idx, test_idx in cv.split(X, y):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        pls = PLSRegression(n_components=n_components)
        pls.fit(Xtr, ytr)
        Ttr = pls.transform(Xtr)
        Tte = pls.transform(Xte)

        clf = LogisticRegression(max_iter=1000, multi_class="ovr")
        clf.fit(Ttr, ytr)
        ypred = clf.predict(Tte)

        accs.append(accuracy_score(yte, ypred))
        f1s.append(f1_score(yte, ypred, average="macro"))

        cm = confusion_matrix(yte, ypred, labels=np.unique(y))
        cm_sum = cm if cm_sum is None else (cm_sum + cm)

    return {
        "Accuracy": float(np.mean(accs)),
        "MacroF1": float(np.mean(f1s)),
        "ConfusionMatrix": cm_sum.tolist() if cm_sum is not None else None,
    }
