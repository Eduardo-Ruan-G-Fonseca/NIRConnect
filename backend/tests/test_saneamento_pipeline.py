import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.saneamento import saneamento_global
from ml.pipeline import build_pls_pipeline
from sklearn.model_selection import KFold, cross_val_score


def test_saneamento_global_handles_nan_inf_and_removes_invalids():
    X = np.array([
        [1, np.nan, np.inf],
        [np.inf, np.nan, 3],
        [2, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
    ])
    y = np.array([1, 2, 3, 4])
    features = ['f1', 'f2', 'f3']

    Xc, yc, fc = saneamento_global(X, y, features)
    assert Xc.shape == (3, 2)
    assert yc.tolist() == [1, 2, 3]
    assert fc == ['f1', 'f3']
    assert np.isnan(Xc).sum() == 0


def test_pipeline_fits_after_saneamento():
    X = np.array([
        [np.inf, 1, 0],
        [2, 1, 0],
        [3, 1, 0],
        [4, 1, 0],
    ])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    features = ['a', 'b', 'c']
    Xc, yc, _ = saneamento_global(X, y, features)

    pipe = build_pls_pipeline(n_components=1)
    cv = KFold(n_splits=2, shuffle=True, random_state=0)
    scores = cross_val_score(pipe, Xc, yc, cv=cv, scoring='r2')
    assert len(scores) == 2
    pipe.fit(Xc, yc)
    preds = pipe.predict(Xc)
    assert preds.shape[0] == Xc.shape[0]
