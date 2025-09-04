import numpy as np
from typing import Optional
from sklearn.base import clone


def select_intervals_mask(X, y, wavelengths, task, cv, max_intervals=2, interval_width=20, k_grid=None, top_pairs=6, time_budget_s=30):
    """
    Seleciona 1–3 intervalos (iPLS/siPLS básico) que maximizam balanced_accuracy (class) ou minimizam RMSE (reg).
    Estratégia: avalia intervalos únicos; pega top-N e avalia combinações 2 a 2 (synergy).
    """
    import time
    t0 = time.time()
    nvars = X.shape[1]
    # constrói janelas
    starts = list(range(0, nvars - interval_width + 1, interval_width))
    intervals = [(s, s + interval_width) for s in starts]
    if not intervals:
        return np.ones(nvars, dtype=bool)
    # modelo base: PLS com k do meio da grade (ou 5)
    k_mid = int(np.median(k_grid)) if k_grid else 5
    from sklearn.cross_decomposition import PLSRegression

    def score_mask(mask):
        Xm = X[:, mask]
        if Xm.shape[1] < k_mid + 1:
            return -np.inf
        pls = PLSRegression(n_components=k_mid)
        # CV básica
        scores = []
        for tr, te in cv.split(Xm, y):
            pls.fit(Xm[tr], y[tr])
            yhat = pls.predict(Xm[te]).ravel() if task != "classification" else pls.predict(Xm[te]).ravel()
            if task == "classification":
                # usa balanced_accuracy aproximando via median split (fallback)
                ypred = (yhat >= np.median(y[tr])).astype(int) if len(np.unique(y)) == 2 else None
                if ypred is None:
                    return -np.inf
                from sklearn.metrics import balanced_accuracy_score
                scores.append(balanced_accuracy_score(y[te], ypred))
            else:
                rmse = float(np.sqrt(np.mean((y[te] - yhat) ** 2)))
                scores.append(-rmse)  # maximizar
        return float(np.mean(scores)) if scores else -np.inf

    # avalia intervalos únicos
    single = []
    for (a, b) in intervals:
        m = np.zeros(nvars, dtype=bool); m[a:b] = True
        s = score_mask(m)
        single.append(((a, b), s))
        if time.time() - t0 > time_budget_s:
            break
    single.sort(key=lambda x: x[1], reverse=True)
    best_mask = np.zeros(nvars, dtype=bool); best_mask[single[0][0][0]:single[0][0][1]] = True

    if max_intervals == 1 or len(single) < 2:
        return best_mask

    # synergy 2 a 2
    cand = single[:top_pairs]
    best = (best_mask, single[0][1])
    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            (a1, b1), _ = cand[i]
            (a2, b2), _ = cand[j]
            m = np.zeros(nvars, dtype=bool); m[a1:b1] = True; m[a2:b2] = True
            s = score_mask(m)
            if s > best[1]:
                best = (m, s)
            if time.time() - t0 > time_budget_s:
                return best[0]
    return best[0]
