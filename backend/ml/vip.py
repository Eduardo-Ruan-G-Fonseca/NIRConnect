from __future__ import annotations
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from typing import Sequence

__all__ = ["compute_vip_pls", "compute_vip_ovr_mean"]

def compute_vip_pls(pls: PLSRegression, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    VIP padrão para y unidimensional.
    Fórmula: VIP_j = sqrt( p * sum_a( SSY_a * w_{ja}^2 / sum_j w_{ja}^2 ) / sum_a SSY_a )
    onde SSY_a = sum_i t_{ia}^2 * q_a^2
    """
    T = pls.x_scores_            # (n, A)
    W = pls.x_weights_           # (p, A)
    Q = pls.y_loadings_          # (A, 1) para y 1D
    p = W.shape[0]
    # variância de Y explicada por componente a (SSY_a)
    SSY = (T ** 2).sum(axis=0) * (Q.ravel() ** 2)  # (A,)
    denom = (W ** 2).sum(axis=0)                   # (A,)
    tmp = (W ** 2) * SSY / denom                   # (p, A)
    vip = np.sqrt(p * tmp.sum(axis=1) / SSY.sum())
    return vip

def compute_vip_ovr_mean(models: Sequence[PLSRegression], X: np.ndarray, Ybin: np.ndarray) -> np.ndarray:
    """
    VIP médio para One-vs-Rest: computa VIP por classe e faz média.
    Ybin é (n, K) com 0/1 para cada classe.
    """
    vips = []
    for k, pls in enumerate(models):
        v = compute_vip_pls(pls, X, Ybin[:, k])
        vips.append(v)
    return np.mean(np.vstack(vips), axis=0)
