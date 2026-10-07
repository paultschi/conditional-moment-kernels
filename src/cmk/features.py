"""Feature maps of the two kernels of the empirical analysis (Section 7.1).

Linear kernel   k(z, z') = 1 + <z, z'>: the feature matrix of a month is [1, z_t], the instrument matrix of IPCA, with z the
                rank-transformed characteristics (the Welch-Goyal predictors, the last N_MACRO covariates, are dropped).
Gaussian kernel k_G(z, z') = exp(-||z - z'||^2 / (2 ell^2)): Nystrom features k_G(z, landmarks) B with B B' =
                k_G(landmarks, landmarks)^{-1} (Section 3.2); the landmarks are selected by a pivoted Cholesky decomposition
                of the kernel matrix of a pool of stock-months from the first months of the panel.
The additive kernel k + k_G has the features [1, z, k_G(z, landmarks) B].
"""
from __future__ import annotations

import numpy as np

from .data import N_MACRO


def characteristics(Z: np.ndarray) -> np.ndarray:
    """The characteristics of a covariate matrix: all columns but the last N_MACRO (the Welch-Goyal predictors)."""
    return Z[:, :Z.shape[1] - N_MACRO]


def linear_features(Z: np.ndarray) -> np.ndarray:
    """[1, z]: a constant and the characteristics."""
    Zc = characteristics(Z)
    return np.hstack([np.ones((len(Zc), 1)), Zc])


def gauss_kernel(Za: np.ndarray, Zb: np.ndarray, ell2: float) -> np.ndarray:
    """exp(-||a - b||^2 / (2 ell^2)) for all pairs of rows of Za and Zb."""
    na = np.sum(Za ** 2, axis=1)[:, None]
    nb = np.sum(Zb ** 2, axis=1)[None, :]
    d2 = np.maximum(na + nb - 2.0 * (Za @ Zb.T), 0.0)
    return np.exp(-d2 / (2.0 * ell2))


def pivoted_cholesky(pool: np.ndarray, ell2: float, L: int) -> np.ndarray:
    """Indices of L landmarks chosen by greedy pivoted Cholesky on the Gaussian kernel matrix of `pool` (the pivot is the
    largest residual diagonal; the diagonal of k_G is one). O(n L^2) without forming the n x n matrix."""
    n = len(pool)
    d = np.ones(n)
    piv, Lmat = [], np.zeros((n, L))
    for j in range(L):
        i = int(np.argmax(d))
        if d[i] <= 1e-12:
            break
        piv.append(i)
        col = gauss_kernel(pool, pool[i:i + 1], ell2)[:, 0].astype(float)
        if j > 0:
            col = col - Lmat[:, :j] @ Lmat[i, :j]
        Lmat[:, j] = col / np.sqrt(d[i])
        d = d - Lmat[:, j] ** 2
        d[i] = 0.0
    return np.array(piv)


def nystrom_basis(landmarks: np.ndarray, ell2: float) -> np.ndarray:
    """B with B B' = k_G(landmarks, landmarks)^{-1} (inverse transpose of the Cholesky factor)."""
    Kzz = gauss_kernel(landmarks, landmarks, ell2)
    Lch = np.linalg.cholesky(Kzz + 1e-10 * np.eye(len(landmarks)))
    return np.linalg.inv(Lch).T


def additive_features(Z: np.ndarray, landmarks: np.ndarray, B: np.ndarray, ell2: float) -> np.ndarray:
    """[1, z, k_G(z, landmarks) B]: the features of the additive kernel."""
    Zc = characteristics(Z)
    return np.hstack([np.ones((len(Zc), 1)), Zc, gauss_kernel(Zc, landmarks, ell2) @ B])


def normalize_rows(Z: np.ndarray) -> np.ndarray:
    """z / ||z||: the feature map of the cosine kernel (used by the simulation study)."""
    nrm = np.linalg.norm(Z, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return Z / nrm
