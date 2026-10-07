"""The CMK estimator: the two-stage closed form of Section 4.5 (Proposition 4.2).

Stage 1 (statistics of the window, independent of the hyper-parameters):
    f_t = (G_t + rho_t I)^{-1} phi_t' x_{t+1}     regularized coordinates, rho_t = varrho tr(G_t) / m   (Section 4.2)
    b   = mean_t f_t                              the mean coefficient (Fama-MacBeth average)
    u^  = sum_t x'(I - phi_t (G_t + rho_t)^{-1} phi_t') x / sum_t (N_t - m)     pooled residual variance
    Xi  = mean_t (G_t + rho_t I)^{-1} G_t (G_t + rho_t I)^{-1}                  G^+_{t,rho} averaged, so that F^ = u^ Xi
    Abar = mean_t f_t f_t' - u^ Xi                the noise-corrected second moment
Stage 2 (floor eta and penalty lambda = 2 theta):
    excess = P_+(Abar - b b' - eta F^ - theta I),   C = eta F^ + excess,   V = C + b b'
The number of factors is the rank of the excess. `former` gives the moment step of IPCA in the full feature space
(F^ added in every direction, Table 3), `shape="identity"` the identity-shaped floor eta (tr F^ / m) I (Section 4.3).
"""
from __future__ import annotations

import numpy as np

from .model import Params, month_features
from .weights import gram_features

ETAS = [0.0, 0.5, 1.0, 2.0]                      # floor multipliers eta of Section 7.1
THETAS = [0.0, 1e-4, 3e-4, 1e-3, 3e-3]          # thresholds theta = lambda / 2 (lambda in {0, 2e-4, 6e-4, 2e-3, 6e-3})


def stage1(st, A, idx, ridge: float = 1e-3):
    """(b, u^, Abar, Xi) of the months idx under the window basis A, with Tikhonov parameter varrho = ridge."""
    m = A.shape[1]
    prs = [gram_features(st, A, t, ridge=ridge, equal_months=True) for t in idx]
    F = np.array([f1["s"] / f1["M"][0, 0] for f1, _ in prs])
    wr = np.array([f2["Nu"] for _, f2 in prs])                    # residual degrees of freedom N_t - m
    ur = np.array([f2["x2"] / f2["Nu"] for _, f2 in prs])
    u = float((wr * ur).sum() / wr.sum())
    b = F.mean(0)
    S = F.T @ F / len(F)
    Xi = np.zeros((m, m))
    for t in idx:
        Mt = month_features(st, A, t)[0]
        Ri = np.linalg.inv(Mt + ridge * np.trace(Mt) / m * np.eye(m))
        Xi += Ri @ Mt @ Ri
    Xi /= len(idx)
    return b, u, S - u * Xi, 0.5 * (Xi + Xi.T)


def stage1_pinv(st, A, idx, tol: float = 1e-9, rel: str = "max"):
    """Stage 1 without the Tikhonov regularization (Online Appendix D.1, "Tikhonov regularization"): coordinates by the
    pseudoinverse with the eigenvalues of G_t below tol times the largest (rel="max") or the mean (rel="mean") eigenvalue
    set to zero, the idiosyncratic covariance u^ G_t^+ and N_t - rank(phi_t) residual degrees of freedom."""
    m = A.shape[1]
    F, comp, dof, Xi = [], 0.0, 0.0, np.zeros((m, m))
    for t in idx:
        Mt, s_t, N, _, x2 = month_features(st, A, t)
        ev, Q = np.linalg.eigh(Mt)
        keep = ev > tol * (max(ev.max(), 1e-300) if rel == "max" else max(ev.mean(), 1e-300))
        Mp = (Q[:, keep] / ev[keep]) @ Q[:, keep].T
        f = Mp @ s_t
        F.append(f)
        Xi += Mp
        comp += float(x2 - s_t @ f)
        dof += float(N - int(keep.sum()))
    F = np.array(F)
    Xi /= len(idx)
    Xi = 0.5 * (Xi + Xi.T)
    u = comp / dof
    b = F.mean(0)
    S = F.T @ F / len(F)
    return b, u, S - u * Xi, Xi


def floor_matrix(u, Xi, eta, shape="gram"):
    """eta F^ (shape "gram") or the identity-shaped floor eta (tr F^ / m) I of the same trace."""
    m = Xi.shape[0]
    return eta * u * Xi if shape == "gram" else eta * u * np.trace(Xi) / m * np.eye(m)


def stage2(b, u, Abar, Xi, eta, theta, former: bool = False, shape: str = "gram") -> Params:
    """The CMK estimator for floor multiplier eta and threshold theta = lambda / 2; P.nfac_ex is the number of factors."""
    m = len(b)
    if former:
        C = Abar + u * Xi - np.outer(b, b)
        C = 0.5 * (C + C.T)
        ev, Q = np.linalg.eigh(C)
        C = (Q * np.maximum(ev, 0.0)) @ Q.T
        nfac = int((ev > 1e-12).sum())
    else:
        shift = floor_matrix(u, Xi, eta, shape)
        R = Abar - np.outer(b, b) - shift - theta * np.eye(m)
        R = 0.5 * (R + R.T)
        ev, Q = np.linalg.eigh(R)
        W = (Q * np.maximum(ev, 0.0)) @ Q.T
        nfac = int((ev > 1e-12).sum())
        C = shift + W
    U = np.zeros((m + 1, m + 1))
    U[0, 0] = 1.0
    U[1:, 0] = U[0, 1:] = b
    U[1:, 1:] = C + np.outer(b, b)
    P = Params(U, u, "closed-form")
    P.iters = 0
    P.nfac_ex = nfac
    return P
