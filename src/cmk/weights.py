"""Per-month statistics of the weighted losses of Sections 3.1, 4.1 and Online Appendix D.1, in the form consumed by
`cmk.solver.FistaQP(feats=...)`.

A feature dict generalizes the statistics of Lemma 3.3 (isotropic case):
    M (m x m), s (m), x2, D (m x m), x2u, Nu    -> the loss  2 kappa (x2 - 2 b's + b'Mb) + x2^2 - 2 s'Vs + tr(VMVM)
                                                              - 2u x2u + 2u tr(V D) + u^2 Nu
(plus kappa_scale, the per-month weight of the first moment, and w, the month weight). The Browne weight couples the
first row of the augmented matrix with the stock block and uses the augmented keys Maug, saug, Daug.
"""
from __future__ import annotations

import numpy as np

from .model import Params, _sym, month_features


def gram_features(stats, A: np.ndarray, t: int, ridge: float = 1e-8, gamma: float = 1.0,
                  u_in_feature_block: bool = False, equal_months: bool = False):
    """The Gram-whitened loss (Lemma 4.1) with Tikhonov parameter `ridge` (eq. of Section 4.2), as two pseudo-months:
        feature block      c * (2 kappa ||f - b||^2 + ||f f' - V - u K||^2),  f = (G_t + rho_t)^{-1} phi_t' x,
                           K = (G_t + rho_t)^{-1} G_t (G_t + rho_t)^{-1} = G^+_{t,rho}
        residual block     gamma (x'(I - P)x  terms),  P the (regularized) projection on the feature span, N_t - m dof
    With equal_months the scale c = N_t^{1/2} cancels the month weight 1/N_t, so that every month has the same weight
    (the loss of the paper). Without u_in_feature_block the term u K is dropped from the feature block: the two-stage
    closed form (cmk.cmk) then fixes u at the pooled residual variance; with it, the solver computes the one-step
    estimator of Section 4.5."""
    Mt, st, N, sumx, x2 = month_features(stats, A, t)
    m = A.shape[1]
    Mr = Mt + ridge * np.trace(Mt) / m * np.eye(m)
    Mi = _sym(np.linalg.inv(Mr))
    c = 1.0 / np.sqrt(stats.w[t]) if equal_months else N / m
    f = Mi @ st
    comp = float(x2 - st @ f)                                     # x'(I - P)x
    if u_in_feature_block:
        K = _sym(Mi @ Mt @ Mi)
        f1 = dict(M=c * np.eye(m), s=c * f, x2=c * float(f @ f), D=c ** 2 * K, x2u=c ** 2 * float(f @ K @ f),
                  Nu=c ** 2 * float(np.trace(K @ K)), N=N)
    else:
        f1 = dict(M=c * np.eye(m), s=c * f, x2=c * float(f @ f), D=np.zeros((m, m)), x2u=0.0, Nu=0.0, N=N)
    if equal_months:
        f1["kappa_scale"] = c
    f2 = dict(M=np.zeros((m, m)), s=np.zeros(m), x2=gamma * comp, D=np.zeros((m, m)), x2u=gamma ** 2 * comp,
              Nu=gamma ** 2 * (N - m), N=N)
    return f1, f2


def plain_features(stats, A: np.ndarray, t: int) -> dict:
    """The Frobenius weight Omega_t = I (augmented by the first-moment weight kappa in the solver)."""
    Mt, st, N, sumx, x2 = month_features(stats, A, t)
    return dict(M=Mt, s=st, x2=x2, D=Mt, x2u=x2, Nu=N, N=N)


def month_loss(P: Params, f: dict, kappa: float = 1.0) -> float:
    """Loss of the model P on the statistics f, without the month weight."""
    b, V, u = P.b, P.V, P.u
    first = f["x2"] - 2.0 * b @ f["s"] + b @ f["M"] @ b
    MV = f["M"] @ V
    second = f["x2"] ** 2 - 2.0 * f["s"] @ V @ f["s"] + np.trace(MV @ MV) - 2.0 * u * f["x2u"] \
        + 2.0 * u * np.trace(V @ f["D"]) + u ** 2 * f["Nu"]
    return float(2.0 * kappa * f.get("kappa_scale", 1.0) * first + second)


def browne_features(stats, A: np.ndarray, t: int, P1: Params, w_month: float) -> dict:
    """The Browne weight (Online Appendix D.1): Omega_t proportional to q_{U^F}(Z_t, Z_t')^{-1}, the inverse of the
    first-stage augmented second-moment matrix, with Q^{-1} = L' Diag(1, Sigma^{-1}) L, L = [[1, 0], [-mu, I]]. Everything
    is m-dimensional through W = u^F Sigma^{-1} = I - phi K phi' (Woodbury) and mu = phi b^F. The weight is stored as
    Diag(u^F, W) = u^F Diag(1, Sigma^{-1}), so that the loss has the scale of the Frobenius loss."""
    Mt, st, N, sumx, x2 = month_features(stats, A, t)
    m = A.shape[1]
    I = np.eye(m)
    u = float(P1.u)
    C = P1.C
    b = P1.b
    if u <= 0:
        raise ValueError("first-stage u^id = 0")
    K = _sym(C @ np.linalg.solve(u * I + Mt @ C, I))            # C (uI + MC)^{-1}
    GK = Mt @ K
    MW = _sym(Mt - GK @ Mt)                                      # phi' W phi
    sW = st - GK @ st                                            # phi' W x
    xWx = float(x2 - st @ K @ st)                                # x' W x
    GKG = GK @ Mt
    DW = _sym(Mt - 2.0 * GKG + GK @ GKG)                         # phi' W^2 phi
    dW = st - 2.0 * GK @ st + GK @ GK @ st                       # phi' W^2 x
    xW2x = float(x2 - 2.0 * st @ K @ st + st @ K @ Mt @ K @ st)  # x' W^2 x
    trW2 = float(N - 2.0 * np.trace(GK) + np.trace(GK @ GK))     # tr W^2
    MWb = MW @ b
    DWb = DW @ b
    muWmu = float(b @ MWb)
    muWe = float(b @ (sW - MWb))
    eWe = float(xWx - 2.0 * sW @ b + muWmu)
    eW2e = float(xW2x - 2.0 * dW @ b + b @ DWb)
    n = m + 1
    Maug = np.zeros((n, n))
    Maug[0, 0] = u + muWmu
    Maug[0, 1:] = -MWb
    Maug[1:, 0] = -MWb
    Maug[1:, 1:] = MW
    saug = np.zeros(n)
    saug[0] = u - muWe
    saug[1:] = sW - MWb
    Daug = np.zeros((n, n))
    Daug[0, 0] = float(b @ DWb)
    Daug[0, 1:] = -DWb
    Daug[1:, 0] = -DWb
    Daug[1:, 1:] = DW
    return dict(Maug=Maug, saug=saug, Daug=Daug, x2=u + eWe, x2u=eW2e, Nu=trW2, w=float(w_month),
                M=MW, s=sW - MWb, D=DW, N=N)


def month_loss_aug(P: Params, f: dict) -> float:
    """Browne-weighted loss of the model P on the augmented statistics f (without the month weight)."""
    U, u = P.U, P.u
    MU = f["Maug"] @ U
    return float(f["x2"] ** 2 - 2.0 * f["saug"] @ U @ f["saug"] + np.trace(MU @ MU) - 2.0 * u * f["x2u"]
                 + 2.0 * u * np.trace(U @ f["Daug"]) + u ** 2 * f["Nu"])
