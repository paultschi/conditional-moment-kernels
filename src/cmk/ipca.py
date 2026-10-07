"""The IPCA benchmark of Kelly, Pruitt and Su (2019) on the sufficient statistics (Remark 4.3, Online Appendix D.1).

Model x_{t+1} = Z_t Gamma g_{t+1} + noise with instruments Z_t = [1, z_t] (the linear features), loadings Gamma (L x K)
and latent factors g_{t+1} (K); no intercept (the restricted model). IPCA minimizes the sum of squared residuals in
profile form: the factor step, the cross-sectional regressions g_{t+1}(Gamma) = (Gamma' W_t Gamma + rho I)^{-1}
Gamma' v_t with W_t = Z_t'Z_t and v_t = Z_t'x_{t+1}, is solved in closed form with a Tikhonov regularization of 1e-4 times
the average eigenvalue (needed in the 1970s, when some instruments are identically zero), and Gamma is found by L-BFGS
in whitened instrument coordinates, started from the estimate of the previous window and from the leading singular
vectors of the characteristic-managed portfolios (the KPS start), keeping the better local optimum.

The moment step maps the fitted model to the parameter of the paper: phi = Z Gamma, b = mean g, V = mean g g' (divisor T),
u = pooled residual variance, so that cmk.model.evaluate_month gives exactly the quantities of the moment kernel model.
"""
from __future__ import annotations

import numpy as np

from .model import Params, evaluate_month


def _orth(G):
    Q, _ = np.linalg.qr(G)
    return Q


def _factor_step(W, v, Gamma, ridge):
    """g_t = (Gamma' W_t Gamma + ridge tr(.)/K I)^{-1} Gamma' v_t for all t; returns F (T x K) and v Gamma."""
    K = Gamma.shape[1]
    WG = W @ Gamma
    Mg = Gamma.T @ WG
    Mg = 0.5 * (Mg + np.swapaxes(Mg, 1, 2))
    Mg += ridge * np.trace(Mg, axis1=1, axis2=2)[:, None, None] * np.eye(K)[None]
    sg = v @ Gamma
    return np.linalg.solve(Mg, sg[..., None])[..., 0], sg


def _whitening(W):
    """Gbar^{-1/2} and Gbar^{1/2} for Gbar = sum_t W_t / sum_t tr W_t."""
    Gbar = W.sum(0) / np.einsum("tii->", W)
    ew, Vw = np.linalg.eigh(Gbar)
    ew = np.maximum(ew, 1e-8 * ew.max())
    return (Vw / np.sqrt(ew)) @ Vw.T, (Vw * np.sqrt(ew)) @ Vw.T


def profile_fit(W, v, x2, K: int, Gamma0=None, max_iter: int = 200, ridge: float = 1e-4, cold_start: bool = True):
    """Minimize sum_t ||x_t - Z_t Gamma g_t(Gamma)||^2 over Gamma; returns Gamma (orthonormal), F, sse, iterations, success."""
    from scipy.optimize import minimize
    T, L = v.shape
    Rm, Rp = _whitening(W)
    Wt = Rm @ W @ Rm
    vt = v @ Rm

    def obj(gvec):
        G = gvec.reshape(L, K)
        F, sg = _factor_step(Wt, vt, G, ridge)
        sse = float(x2.sum() - np.einsum("tk,tk->", F, sg))
        grad = 2.0 * (((Wt @ G) @ np.einsum("tk,tl->tkl", F, F)).sum(0) - vt.T @ F)
        return sse, grad.ravel()

    starts = []
    if Gamma0 is not None and Gamma0.shape == (L, K):
        starts.append(_orth(Rp @ Gamma0))
    if cold_start or not starts:
        Gsvd = np.linalg.svd(v / np.sqrt(np.einsum("tii->t", W))[:, None], full_matrices=False)[2][:K].T
        starts.append(_orth(Rp @ Gsvd))
    best = None
    for G0 in starts:
        res = minimize(obj, G0.ravel(), jac=True, method="L-BFGS-B",
                       options=dict(maxiter=max_iter, ftol=1e-13, gtol=1e-10, maxcor=30))
        if best is None or res.fun < best.fun:
            best = res
    Gamma = _orth(Rm @ best.x.reshape(L, K))
    F, sg = _factor_step(W, v, Gamma, ridge)
    sse = float(x2.sum() - np.einsum("tk,tk->", F, sg))
    return Gamma, F, sse, int(best.nit), bool(best.success)


def fit_window(st, idx, K: int, Gamma0=None, **kw) -> Params:
    """IPCA on the months idx and its moment step; P.Gamma is the loading matrix (the window basis of evaluate_month)."""
    idx = list(idx)
    W = st.G[idx]
    v = st.s[idx]
    x2 = st.sumx2[idx]
    Gamma, F, sse, it, conv = profile_fit(W, v, x2, K, Gamma0, **kw)
    lam = F.mean(0)
    V = F.T @ F / len(idx)
    u = sse / st.N[idx].sum()
    U = np.zeros((K + 1, K + 1))
    U[0, 0] = 1.0
    U[0, 1:] = lam
    U[1:, 0] = lam
    U[1:, 1:] = V
    P = Params(U, u, "optimal" if conv else "max_iter")
    P.iters = it
    P.Gamma = Gamma
    P.F = F
    P.sse = sse
    P.tot_r2_train = 1.0 - sse / x2.sum()
    return P


def evaluate_ipca(P: Params, st, t: int, sigma2_bm: float, gmv: bool = False) -> dict:
    """Out-of-sample quantities of month index t, as for the moment kernel model, plus the return of the tangency
    portfolio of the K factors."""
    out = evaluate_month(P, P.Gamma, st, t, sigma2_bm, gmv=gmv)
    Mt = P.Gamma.T @ st.G[t] @ P.Gamma
    s_t = P.Gamma.T @ st.s[t]
    f = np.linalg.pinv(0.5 * (Mt + Mt.T), rcond=1e-10, hermitian=True) @ s_t
    wt = np.linalg.pinv(P.C, rcond=1e-10, hermitian=True) @ P.b
    out["ret_tan"] = float(wt @ f)
    out["tot_r2_train"] = P.tot_r2_train
    return out
