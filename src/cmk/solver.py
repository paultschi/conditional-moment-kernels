"""The solver of Appendix B: accelerated projected gradient with adaptive restart for the CMK program, isotropic
specification, in preconditioned coordinates.

Problem:  min_{U - S PSD, U_00 = 1, u >= 0}  (1/T) sum_t w_t L_t(U, u) + lambda^sy tr U + lambda^id u,
    L_t = ||X||^4 - 2 s_t'U s_t - 2u x2_t + tr(U M_t U M_t) + 2u tr(U D_t) + N_t u^2,
with S = Diag(0, eta F^) the floor (the spectral floor of Section 4.3; S = 0 without floor).

Preconditioning: with R = Mbar^{1/2}, Mbar = (1/T) sum_t w_t M_t (block-diagonal part), the variable U~ = R U R has a
Hessian (1/T) sum_t w_t M~_t (x) M~_t with M~_t = R^{-1} M_t R^{-1} averaging to the identity. The constraint U_00 = 1
becomes U~_00 = Mbar_00, the trace penalty <R^{-2}, U~>, and u is rescaled to the curvature of the U~ block.
Projection onto {U~ - S~ PSD, U~_00 = c}: S~ + P_+(Y - S~ - nu E00) with nu the root of the monotone scalar equation
[P_+(Y - S~ - nu E00)]_00 = c, found by a safeguarded secant (Illinois) iteration.
"""
from __future__ import annotations

import numpy as np

from .model import Params, _sym, month_features


def _psd_part(Y):
    ev, V = np.linalg.eigh(Y)
    pos = ev > 0
    if not pos.any():
        return np.zeros_like(Y), 0
    Vp = V[:, pos]
    return (Vp * ev[pos]) @ Vp.T, int(pos.sum())


def project_psd_fixed00(Y, c, nu0=0.0, tol=1e-12, maxit=100):
    """Projection of the symmetric Y onto {U PSD : U_00 = c}; returns (U, nu, rank)."""
    E = np.zeros_like(Y)
    E[0, 0] = 1.0

    def g(nu):
        P, r = _psd_part(Y - nu * E)
        return P[0, 0] - c, P, r

    lo, hi = nu0, nu0
    glo, Plo, rlo = g(lo)
    if abs(glo) <= tol * max(1.0, c):
        return Plo, lo, rlo
    step = max(1.0, abs(Y[0, 0]) + c)
    if glo > 0:                                       # g is nonincreasing in nu: increase nu
        ghi = glo
        hi = lo
        while ghi > 0:
            hi += step
            step *= 2
            ghi, Phi, rhi = g(hi)
    else:                                             # decrease nu
        ghi, Phi, rhi = glo, Plo, rlo
        hi = lo
        while glo < 0:
            lo -= step
            step *= 2
            glo, Plo, rlo = g(lo)
    a, b, fa, fb = lo, hi, glo, ghi
    P, r, nu = Phi, rhi, hi
    for _ in range(maxit):
        if fa - fb == 0:
            nu = 0.5 * (a + b)
        else:
            nu = b - fb * (b - a) / (fb - fa)
            if not (min(a, b) < nu < max(a, b)):
                nu = 0.5 * (a + b)
        fnu, P, r = g(nu)
        if abs(fnu) <= tol * max(1.0, c) or abs(b - a) < 1e-15 * max(1.0, abs(nu)):
            return P, nu, r
        if fnu > 0:
            a, fa = nu, fnu
            fb *= 0.5
        else:
            b, fb = nu, fnu
            fa *= 0.5
    return P, nu, r


class FistaQP:
    """The CMK program on a window. feats: optional list (one dict per pseudo-month of idx) of weighted statistics, see
    cmk.weights; without it the Frobenius weight. kappa: weight of the first-moment error (the augmented feature map with
    sqrt(kappa) e_0 at Delta). wts: optional month weights w_t replacing 1/N_t."""

    def __init__(self, stats, A: np.ndarray, idx, tol: float = 1e-7, max_iter: int = 20000, kappa: float = 1.0,
                 feats=None, wts=None):
        m = A.shape[1]
        self.m = m
        n = m + 1
        idx = list(idx)
        T = len(idx)
        self.T = T
        self.kappa = float(kappa)
        M = np.zeros((T, n, n))
        s = np.zeros((T, n))
        N = np.zeros(T)
        x2 = np.zeros(T)
        w = np.zeros(T)
        Dm = np.zeros((T, n, n))
        kap = np.zeros(T)
        x2u_all = np.zeros(T)
        for k, t in enumerate(idx):
            if feats is None:
                Mt, st, N[k], _, x2[k] = month_features(stats, A, t)
                Dk, x2u, Nu = Mt, x2[k], N[k]
            else:
                f = feats[k]
                Mt, st, x2[k] = f["M"], f["s"], f["x2"]
                Dk, x2u, Nu = f["D"], f["x2u"], f["Nu"]
                N[k] = Nu
            if feats is not None and "Maug" in feats[k]:       # Browne weight: fully augmented statistics
                f = feats[k]
                M[k] = f["Maug"]
                s[k] = f["saug"]
                Dm[k] = f["Daug"]
                kap[k] = 0.0
                N[k] = Nu
            else:
                kap_k = self.kappa * (feats[k].get("kappa_scale", 1.0) if feats is not None else 1.0)
                kap[k] = kap_k
                M[k, 0, 0] = kap_k
                M[k, 1:, 1:] = Mt
                s[k, 0] = kap_k
                s[k, 1:] = st
                Dm[k, 1:, 1:] = Dk
                N[k] = Nu
            w[k] = wts[k] if wts is not None else (feats[k]["w"] if (feats is not None and "w" in feats[k]) else stats.w[t])
            x2u_all[k] = x2u
        if feats is None:
            x2u_all = x2
        Mbar = np.einsum("t,tij->ij", w, M) / T
        if feats is not None and any("Maug" in f for f in feats):
            Mbar[0, 1:] = 0.0                                   # block-diagonal preconditioner for the Browne weight
            Mbar[1:, 0] = 0.0
        ev, V = np.linalg.eigh(Mbar)
        ev = np.maximum(ev, 1e-14 * ev.max())
        self.R = (V * np.sqrt(ev)) @ V.T
        self.Rinv = (V / np.sqrt(ev)) @ V.T
        self.Rinv2 = _sym(self.Rinv @ self.Rinv)
        Ri = self.Rinv
        self.Mt = np.einsum("ij,tjk,kl->til", Ri, M, Ri)
        self.wMt = self.Mt * w[:, None, None] / T
        st_ = s @ Ri.T
        Dt = np.einsum("ij,tjk,kl->til", Ri, Dm, Ri)
        self.S = np.einsum("t,ti,tj->ij", w, st_, st_) / T
        self.Dbar = np.einsum("t,tij->ij", w, Dt) / T
        self.wx2 = float((w * x2u_all).sum() / T)
        self.wN = float((w * N).sum() / T)
        self.const = float((w * (kap + x2) ** 2).sum() / T)
        self.c00 = float(Mbar[0, 0])
        curvU = float(np.mean(w * np.linalg.eigvalsh(self.Mt)[:, -1] ** 2))
        self.alpha = np.sqrt(max(curvU, 1e-12) / max(self.wN, 1e-12))   # u = alpha * u~
        self.tol, self.max_iter = tol, max_iter
        self.L = self._lipschitz()
        self.Ut = None
        self.ut = 0.0
        self.nu = 0.0

    def _grad_quad(self, Ut, ut):
        u = self.alpha * ut
        MUM = ((self.wMt @ Ut) @ self.Mt).sum(axis=0)
        gU = 2.0 * (_sym(MUM) - self.S + u * self.Dbar)
        gu = 2.0 * (np.sum(Ut * self.Dbar) - self.wx2 + self.wN * u) * self.alpha
        return gU, gu

    def _obj(self, Ut, ut, lam_sy, lam_id):
        u = self.alpha * ut
        MUM = ((self.wMt @ Ut) @ self.Mt).sum(axis=0)
        quad = np.sum(Ut * MUM) - 2.0 * np.sum(Ut * self.S) - 2.0 * u * self.wx2 \
            + 2.0 * u * np.sum(Ut * self.Dbar) + self.wN * u ** 2 + self.const
        return quad + lam_sy * np.sum(Ut * self.Rinv2) + lam_id * u

    def _lipschitz(self, iters=50):
        n = self.m + 1
        rng = np.random.default_rng(0)
        V = _sym(rng.normal(size=(n, n)))
        v = rng.normal()
        g0U, g0u = self._grad_quad(np.zeros((n, n)), 0.0)
        lam = 1.0
        for _ in range(iters):
            nrm = np.sqrt(np.sum(V * V) + v * v)
            V /= nrm
            v /= nrm
            HU, Hu = self._grad_quad(V, v)
            HU = HU - g0U
            Hu = Hu - g0u
            lam = float(np.sum(V * HU) + v * Hu)
            V, v = HU, Hu
        return 1.05 * lam

    def solve(self, lam_sy: float, lam_id: float = 0.0, warm: bool = True, shift=None) -> Params:
        """shift: the floor S ((m+1) x (m+1), zero first row and column); the feasible set is {U - S PSD, U_00 = 1}."""
        n = self.m + 1
        St = np.zeros((n, n)) if shift is None else _sym(self.R @ shift @ self.R)
        if not warm or self.Ut is None:
            Ut = St.copy()
            Ut[0, 0] = self.c00
            ut = 0.0
        else:
            Ut, ut = self.Ut.copy(), self.ut
        pen = lam_sy * self.Rinv2
        step = 1.0 / self.L
        Y, y = Ut.copy(), ut
        Uprev, uprev = Ut.copy(), ut
        tk = 1.0
        nu = self.nu
        converged = False
        for it in range(self.max_iter):
            gU, gu = self._grad_quad(Y, y)
            gU += pen
            gu += lam_id * self.alpha
            Unew, nu, _ = project_psd_fixed00(Y - step * gU - St, self.c00, nu0=nu)
            Unew = Unew + St
            unew = max(y - step * gu, 0.0)
            dU, du = Unew - Uprev, unew - uprev
            if np.sum((Y - Unew) * dU) + (y - unew) * du > 0:     # adaptive restart (O'Donoghue and Candes)
                tk = 1.0
                Y, y = Unew.copy(), unew
            else:
                tnew = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * tk * tk))
                beta = (tk - 1.0) / tnew
                tk = tnew
                Y = Unew + beta * dU
                y = unew + beta * du
            res = np.sqrt(np.sum(dU * dU) + du * du) / step
            scale = max(1.0, np.sqrt(np.sum(gU * gU) + gu * gu))
            Uprev, uprev = Unew, unew
            if res <= self.tol * scale:
                converged = True
                break
        self.Ut, self.ut, self.nu = Uprev, uprev, nu
        self.iters = it + 1
        U = _sym(self.Rinv @ Uprev @ self.Rinv)
        U[0, 0] = 1.0
        P = Params(U, self.alpha * uprev, "optimal" if converged else "max_iter")
        P.iters = self.iters
        return P

    def loss(self, P: Params) -> float:
        Ut = self.R @ P.U @ self.R
        return float(self._obj(Ut, P.u / self.alpha, 0.0, 0.0))
