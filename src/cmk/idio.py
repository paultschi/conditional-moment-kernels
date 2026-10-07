"""The characteristic-driven idiosyncratic specification of Online Appendix D.1 ("Idiosyncratic specification").

phi^id(z) = psi(z) = (1, a, b) with a = rank-transformed size and b = rank-transformed idiosyncratic volatility,
U^id a PSD 3 x 3 matrix and Sigma^id_t = Diag(psi_t U^id psi_t'). With the systematic features phi = Phi A the loss of
Lemma A.3 (sufficient statistics, general case) reads
    L_t = w_t [ ||X||^4 - 2 s'U s - 2 tr(U^id S^id_t) + tr(U M U M)
                + 2 sum_{kl} U^id_{kl} <V, A' Gc_{kl,t} A> + vec(U^id)' Q_t vec(U^id) ],
with the statistics Gc_<mono>, Sid4, Q4 of stats_id.npz. The joint program in (U^sy, U^id) is solved by the accelerated
projected gradient method of cmk.solver (preconditioned U^sy, whitened U^id). Out of sample, the first-moment R^2 and the
second-moment R^{2,2} of the non-isotropic model are closed-form (r22_terms).
"""
from __future__ import annotations

import numpy as np

from .model import Params, _sym, month_features
from .solver import _psd_part, project_psd_fixed00

MONO_OF_PAIR = {(0, 0): "1", (0, 1): "a", (0, 2): "b", (0, 3): "c", (1, 1): "a2", (2, 2): "b2",
                (3, 3): "c2", (1, 2): "ab", (1, 3): "ac", (2, 3): "bc"}
NAMES = ["1", "mvel1", "idiovol", "retvol"]


class IdStats:
    """stats_id.npz; idx selects the coordinates of psi = (1, a, b, c), e.g. (0, 1, 2) = (1, size, idiosyncratic volatility)."""

    def __init__(self, path: str, idx=(0, 1, 2)):
        z = np.load(path, allow_pickle=False)
        self.idx = tuple(idx)
        self.p = len(idx)
        self.names = [NAMES[i] for i in idx]
        self.T = len(z["t"])
        self.d = int(z["d"])
        d = self.d
        self.iu = np.triu_indices(d)
        self.Gc = {}
        for a_, k in enumerate(idx):
            for b_, l in enumerate(idx):
                if a_ <= b_:
                    key = MONO_OF_PAIR[(min(k, l), max(k, l))]
                    self.Gc[(a_, b_)] = None if key == "1" else z["Gc_" + key].astype(np.float64)
        sel = np.ix_(np.arange(self.T), idx, idx)
        self.Sid = z["Sid4"][sel]
        kr = np.array([[i * 4 + j for j in idx] for i in idx]).ravel()
        self.Q = z["Q4"][np.ix_(np.arange(self.T), kr, kr)]

    def cross_tensor(self, stats, A: np.ndarray, t: int):
        """C[k, l] = A' Gc_{kl,t} A (m x m)."""
        d = self.d
        m = A.shape[1]
        p = self.p
        out = np.zeros((p, p, m, m))
        for (k, l), tri in self.Gc.items():
            if tri is None:
                Ct = A.T @ stats.G[t] @ A
            else:
                Gk = np.zeros((d, d))
                Gk[self.iu] = tri[t]
                Gk = Gk + Gk.T - np.diag(np.diag(Gk))
                Ct = A.T @ Gk @ A
            Ct = _sym(Ct)
            out[k, l] = Ct
            out[l, k] = Ct
        return out


class ParamsId(Params):
    def __init__(self, U, Uid, status="optimal"):
        super().__init__(U, float(Uid[0, 0]), status)
        self.Uid = 0.5 * (Uid + Uid.T)

    def rank_id(self, rtol=1e-8):
        ev = np.linalg.eigvalsh(self.Uid)
        return int((ev > max(rtol * ev.max(), 1e-14)).sum())


class FistaQPId:
    """The joint program in (U^sy, U^id) on a training window (Frobenius weight with month weight 1/N_t)."""

    def __init__(self, stats, ids: IdStats, A: np.ndarray, idx, tol: float = 1e-7, max_iter: int = 50000):
        m = A.shape[1]
        self.m = m
        n = m + 1
        p = ids.p
        self.p = p
        idx = list(idx)
        T = len(idx)
        self.T = T
        M = np.zeros((T, n, n))
        s = np.zeros((T, n))
        w = np.zeros(T)
        x2 = np.zeros(T)
        Cr = np.zeros((p, p, m, m))
        Sid = np.zeros((p, p))
        Q = np.zeros((p * p, p * p))
        for k, t in enumerate(idx):
            Mt, st, N, _, x2[k] = month_features(stats, A, t)
            M[k, 0, 0] = 1.0
            M[k, 1:, 1:] = Mt
            s[k, 0] = 1.0
            s[k, 1:] = st
            w[k] = stats.w[t]
            Cr += w[k] * ids.cross_tensor(stats, A, t)
            Sid += w[k] * ids.Sid[t]
            Q += w[k] * ids.Q[t]
        Cr /= T
        Sid /= T
        Q /= T
        # whitening of the idiosyncratic features: psi~ = L psi, L = Spsi^{-1/2}, U^id = L' U~ L
        Spsi = 0.5 * (Q[:, 0].reshape(p, p) + Q[:, 0].reshape(p, p).T)
        es, Vs = np.linalg.eigh(Spsi)
        es = np.maximum(es, 1e-10 * es.max())
        self.Lw = (Vs / np.sqrt(es)) @ Vs.T
        LL = np.kron(self.Lw, self.Lw)
        self.LwLw = self.Lw @ self.Lw.T
        Q = LL @ Q @ LL.T
        Sid = self.Lw @ Sid @ self.Lw.T
        Cr = np.einsum("ka,lb,abij->klij", self.Lw, self.Lw, Cr)
        Mbar = np.einsum("t,tij->ij", w, M) / T
        ev, V = np.linalg.eigh(Mbar)
        ev = np.maximum(ev, 1e-14 * ev.max())
        self.R = (V * np.sqrt(ev)) @ V.T
        self.Rinv = (V / np.sqrt(ev)) @ V.T
        self.Rinv2 = _sym(self.Rinv @ self.Rinv)
        Ri = self.Rinv
        self.Mt = np.einsum("ij,tjk,kl->til", Ri, M, Ri)
        self.wMt = self.Mt * w[:, None, None] / T
        st_ = s @ Ri.T
        self.S = np.einsum("t,ti,tj->ij", w, st_, st_) / T
        Cemb = np.zeros((p, p, n, n))
        Cemb[:, :, 1:, 1:] = Cr
        self.Ct = np.einsum("ij,kljm,mn->klin", Ri, Cemb, Ri)
        self.Sid = Sid
        self.Q = 0.5 * (Q + Q.T)
        self.const = float((w * (1.0 + x2) ** 2).sum() / T)
        self.c00 = float(Mbar[0, 0])
        self.tol, self.max_iter = tol, max_iter
        self.L = self._lipschitz()
        self.Ut = None
        self.Uid = None
        self.nu = 0.0

    def _grad_quad(self, Ut, Uid):
        MUM = ((self.wMt @ Ut) @ self.Mt).sum(axis=0)
        gU = 2.0 * (_sym(MUM) - self.S + np.einsum("kl,klij->ij", Uid, self.Ct))
        gI = 2.0 * (np.einsum("ij,klij->kl", Ut, self.Ct) - self.Sid + (self.Q @ Uid.ravel()).reshape(self.p, self.p))
        return gU, _sym(gI)

    def _obj(self, Ut, Uid, lam_sy, lam_id):
        MUM = ((self.wMt @ Ut) @ self.Mt).sum(axis=0)
        cross = np.einsum("ij,kl,klij->", Ut, Uid, self.Ct)
        quad = np.sum(Ut * MUM) - 2.0 * np.sum(Ut * self.S) - 2.0 * np.sum(Uid * self.Sid) \
            + 2.0 * cross + Uid.ravel() @ self.Q @ Uid.ravel() + self.const
        return quad + lam_sy * np.sum(Ut * self.Rinv2) + lam_id * np.sum(Uid * self.LwLw)

    def _lipschitz(self, iters=60):
        n = self.m + 1
        p = self.p
        rng = np.random.default_rng(0)
        V = _sym(rng.normal(size=(n, n)))
        W = _sym(rng.normal(size=(p, p)))
        g0U, g0I = self._grad_quad(np.zeros((n, n)), np.zeros((p, p)))
        lam = 1.0
        for _ in range(iters):
            nrm = np.sqrt(np.sum(V * V) + np.sum(W * W))
            V /= nrm
            W /= nrm
            HU, HI = self._grad_quad(V, W)
            HU = HU - g0U
            HI = HI - g0I
            lam = float(np.sum(V * HU) + np.sum(W * HI))
            V, W = HU, HI
        return 1.05 * lam

    def solve(self, lam_sy: float, lam_id: float = 0.0, warm: bool = True, U0=None, Uid0=None) -> ParamsId:
        n = self.m + 1
        p = self.p
        Lwi = np.linalg.inv(self.Lw)
        if U0 is not None:
            Ut = self.R @ U0 @ self.R
            Uid = Lwi.T @ Uid0 @ Lwi
        elif not warm or self.Ut is None:
            Ut = np.zeros((n, n))
            Ut[0, 0] = self.c00
            Uid = np.zeros((p, p))
        else:
            Ut, Uid = self.Ut.copy(), self.Uid.copy()
        pen = lam_sy * self.Rinv2
        penI = lam_id * self.LwLw
        step = 1.0 / self.L
        Y, Yi = Ut.copy(), Uid.copy()
        Uprev, Iprev = Ut.copy(), Uid.copy()
        tk = 1.0
        nu = self.nu
        converged = False
        for it in range(self.max_iter):
            gU, gI = self._grad_quad(Y, Yi)
            gU += pen
            gI += penI
            Unew, nu, _ = project_psd_fixed00(Y - step * gU, self.c00, nu0=nu)
            Inew, _ = _psd_part(_sym(Yi - step * gI))
            dU, dI = Unew - Uprev, Inew - Iprev
            if np.sum((Y - Unew) * dU) + np.sum((Yi - Inew) * dI) > 0:
                tk = 1.0
                Y, Yi = Unew.copy(), Inew.copy()
            else:
                tnew = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * tk * tk))
                beta = (tk - 1.0) / tnew
                tk = tnew
                Y = Unew + beta * dU
                Yi = Inew + beta * dI
            res = np.sqrt(np.sum(dU * dU) + np.sum(dI * dI)) / step
            scale = max(1.0, np.sqrt(np.sum(gU * gU) + np.sum(gI * gI)))
            Uprev, Iprev = Unew, Inew
            if res <= self.tol * scale:
                converged = True
                break
        self.Ut, self.Uid, self.nu = Uprev, Iprev, nu
        U = _sym(self.Rinv @ Uprev @ self.Rinv)
        U[0, 0] = 1.0
        P = ParamsId(U, _sym(self.Lw.T @ Iprev @ self.Lw), "optimal" if converged else "max_iter")
        P.iters = it + 1
        return P

    def loss(self, P) -> float:
        Lwi = np.linalg.inv(self.Lw)
        return float(self._obj(self.R @ P.U @ self.R, Lwi.T @ P.Uid @ Lwi, 0.0, 0.0))


def r22_terms(P: ParamsId, A: np.ndarray, stats, ids: IdStats, t: int, sigma2_bm: float) -> dict:
    """Closed-form R^2 and R^{2,2} summands of the non-isotropic model in month index t, and the mean and standard deviation
    of the implied idiosyncratic variances d_i = psi_i U^id psi_i'."""
    Mt, st, N, sumx, x2 = month_features(stats, A, t)
    b, V, Uid = P.b, P.V, P.Uid
    w = 1.0 / N
    res2 = x2 - 2.0 * b @ st + b @ Mt @ b
    MV = Mt @ V
    Cr = ids.cross_tensor(stats, A, t)
    cross = float(np.einsum("kl,klij,ij->", Uid, Cr, V))
    dx2 = float(np.sum(Uid * ids.Sid[t]))
    d2 = float(Uid.ravel() @ ids.Q[t] @ Uid.ravel())
    dsum = float(np.sum(Uid * ids.Q[t][:, 0].reshape(ids.p, ids.p)))
    r22_num = w * (x2 ** 2 - 2.0 * st @ V @ st - 2.0 * dx2 + np.trace(MV @ MV) + 2.0 * cross + d2)
    r22_den = w * (x2 ** 2 - 2.0 * sigma2_bm * x2 + sigma2_bm ** 2 * N)
    return dict(N=N, r2_num=w * res2, r2_den=w * x2, r22_num=r22_num, r22_den=r22_den,
                d_mean=dsum / N, d_sd=float(np.sqrt(max(d2 / N - (dsum / N) ** 2, 0.0))))
