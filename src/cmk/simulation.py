"""The simulation study of Online Appendix C (Figure C.1, Table C.1).

DGP (20 covariates, 3 factors, N = 500 stocks per month, idiosyncratic variance u = 0.01):
    z_i ~ N(0, diag(1, 1/2, ..., 1/20)),  phi(z) = z / ||z||  (cosine kernel; the misspecified variant appends three
    nonlinear features c_k z~_k z~_{k+1} that the estimator does not see),
    x_{t+1} = phi_t b + phi_t beta g_{t+1} + sqrt(u) eps,  g ~ N(0, I_3),  b = beta (0.2, 0.15, 0.1)' + pricing error,
so that U^pop = [[1, b'], [b, beta beta' + b b']] and C^pop = beta beta' has rank three.

Experiments
    rate   E||U_T - U^pop||^2 against T for the Frobenius weight (solver) and the Gram-whitened weight (closed form,
           eta = 0, lambda = 0), 100 panels per T; log-log slopes; the plug-in bound sigma^2 / (alpha^2 T)
    path   number of factors and squared distance along the lambda path, T = 108 and T = 1000
    oos    the protocol of Section 7.1 on a 720-month panel (96 + 12 months, (eta, lambda) by validation, re-estimation on
           108 months), m = 5, 10, 20, against the population moments, well-specified and misspecified DGP
All random numbers come from numpy generators seeded by (seed, experiment, T, replication), so the results are
reproducible. About 15 minutes on one core with 100 replications.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .cmk import stage1, stage2
from .model import Params, benchmark_sigma2, evaluate_month, month_features
from .post.common import metrics
from .solver import FistaQP


class SimStats:
    """The sufficient statistics of a simulated panel (the interface of cmk.stats.Stats)."""

    def __init__(self, G, s, N, sumx, sumx2):
        self.G, self.s, self.N, self.sumx, self.sumx2 = G, s, N.astype(float), sumx, sumx2
        self.T, self.d = len(N), G.shape[1]
        self.w = 1.0 / self.N
        self.t = np.arange(self.T)
        self.date0 = np.array([f"{1962 + k // 12:04d}-{1 + k % 12:02d}" for k in range(self.T)])


class DGP:
    def __init__(self, d=20, r=3, N=500, u=0.01, nonlinear=False, seed=0, lam_scale=0.04, gamma=(0.2, 0.15, 0.1),
                 alpha_scale=0.001):
        rng = np.random.default_rng(seed)
        self.d, self.r, self.N, self.u, self.nonlinear = d, r, N, u, nonlinear
        self.sig = 1.0 / np.sqrt(np.arange(1, d + 1))
        self.q = 3 if nonlinear else 0
        self.p = d + self.q
        self.Lam = lam_scale * rng.normal(size=(self.p, r))
        self.gamma = np.asarray(gamma[:r])
        self.a = alpha_scale * rng.normal(size=self.p)
        self.b = self.Lam @ self.gamma + self.a
        self.C = self.Lam @ self.Lam.T
        U = np.zeros((self.p + 1, self.p + 1))
        U[0, 0] = 1.0
        U[1:, 0] = U[0, 1:] = self.b
        U[1:, 1:] = self.C + np.outer(self.b, self.b)
        self.U = U
        self.P = Params(U, u, "population")
        if nonlinear:          # scale of the nonlinear features: half the root mean square of the coordinate they use
            zn = self._zn(np.random.default_rng(12345), 200000)
            self.cpsi = np.array([0.5 * np.sqrt(np.mean(zn[:, k] ** 2) / np.mean((zn[:, k] * zn[:, k + 1]) ** 2))
                                  for k in range(self.q)])
            Phi = self.features(np.random.default_rng(12346), 200000)
            sysvar = np.mean((Phi @ self.Lam) ** 2, axis=0).sum()
            nl = np.mean((Phi[:, self.d:] @ self.Lam[self.d:]) ** 2, axis=0).sum()
            self.share_nonlinear = float(nl / sysvar)

    def _zn(self, rng, N):
        z = rng.normal(size=(N, self.d)) * self.sig
        return z / np.linalg.norm(z, axis=1, keepdims=True)

    def features(self, rng, N=None):
        zn = self._zn(rng, N or self.N)
        if not self.nonlinear:
            return zn
        psi = np.column_stack([self.cpsi[k] * zn[:, k] * zn[:, k + 1] for k in range(self.q)])
        return np.column_stack([zn, psi])

    def month(self, rng, N=None):
        Phi = self.features(rng, N)
        f = rng.normal(size=self.r)
        x = Phi @ (self.b + self.Lam @ f) + np.sqrt(self.u) * rng.normal(size=len(Phi))
        return Phi, x

    def panel(self, rng, T, N=None):
        G = np.zeros((T, self.p, self.p))
        s = np.zeros((T, self.p))
        Ns, sumx, sumx2 = np.zeros(T), np.zeros(T), np.zeros(T)
        for t in range(T):
            Phi, x = self.month(rng, N)
            G[t] = Phi.T @ Phi
            s[t] = Phi.T @ x
            Ns[t] = len(x)
            sumx[t] = x.sum()
            sumx2[t] = x @ x
        return SimStats(G, s, Ns, sumx, sumx2)

    def basis(self, A_lin):
        """The estimator's basis (linear block) padded to the dimension of the DGP features."""
        A = np.zeros((self.p, A_lin.shape[1]))
        A[:self.d] = A_lin
        return A


def lin_basis(stats, idx, d, m):
    """The m dominant eigenvectors of the pooled Gram matrix of the linear block."""
    Gsum = stats.G[idx][:, :d, :d].sum(axis=0)
    ev, V = np.linalg.eigh(Gsum)
    order = np.argsort(ev)[::-1]
    return V[:, order[:m]]


def err2(P, dgp):
    """||U^sy - U^pop||_F^2 and (u - u^pop)^2."""
    return float(np.sum((P.U - dgp.U) ** 2)), float((P.u - dgp.u) ** 2)


def theorem_constants(dgp, stats):
    """alpha = minimum over months of the smallest eigenvalue of the per-month Hessian A(xi) (on S and on the differences of
    the feasible set, i.e. without the direction E00), and sigma^2 = E ||grad R(U^pop, xi)||^2, Frobenius weight."""
    n = dgp.p + 1
    p = n * n
    B = []
    for i in range(n):
        for j in range(i, n):
            E = np.zeros((n, n))
            if i == j:
                E[i, i] = 1.0
            else:
                E[i, j] = E[j, i] = 1.0 / np.sqrt(2.0)
            B.append(E.ravel())
    B = np.array(B).T
    Bx = np.zeros((p + 1, B.shape[1] + 1))
    Bx[:p, :-1] = B
    Bx[p, -1] = 1.0
    e0 = np.zeros(n)
    e0[0] = 1.0
    amin = np.inf
    amin_D = np.inf
    z2 = []
    U, u = dgp.U, dgp.u
    for t in range(stats.T):
        Mt, st, N, _, x2 = month_features(stats, np.eye(dgp.p), t)
        M = np.zeros((n, n))
        M[0, 0] = 1.0
        M[1:, 1:] = Mt
        s = np.concatenate(([1.0], st))
        D = M - np.outer(e0, e0)
        H = np.zeros((p + 1, p + 1))
        H[:p, :p] = np.kron(M, M)
        H[:p, p] = H[p, :p] = D.ravel()
        H[p, p] = N
        Ax = 2.0 * Bx.T @ H @ Bx
        amin = min(amin, float(np.linalg.eigvalsh(Ax)[0]))
        amin_D = min(amin_D, float(np.linalg.eigvalsh(Ax[1:, 1:])[0]))
        gU = 2.0 * (M @ U @ M - np.outer(s, s) + u * D)
        gu = 2.0 * (np.sum(U * D) - x2 + N * u)
        z2.append(np.sum(gU * gU) + gu * gu)
    return amin, amin_D, float(np.mean(z2))


def run_rate(dgp, Ts, reps, seed, log):
    rows = []
    t0 = time.time()
    for T in Ts:
        for rep in range(reps):
            rng = np.random.default_rng([seed, T, rep])
            st = dgp.panel(rng, T)
            A = np.eye(dgp.p)
            idx = list(range(T))
            P_cf = stage2(*stage1(st, A, idx, 0.0), 0.0, 0.0)
            P_fr = FistaQP(st, A, idx, kappa=1.0, tol=1e-9).solve(0.0)
            e_cf, e_fr = err2(P_cf, dgp), err2(P_fr, dgp)
            rows.append(dict(T=T, rep=rep, est="frobenius", err_sy=e_fr[0], err_id=e_fr[1], err=e_fr[0] + e_fr[1],
                             rank_C=P_fr.rank_C(), iters=P_fr.iters, status=P_fr.status))
            rows.append(dict(T=T, rep=rep, est="gram", err_sy=e_cf[0], err_id=e_cf[1], err=e_cf[0] + e_cf[1],
                             rank_C=P_cf.nfac_ex, iters=0, status="closed-form"))
        log(f"rate T={T}: {reps} panels, {time.time() - t0:.0f}s")
    return pd.DataFrame(rows)


def run_path(dgp, Ts, lams, seed, log):
    rows = []
    for T in Ts:
        rng = np.random.default_rng([seed, 777, T])
        st = dgp.panel(rng, T)
        A = np.eye(dgp.p)
        idx = list(range(T))
        qp = FistaQP(st, A, idx, kappa=1.0, tol=1e-9)
        s1 = stage1(st, A, idx, 0.0)
        for lam in lams:
            P = qp.solve(lam)
            e = err2(P, dgp)
            rows.append(dict(T=T, est="frobenius", lam=lam, rank_C=P.rank_C(), err=e[0] + e[1], u=P.u, trC=float(np.trace(P.C))))
            for est, eta in [("gram", 0.0), ("gram_floor", 1.0)]:
                Pg = stage2(*s1, eta, lam / 2.0)
                e = err2(Pg, dgp)
                rows.append(dict(T=T, est=est, lam=lam, rank_C=Pg.nfac_ex, err=e[0] + e[1], u=Pg.u, trC=float(np.trace(Pg.C))))
        log(f"path T={T} done")
    return pd.DataFrame(rows)


ETAS = [0.0, 0.5, 1.0, 2.0]
THETAS = [0.0, 1e-4, 3e-4, 1e-3, 3e-3]


def run_oos(dgp, T, train, ms, seed, log, nval=12):
    rng = np.random.default_rng([seed, 999, int(dgp.nonlinear)])
    st = dgp.panel(rng, T)
    rows = []
    for t0 in range(0, T - train):
        idx = list(range(t0, t0 + train))
        t_te = t0 + train
        idx_tr, idx_val = idx[:-nval], idx[-nval:]
        s2 = benchmark_sigma2(st, idx)
        for m in ms:
            A_tr = dgp.basis(lin_basis(st, idx_tr, dgp.d, m))
            s1 = stage1(st, A_tr, idx_tr, 1e-3)
            best, bv = None, np.inf
            for g in ETAS:
                for th in THETAS:
                    P = stage2(*s1, g, th)
                    vs = float(np.mean([evaluate_month(P, A_tr, st, t, s2)["score"] for t in idx_val]))
                    if vs < bv:
                        best, bv = (g, th), vs
            A = dgp.basis(lin_basis(st, idx, dgp.d, m))
            P = stage2(*stage1(st, A, idx, 1e-3), *best)
            ev = evaluate_month(P, A, st, t_te, s2)
            ev.pop("gamma_sdf")
            ev.pop("gamma_cmve")
            ev["rank_C"] = P.nfac_ex
            rows.append(dict(t=t_te, model=f"m={m}", m=m, lam=2 * best[1], gamma=best[0], **ev))
        ev = evaluate_month(dgp.P, np.eye(dgp.p), st, t_te, s2)
        ev.pop("gamma_sdf")
        ev.pop("gamma_cmve")
        rows.append(dict(t=t_te, model="population", m=dgp.p, lam=0.0, gamma=0.0, **ev))
    log(f"oos, nonlinear = {dgp.nonlinear}: {T - train} test months")
    return pd.DataFrame(rows)


def slope(x, y):
    X = np.column_stack([np.ones(len(x)), np.log(x)])
    c = np.linalg.lstsq(X, np.log(y), rcond=None)[0]
    return float(c[1]), float(c[0])


def make_figure(rate, path, consts, fig):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8})
    f, ax = plt.subplots(1, 3, figsize=(10.5, 3.2))
    for est, lab, mk in [("frobenius", "Frobenius loss", "o"), ("gram", "Gram-whitened loss", "s")]:
        g = rate[rate.est == est].groupby("T").err.agg(["mean", lambda v: np.quantile(v, 0.95)])
        g.columns = ["mean", "q95"]
        sl, _ = slope(g.index.values, g["mean"].values)
        ax[0].loglog(g.index, g["mean"], marker=mk, ms=4, label=f"{lab}, MSE (slope {sl:.2f})")
        ax[0].loglog(g.index, g["q95"], marker=mk, ms=3, ls=":", color=ax[0].lines[-1].get_color(), label=f"{lab}, 95% quantile")
    alpha, sig2 = consts
    Ts = np.array(sorted(rate["T"].unique()))
    ax[0].loglog(Ts, sig2 / (alpha ** 2 * Ts), "k--", lw=0.8, label=r"bound $\sigma^2/(\alpha^2 T)$")
    ax[0].set_xlabel("$T$ (months)")
    ax[0].set_ylabel(r"$\|\mathbf{U}_T-\mathbf{U}^{\rm pop}\|^2$")
    ax[0].legend(loc="center right", fontsize=7, framealpha=0.9)
    ax[0].set_title("(a) mean squared error")
    labs = {"frobenius": "Frobenius", "gram": "Gram-whitened, $\\eta=0$", "gram_floor": "Gram-whitened, $\\eta=1$"}
    for T, ls in [(108, "-"), (1000, "--")]:
        for est, col in [("frobenius", "C0"), ("gram", "C1"), ("gram_floor", "C2")]:
            g = path[(path["T"] == T) & (path.est == est)].sort_values("lam")
            ax[1].semilogx(g.lam, g.rank_C, ls=ls, color=col, drawstyle="steps-post", label=f"{labs[est]}, $T={T}$")
            ax[2].loglog(g.lam, g.err, ls=ls, color=col, label=f"{labs[est]}, $T={T}$")
    ax[1].axhline(3, color="k", lw=0.6, ls=":")
    ax[1].set_xlabel(r"$\lambda^{\rm sy}$")
    ax[1].set_ylabel("number of factors")
    ax[1].set_title("(b) number of factors along the path")
    ax[1].legend(loc="lower left", fontsize=6)
    ax[2].set_xlabel(r"$\lambda^{\rm sy}$")
    ax[2].set_ylabel(r"$\|\mathbf{U}_T-\mathbf{U}^{\rm pop}\|^2$")
    ax[2].set_title("(c) error along the path")
    ax[2].legend(loc="upper left", fontsize=7)
    f.tight_layout()
    f.savefig(fig + ".pdf")
    plt.close(f)


def make_table(oos_lin, oos_nl, tab):
    lines = [r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
             r"DGP & model & $R^2$ (\%) & $R^{2,2}$ (\%) & score & total $R^2$ (\%) & \# factors & SR pred. & SR UMVE & SR CMVE \\",
             r"\midrule"]
    for k, (name, d) in enumerate([("well specified", oos_lin), ("misspecified", oos_nl)]):
        if k:
            lines.append(r"\midrule")
        first = True
        for model in ["m=5", "m=10", "m=20", "population"]:
            m = metrics(d[d.model == model])
            lab = "$" + model + "$" if model != "population" else "population"
            lines.append(f"{name if first else ''} & {lab} & {100 * m['R2']:.2f} & {100 * m['R22']:.2f} & {m['Score']:.1f} & "
                         f"{100 * m['TotR2']:.1f} & {m['nfac']:.1f} & {m['SR_pred']:.2f} & {m['SR_umve']:.2f} & {m['SR_cmve']:.2f} \\\\")
            first = False
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(tab, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def run_simulation(out, fig, tab, reps=100, Ts=(50, 100, 200, 400, 800, 1600, 3200), seed=0, skip=(), log=print):
    """All three experiments; csv files <out>_*.csv, Figure C.1 <fig>.pdf, Table C.1 <tab>."""
    dgp = DGP(seed=seed)
    dgp_nl = DGP(seed=seed, nonlinear=True)
    stc = dgp.panel(np.random.default_rng([seed, 4242]), 2000)
    alpha_S, alpha, sig2 = theorem_constants(dgp, stc)
    log(f"constants: alpha = {alpha:.4g} (on S: {alpha_S:.4g}), sigma^2 = {sig2:.4g}; nonlinear features carry "
        f"{100 * dgp_nl.share_nonlinear:.0f}% of the systematic variance of the misspecified DGP")
    pd.DataFrame([dict(alpha_S=alpha_S, alpha=alpha, sigma2=sig2, share_nonlinear=dgp_nl.share_nonlinear,
                       norm_b2=float(dgp.b @ dgp.b), trC=float(np.trace(dgp.C)))]).to_csv(out + "_constants.csv", index=False)
    if "rate" not in skip:
        rate = run_rate(dgp, list(Ts), reps, seed, log)
        rate.to_csv(out + "_rate.csv", index=False)
    else:
        rate = pd.read_csv(out + "_rate.csv")
    lams = np.concatenate([[0.0], np.logspace(-6, -1, 26)])
    if "path" not in skip:
        path = run_path(dgp, [108, 1000], lams, seed, log)
        path.to_csv(out + "_path.csv", index=False)
    else:
        path = pd.read_csv(out + "_path.csv")
    if "oos" not in skip:
        oos_lin = run_oos(dgp, 720, 108, [5, 10, 20], seed, log)
        oos_lin.to_csv(out + "_oos_lin.csv", index=False)
        oos_nl = run_oos(dgp_nl, 720, 108, [5, 10, 20], seed, log)
        oos_nl.to_csv(out + "_oos_nl.csv", index=False)
    else:
        oos_lin, oos_nl = pd.read_csv(out + "_oos_lin.csv"), pd.read_csv(out + "_oos_nl.csv")
    summ = rate.groupby(["est", "T"]).err.agg(["mean", "std", lambda v: np.quantile(v, 0.95)])
    summ.columns = ["mean", "std", "q95"]
    summ["bound"] = sig2 / (alpha ** 2 * summ.index.get_level_values("T"))
    summ.to_csv(out + "_rate_summary.csv")
    for est in ["frobenius", "gram"]:
        g = summ.loc[est]
        log(f"{est}: log-log slope of the MSE {slope(g.index.values, g['mean'].values)[0]:.3f}, of the 95% quantile "
            f"{slope(g.index.values, g['q95'].values)[0]:.3f}")
    make_figure(rate, path[path.lam > 0], (alpha, sig2), fig)
    make_table(oos_lin, oos_nl, tab)
    log(f"Figure C.1 -> {fig}.pdf, Table C.1 -> {tab}")
