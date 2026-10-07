"""Pass 1: the per-month sufficient statistics of a panel (Lemma 3.3 of the paper), and their loader.

Every quantity of the estimation and of the out-of-sample evaluation in Section 7 is a function of, per month t and for
the feature matrix Phi_t of the month (raw feature coordinates, before the window basis of Section 7.1 is applied),

    N_t,  1'x_{t+1},  ||x_{t+1}||^2,  s_t = Phi_t' x_{t+1},  G_t = Phi_t' Phi_t,

with x_{t+1} the excess returns of month t+1. Pass 1 writes four files into the statistics folder:

    stats_linear.npz    the linear kernel, Phi_t = [1, z_t] (constant + characteristics), for all stocks and for the stocks
                        above the micro-cap cut (suffix _micro: rank-transformed size >= -0.6, the largest 80%)
    stats_id.npz        the statistics of the characteristic-driven idiosyncratic block (Online Appendix D):
                        Gc_<mono> = sum_i mono_i Phi_i Phi_i' for the monomials of psi = (1, a, b, c), a = size, b =
                        idiosyncratic volatility, c = return volatility, Sid4 = sum_i x_i^2 psi_i psi_i',
                        Q4 = sum_i (psi_i (x) psi_i)(psi_i (x) psi_i)'
    stats_gauss_l1.npz, stats_gauss_l0.5.npz
                        the additive kernel, Phi_t = [1, z_t, k_G(z_t, landmarks) B], 200 landmarks chosen by pivoted
                        Cholesky on 20,000 stock-months of the first 96 months, bandwidth ell^2 = mult^2 times the median
                        squared distance (mult = 1 and 0.5); G_t is stored in float32 (upper triangle)

The excess return is the gross return minus the riskless return 1/bond_t - 1 of the panel. With a Fama-French factor
file, pass 1 also reports the agreement of this rate with Kenneth French's RF (Online Appendix D.1).
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .data import char_index, excess_returns, load_ff5, load_month, month_available, panel_columns, read_datefile
from .features import additive_features, characteristics, gauss_kernel, linear_features, nystrom_basis, pivoted_cholesky

ID_CHARS = {"a": "mvel1", "b": "idiovol", "c": "retvol"}
MONOS = ["a", "b", "c", "a2", "b2", "c2", "ab", "ac", "bc"]
MICROCAP_CUT = -0.6


# ----------------------------------------------------------------------------------------------------------- loading
def unpack_triu(tri: np.ndarray, d: int) -> np.ndarray:
    """(T, d(d+1)/2) upper triangles -> (T, d, d) symmetric float64 matrices."""
    iu = np.triu_indices(d)
    G = np.zeros((tri.shape[0], d, d))
    for k in range(tri.shape[0]):
        Gk = np.zeros((d, d))
        Gk[iu] = tri[k]
        G[k] = Gk + Gk.T - np.diag(np.diag(Gk))
    return G


class Stats:
    """The statistics of one feature construction, as arrays over the months k = 0..T-1 of the panel:
    t (month number of the panel), date0 (end of month t), N, sumx, sumx2, s (T x d), G (T x d x d), w = 1/N."""

    def __init__(self, path: str, universe: str = "full"):
        z = np.load(path, allow_pickle=False)
        suf = "" if universe == "full" else "_micro"
        if "N" + suf not in z.files:
            raise ValueError(f"{path} has no statistics of the universe '{universe}'")
        self.path = path
        self.t, self.date0, self.date1 = z["t"], z["date0"], z["date1"]
        self.d = int(z["d"])
        self.T = len(self.t)
        self.N = z["N" + suf].astype(float)
        self.sumx, self.sumx2 = z["sumx" + suf], z["sumx2" + suf]
        self.s = z["s" + suf]
        self.G = unpack_triu(z["G" + suf], self.d)
        self.w = 1.0 / np.maximum(self.N, 1.0)
        self.meta = {k: z[k] for k in z.files if k in ("ell2", "ell2_median", "landmarks", "B", "d_lin", "nystrom_share",
                                                        "rf", "bond", "microcap_cut", "char_names")}

    def constant_first(self, tol: float = 1e-6) -> bool:
        """Whether the first feature is the constant (G_t[0, 0] = N_t and s_t[0] = 1'x_{t+1}): needed for the GMV portfolio."""
        e1 = np.max(np.abs(self.G[:, 0, 0] - self.N) / self.N)
        e2 = np.max(np.abs(self.s[:, 0] - self.sumx) / np.sqrt(self.N * self.sumx2))
        return bool(e1 < tol and e2 < tol)


# ----------------------------------------------------------------------------------------------------------- pass 1
def _month_linear(folder, t, cols):
    """Statistics of month t for the linear kernel, the micro-cap universe and the idiosyncratic block."""
    d = load_month(folder, t)
    x = excess_returns(d["R"], d["bond"])
    Z = d["Z"]
    Phi = linear_features(Z)
    out = dict(N=len(x), sumx=float(x.sum()), sumx2=float(x @ x), s=Phi.T @ x, G=(Phi.T @ Phi)[np.triu_indices(Phi.shape[1])],
               bond=d["bond"], nperm=len(np.unique(d["permno"])) if d["permno"] is not None else len(x))
    f = Z[:, cols["mvel1"]] >= MICROCAP_CUT
    Pf, xf = Phi[f], x[f]
    out.update(N_micro=len(xf), sumx_micro=float(xf.sum()), sumx2_micro=float(xf @ xf), s_micro=Pf.T @ xf,
               G_micro=(Pf.T @ Pf)[np.triu_indices(Phi.shape[1])])
    # idiosyncratic block: psi = (1, a, b, c)
    a, b, c = (Z[:, cols[ID_CHARS[k]]] for k in "abc")
    mono = dict(a=a, b=b, c=c, a2=a * a, b2=b * b, c2=c * c, ab=a * b, ac=a * c, bc=b * c)
    iu = np.triu_indices(Phi.shape[1])
    for k in MONOS:
        out[f"Gc_{k}"] = ((Phi * mono[k][:, None]).T @ Phi)[iu].astype(np.float32)
    psi = np.column_stack([np.ones_like(a), a, b, c])
    out["Sid4"] = (psi * (x * x)[:, None]).T @ psi
    kr = np.einsum("ni,nj->nij", psi, psi).reshape(len(x), 16)
    out["Q4"] = kr.T @ kr
    return out


def _month_gauss(folder, t, gauss):
    """Statistics of month t for the additive kernel at every bandwidth of `gauss` ({mult: (landmarks, B, ell2)})."""
    d = load_month(folder, t)
    x = excess_returns(d["R"], d["bond"])
    out = {}
    for mult, (zeta, B, ell2) in gauss.items():
        Phi = additive_features(d["Z"], zeta, B, ell2)
        PhiG = Phi[:, 1 + characteristics(d["Z"]).shape[1]:]
        out[mult] = dict(N=len(x), sumx=float(x.sum()), sumx2=float(x @ x), s=Phi.T @ x,
                         G=(Phi.T @ Phi)[np.triu_indices(Phi.shape[1])].astype(np.float32),
                         nres=float(np.mean(1.0 - np.sum(PhiG ** 2, axis=1))))
    return out


def _pool_map(fn, args, workers):
    if workers <= 1:
        return [fn(*a) for a in args]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fn, *a) for a in args]
        return [f.result() for f in futs]


def riskless_check(bond: np.ndarray, date1: np.ndarray, ff5_path: str) -> str:
    """Agreement of the panel's riskless rate 1/bond_t - 1 with Kenneth French's RF of the return month."""
    rf = load_ff5(ff5_path)["RF"]
    ym = np.array([str(x)[:7] for x in date1])
    ok = np.isin(ym, rf.index)
    a, b = 1.0 / bond[ok] - 1.0, rf.loc[ym[ok]].values
    return (f"riskless rate over {ok.sum()} months ({ym[ok][0]} to {ym[ok][-1]}): correlation with Kenneth French's RF "
            f"{np.corrcoef(a, b)[0, 1]:.3f}, mean absolute difference {1e4 * np.abs(a - b).mean():.2f} bp per month")


def build_stats(panel: str, out: str, landmarks: int = 200, pool_months: int = 96, pool_size: int = 20000,
                bw_mults=(1.0, 0.5), seed: int = 0, workers: int = 1, ff5: str | None = None, log=print):
    """Pass 1 over the panel folder `panel`; writes the statistics files into the folder `out`."""
    os.makedirs(out, exist_ok=True)
    tt, d0, d1 = read_datefile(panel)
    keep = [i for i, t in enumerate(tt) if month_available(panel, int(t))]
    tt, d0, d1 = tt[keep], d0[keep], d1[keep]
    T = len(tt)
    names = panel_columns(panel)[1:]
    chars = names[:len(names) - 8]
    cols = {v: char_index(panel, v) for v in ID_CHARS.values()}
    log(f"panel {panel}: {T} months, {len(chars)} characteristics; micro-cap filter and idiosyncratic block on "
        + ", ".join(f"{k} = {names[v]}" for k, v in cols.items()))
    t0 = time.time()

    # ---- linear kernel, micro-cap universe, idiosyncratic block
    res = _pool_map(_month_linear, [(panel, int(t), cols) for t in tt], workers)
    keys = list(res[0].keys())
    arr = {k: np.asarray([r[k] for r in res]) for k in keys}
    p = 1 + len(chars)
    base = dict(t=tt, date0=d0, date1=d1, d=p, char_names=np.array(chars))
    lin_keys = ["N", "sumx", "sumx2", "s", "G", "N_micro", "sumx_micro", "sumx2_micro", "s_micro", "G_micro", "bond", "nperm"]
    np.savez_compressed(os.path.join(out, "stats_linear.npz"), microcap_cut=MICROCAP_CUT, rf=1.0 / arr["bond"] - 1.0,
                        **base, **{k: arr[k] for k in lin_keys})
    np.savez_compressed(os.path.join(out, "stats_id.npz"), t=tt, d=p, id_chars=np.array([names[cols[ID_CHARS[k]]] for k in "abc"]),
                        monos=np.array(MONOS), **{k: arr[k] for k in [f"Gc_{m}" for m in MONOS] + ["Sid4", "Q4"]})
    log(f"stats_linear.npz, stats_id.npz: {T} months, {p} features, N_t from {int(arr['N'].min())} to {int(arr['N'].max())} "
        f"(mean {arr['N'].mean():.0f}), {time.time() - t0:.0f}s")
    if ff5:
        log(riskless_check(arr["bond"], d1, ff5))

    # ---- additive kernel: landmark pool and bandwidth from the first pool_months months
    rng = np.random.default_rng(seed)
    pool = np.vstack([characteristics(load_month(panel, int(tt[k]))["Z"]) for k in range(min(pool_months, T))])
    if len(pool) > pool_size:
        pool = pool[rng.choice(len(pool), pool_size, replace=False)]
    n = len(pool)
    ia, ib = rng.integers(0, n, 200000), rng.integers(0, n, 200000)
    d2 = np.sum((pool[ia] - pool[ib]) ** 2, axis=1)
    ell2_med = float(np.median(d2[ia != ib]))
    log(f"landmark pool: {n} stock-months of the first {min(pool_months, T)} months; median squared distance ell_0^2 = "
        f"{ell2_med:.4f} ({ell2_med / pool.shape[1]:.3f} per characteristic)")
    gauss = {}
    for mult in bw_mults:
        ell2 = mult ** 2 * ell2_med
        piv = pivoted_cholesky(pool, ell2, min(landmarks, n))
        zeta = pool[piv]
        B = nystrom_basis(zeta, ell2)
        share = 1.0 - float(np.mean(1.0 - np.sum((gauss_kernel(pool, zeta, ell2) @ B) ** 2, axis=1)))
        log(f"bandwidth {mult:g} x ell_0: {len(piv)} landmarks capture {100 * share:.0f}% of the trace of the Gaussian "
            f"kernel matrix of the pool")
        gauss[mult] = (zeta, B, ell2, share)
    res = _pool_map(_month_gauss, [(panel, int(t), {m: g[:3] for m, g in gauss.items()}) for t in tt], workers)
    for mult, (zeta, B, ell2, share) in gauss.items():
        arr = {k: np.asarray([r[mult][k] for r in res]) for k in res[0][mult]}
        path = os.path.join(out, f"stats_gauss_l{mult:g}.npz")
        np.savez_compressed(path, t=tt, date0=d0, date1=d1, d=p + len(zeta), d_lin=p, ell2=ell2, ell2_median=ell2_med,
                            landmarks=zeta, B=B, nystrom_share=share, pool_months=min(pool_months, T), seed=seed, **arr)
        log(f"{os.path.basename(path)}: {p + len(zeta)} features, {time.time() - t0:.0f}s")
