"""The tables of the paper from the run files of a panel (<work>/runs) and the pass-2 files (<work>/pass2).

Main panel (gkx):
    table1_stat.tex         Table 1    out-of-sample statistical performance
    table2_ap.tex           Table 2    portfolios
    table3_decomp.tex       Table 3    decomposition of the gain over IPCA
    tableD1_periods.tex     Table D.1  sub-periods
    tableD2_robust.tex      Table D.2  robustness
    tableD3_floorshape.tex  Table D.3  the shape of the floor along the multiplier grid
    tableD4_gmv.tex         Table D.4  covariance forecasts
Second panel (jkp):
    tableD5_jkp.tex         Table D.5  the study on the panel of Jensen, Kelly and Pedersen (2023)
    tableD6_decomp_jkp.tex  Table D.6  its decomposition
For the second panel the tables 1, 2, D.1-D.4 are written as well (not in the paper; the numbers of Online Appendix D.2
are quoted from them). Every table is also written as CSV with all columns.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ..config import PanelConfig
from ..data import load_ff5
from .common import (exists, ff5_regression, in_period, load_run, max_drawdown, nw_var_mean, paired, pick, sharpe_diff_test,
                     sr, stat, tex_table, tstat)

PASS2_RUNS = (("cmk_linear", "linear"), ("cmk_gauss_l0.5", "l0.5"), ("cmk_gauss_l1", "l1"))
LAB_LIN = "CMK, linear"
LAB_ADD = {"l1": r"CMK, linear + Gaussian ($\ell=1\times$)", "l0.5": r"CMK, linear + Gaussian ($\ell=0.5\times$)"}
LAB_SEL = r"CMK, linear + Gaussian, $(\ell,m)$ selected"


class Results:
    """The runs of a panel restricted to the common test months (t >= first_test)."""

    def __init__(self, work: str, cfg: PanelConfig, ff5_path: str | None = None):
        self.work, self.cfg = work, cfg
        self.ff5 = load_ff5(ff5_path) if ff5_path else None
        MF = cfg.MF
        F = lambda n, refit=True: load_run(work, n, refit)
        lin_all, g1_all, g05_all = F("cmk_linear"), F("cmk_gauss_l1"), F("cmk_gauss_l0.5")
        lin, g1, g05 = (d[d.selected == 1] for d in (lin_all, g1_all, g05_all))
        ip = pd.concat([load_run(work, f"ipca_K{K}") for K in cfg.ipca_K if exists(work, f"ipca_K{K}")], ignore_index=True)
        allg = pd.concat([g05.assign(run="l0.5"), g1.assign(run="l1"), lin[lin.m == MF].assign(run="linear")], ignore_index=True)
        allg = allg.sort_values(["t", "run", "m"])
        gsel = allg.loc[allg.groupby("t")["val_score"].idxmin()]
        common = sorted(t for t in set(gsel.t) & set(lin.t) & set(ip.t) if t >= cfg.first_test)
        self.common = common
        cut = lambda d: d[d.t.isin(common)]
        self.cut = cut
        self.lin, self.gsel, self.g05, self.g1, self.ip = map(cut, (lin, gsel, g05, g1, ip))
        self.lin_all, self.g1_all, self.g05_all = map(cut, (lin_all, g1_all, g05_all))
        self.lin_sel = {m: self.lin[self.lin.m == m].sort_values("t") for m in sorted(self.lin.m.unique())}
        self.lin_joint = pick(self.lin, "val_score", ["t"])
        self.gsel_i = self.gsel.set_index("t").sort_index()
        self.lin_ref = self.lin_sel[MF].set_index("t").sort_index()
        self.Ks = [K for K in cfg.ipca_K if (self.ip.m == K).any()]

    def load(self, name, refit=True):
        return self.cut(load_run(self.work, name, refit)) if exists(self.work, name, refit) else None

    def pass2(self, f):
        p = os.path.join(self.work, "pass2", f)
        return pd.read_csv(p) if os.path.exists(p) else None


def _sub(d, m, gamma=None, thr=None):
    g = d[d.m == m]
    if gamma is not None:
        g = g[(g.gamma == gamma) & (g.thr == thr)]
    return g.sort_values("t")


# ------------------------------------------------------------------------------------------------- Table 1
def table1(R: Results, out: str):
    cfg = R.cfg
    rows, csv = [], []

    def add(label, g, ref=None):
        r = dict(model=label, **stat(g))
        gi = g.set_index("t").sort_index()
        r["TotR2adj"] = 100 * (g.tot_r2 - g.u_id * g.rank_phi / (g.N * g.r2_den)).mean()
        if ref is not None:
            r.update(paired(gi, ref))
        pg = paired(gi, R.gsel_i)
        r.update(d_score_g=pg["d_score"], t_g=pg["t_score"], p_LW_g=pg["p_LW"])
        r.update(gamma_pos=100 * (g.gamma > 0).mean() if "gamma" in g.columns else np.nan,
                 lam_pos=100 * (g.thr > 0).mean() if "thr" in g.columns else np.nan)
        rows.append(r)
        csv.append(r)
    for m in cfg.m_lin:
        add(f"{LAB_LIN}, $m={m}$", R.lin_sel[m], R.lin_ref)
    add(f"{LAB_LIN}, $m$ selected", R.lin_joint, R.lin_ref)
    rows.append(None)
    for bw, d in (("l1", R.g1), ("l0.5", R.g05)):
        for m in cfg.m_add:
            add(f"{LAB_ADD[bw]}, $m={m}$", d[d.m == m], R.lin_ref)
    add(LAB_SEL, R.gsel, R.lin_ref)
    rows[-1]["d_score_g"] = np.nan
    rows[-1]["t_g"] = np.nan
    rows.append(None)
    for K in R.Ks:
        add(f"IPCA, $K={K}$", R.ip[R.ip.m == K], R.lin_ref)
    cols = ["model", "nfac", "R2", "R22", "Score", "TotR2", "TotR2adj", "rho", "d_score", "t_score", "d_score_g", "t_g"]
    hdr = ["model", "\\# factors", "$R^2$ (\\%)", "$R^{2,2}$ (\\%)", "score", "total $R^2$ (\\%)", "adj.\\ total $R^2$ (\\%)",
           "$\\rho^{\\bm f}$ (\\%)", "diff.\\ lin.", "$t$", "diff.\\ Gauss.", "$t$"]
    tex_table(rows, cols, hdr, [0, 1, 2, 2, 0, 1, 1, 1, 1, 1, 1, 1], out + ".tex")
    pd.DataFrame(csv).to_csv(out + ".csv", index=False)
    return pd.DataFrame(csv)


# ------------------------------------------------------------------------------------------------- Table 2
def _costs(g):
    if g is None or not len(g):
        return {}
    g = g.sort_values("t")
    tu = g.sdf_turnover_unit.fillna(0)
    ru = g.sdf_ret_unit
    return dict(turn=tu.mean(), net0=sr(ru), net10=sr(ru - 10e-4 * tu), net25=sr(ru - 25e-4 * tu),
                net50=sr(ru - 50e-4 * tu), net100=sr(ru - 100e-4 * tu), lev=g.sdf_lev.mean(), nlong=g.sdf_nlong.mean(),
                nshort=g.sdf_nshort.mean(), maxw=g.sdf_maxw.mean(), months_cost=len(g))


def pass2_frames(R: Results):
    """The pass-2 rows of the selected models (name refit_sel) of the three runs and IPCA, and the switching pass."""
    parts = []
    for f, run in PASS2_RUNS + (("ipca", "ipca"),):
        p = R.pass2(f"{f}_pass.csv.gz")
        if p is not None:
            parts.append(p.assign(run=run))
    p2 = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["t", "m", "run", "name"])
    p2 = p2[p2.t.isin(R.common) & (p2.name == "refit_sel")]
    sw = R.pass2("switch_pass.csv.gz")
    if sw is not None:
        sw = sw[sw.t.isin(R.common)]
    return p2, sw


def table2(R: Results, out: str, mdd: bool = False):
    cfg = R.cfg
    p2, sw = pass2_frames(R)
    ip60 = R.ip[R.ip.m == 60].set_index("t").sort_index().ret_sdf

    def p_ipca(g):
        r = g.set_index("t").sort_index().ret_sdf
        tt = r.index.intersection(ip60.index)
        return sharpe_diff_test(r.loc[tt].values, ip60.loc[tt].values)[2] if len(tt) > 24 else np.nan

    def cost(m, run):
        return _costs(p2[(p2.m == m) & (p2.run == run)])

    def cost_sel(sel):
        if sw is not None:
            return _costs(sw.merge(sel[["t", "run", "m"]], on=["t", "run", "m"]))
        return _costs(p2.merge(sel[["t", "run", "m"]], on=["t", "run", "m"]))
    rows, csv = [], []

    def add(label, g, costs):
        r = dict(model=label, **stat(g))
        thetabar = g.theta_frac.mean()
        r["SR_pred"] = np.sqrt(12 * thetabar / (1 - thetabar))
        r["p_ew"] = sharpe_diff_test(g.ret_sdf.values, g.ret_ew.values, 12)[2]
        if R.ff5 is not None:
            r.update(ff5_regression(g.ret_sdf.values, g.ym.values, R.ff5))
        r["p_ipca"] = p_ipca(g)
        r.update(costs)
        rows.append(r)
        csv.append(r)
    for m in cfg.m_lin:
        add(f"{LAB_LIN}, $m={m}$", R.lin_sel[m], cost(m, "linear"))
    rows.append(None)
    add(f"{LAB_ADD['l1']}, $m={cfg.MA}$", R.g1[R.g1.m == cfg.MA], cost(cfg.MA, "l1"))
    add(f"{LAB_ADD['l0.5']}, $m={cfg.MA}$", R.g05[R.g05.m == cfg.MA], cost(cfg.MA, "l0.5"))
    add(LAB_SEL, R.gsel, cost_sel(R.gsel))
    rows.append(None)
    for K in [K for K in (20, 40, 60) if K in R.Ks]:
        add(f"IPCA, $K={K}$", R.ip[R.ip.m == K], cost(K, "ipca"))
        if K == 60:
            rows[-1]["p_ipca"] = np.nan
    rows.append(None)
    g = R.gsel
    ew = dict(model="$1/N_t$", SR_umve=sr(g.ret_ew), t_NW=g.ret_ew.mean() / np.sqrt(nw_var_mean(g.ret_ew.values, 12)),
              MDD=100 * max_drawdown(g.sort_values("t").ret_ew))
    if R.ff5 is not None:
        ew.update(ff5_regression(g.ret_ew.values, g.ym.values, R.ff5))
    rows.append(ew)
    csv.append(ew)
    cols = ["model", "SR_pred", "SR_umve", "t_NW", "MDD", "SR_cmve", "p_ew", "p_ipca", "alpha", "t_alpha", "adjR2", "turn",
            "net0", "net10", "net25", "net50", "net100"]
    hdr = ["model", "SR pred.", "SR UMVE", "$t$", "max DD (\\%)", "SR CMVE", "$p$ vs $1/N$", "$p$ vs IPCA", "intercept (\\%)",
           "$t$ (int.)", "adj.\\ $R^2$ (\\%)", "turnover", "SR net 0", "SR net 10", "SR net 25", "SR net 50", "SR net 100"]
    fmts = [0, 2, 2, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2]
    if not mdd:
        k = cols.index("MDD")
        cols, hdr, fmts = cols[:k] + cols[k + 1:], hdr[:k] + hdr[k + 1:], fmts[:k] + fmts[k + 1:]
    tex_table(rows, cols, hdr, fmts, out + ".tex")
    pd.DataFrame(csv).to_csv(out + ".csv", index=False)
    return pd.DataFrame(csv)


# ------------------------------------------------------------------------------------------------- Table 3 / D.6
STEPS = [("IPCA K=60", r"IPCA, $K=60$"), ("IPCA K=95", r"IPCA, $K=95$"),
         ("moment step", r"Moment step, full feature space, Tikhonov $\varrho=10^{-3}$"),
         ("eta=0 lam=0", r"Noise correction: subtract $\hat u\bm G^+_{t,\varrho}$"),
         ("eta=1 lam=0", r"Floor at $\eta=1$"), ("CMK", r"Select $(\eta,\lsy)$: CMK estimator")]


def decomposition(R: Results, out: str):
    MF = R.cfg.MF
    ms = R.load("moment_step_linear")
    spec = {"IPCA K=60": R.ip[R.ip.m == 60], "IPCA K=95": R.ip[R.ip.m == 95] if 95 in R.Ks else None,
            "moment step": ms[ms.m == MF] if ms is not None else None,
            "eta=0 lam=0": _sub(R.lin_all, MF, 0.0, 0.0), "eta=1 lam=0": _sub(R.lin_all, MF, 1.0, 0.0), "CMK": R.lin_sel[MF]}
    steps = [(k, lab) for k, lab in STEPS if spec[k] is not None and len(spec[k])]
    rows, lines = [], []
    for j, (k, lab) in enumerate(steps):
        g = spec[k].sort_values("t")
        r = dict(step=k, label=lab, score=(g.score_bm - g.score).mean(), sr=sr(g.ret_sdf.values))
        if j:
            a = g.set_index("t")
            b = spec[steps[j - 1][0]].set_index("t")
            ix = a.index.intersection(b.index)
            d = ((a.score_bm - a.score) - (b.score_bm - b.score)).loc[ix].values
            r.update(diff=d.mean(), t=tstat(d))
            lines.append(" & ".join([lab, f"{r['score']:.0f}", f"${r['diff']:+.1f}$", f"${r['t']:.1f}$", f"{r['sr']:.2f}"]) + r" \\")
        else:
            lines.append(" & ".join([lab, f"{r['score']:.0f}", "", "", f"{r['sr']:.2f}"]) + r" \\")
        rows.append(r)
    head = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Step & Score & Change & $t$ & SR \\", r"\midrule"]
    with open(out + ".tex", "w") as fh:
        fh.write("\n".join(head + lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")
    pd.DataFrame(rows).to_csv(out + ".csv", index=False)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------- Table D.1
def table_periods(R: Results, out: str):
    MF = R.cfg.MF
    models = [(LAB_SEL, R.gsel), (f"{LAB_LIN}, $m={MF}$", R.lin_sel[MF]), ("IPCA, $K=60$", R.ip[R.ip.m == 60]),
              ("IPCA, $K=20$", R.ip[R.ip.m == 20])]
    rows, csv = [], []
    for pname, a, b in R.cfg.periods:
        for lab, d in models:
            gg = d[in_period(d.ym, a, b)] if (a or b) else d
            r = dict(period=pname, model=lab, **stat(gg))
            if lab != models[0][0]:
                r.update(paired(gg.set_index("t").sort_index(), R.gsel_i))
            rows.append(r)
            csv.append(r)
        rows.append(None)
    rows.pop()
    cols = ["period", "model", "months", "R2", "R22", "Score", "SR_umve", "SR_ew", "d_score", "t_score"]
    hdr = ["period", "model", "months", "$R^2$ (\\%)", "$R^{2,2}$ (\\%)", "score", "SR UMVE", "SR $1/N$", "diff.\\ score", "$t$"]
    tex_table(rows, cols, hdr, [0, 0, 0, 2, 2, 0, 2, 2, 1, 1], out + ".tex", align="llrrrrrrrr")
    pd.DataFrame(csv).to_csv(out + ".csv", index=False)
    return pd.DataFrame(csv)


# ------------------------------------------------------------------------------------------------- Table D.2
def table_robust(R: Results, out: str):
    MF, MA = R.cfg.MF, R.cfg.MA
    rows, csv = [], []
    lin_sel = R.lin_sel

    def add(label, g, ref=None, note=""):
        r = dict(model=label, note=note, **stat(g))
        if ref is not None:
            r.update(paired(g.set_index("t").sort_index(), ref.set_index("t").sort_index()))
        rows.append(r)
        csv.append(r)
    # block 1: the regularizers
    blocks = [(f"linear $m={MF}$", R.lin_all, lin_sel[MF], R.load("moment_step_linear"), R.load("idfloor_linear"), MF),
              (f"linear + Gaussian ($\\ell=1\\times$) $m={MA}$", R.g1_all, R.g1[R.g1.m == MA], R.load("moment_step_gauss_l1"),
               R.load("idfloor_gauss_l1"), MA),
              (f"linear + Gaussian ($\\ell=0.5\\times$) $m={MA}$", R.g05_all, R.g05[R.g05.m == MA],
               R.load("moment_step_gauss_l0.5"), None, MA)]
    for lab, d_all, d_sel, former, idf, m in blocks:
        add(f"{lab}: $(\\eta,\\lsy)$ selected (Table~\\ref{{tab_stat}})", d_sel)
        add(f"{lab}: $\\eta=0$, $\\lsy=0$", _sub(d_all, m, 0.0, 0.0), d_sel)
        add(f"{lab}: $\\eta=1$, $\\lsy=0$", _sub(d_all, m, 1.0, 0.0), d_sel)
        add(f"{lab}: $\\eta=0$, $\\lsy$ selected", pick(_sub(d_all, m)[lambda g: g.gamma == 0], "val_score", ["t"]), d_sel)
        if former is not None:
            add(f"{lab}: moment step, $\\hat{{\\bm F}}$ in every direction", _sub(former, m), d_sel)
        if idf is not None:
            idf_all, idf = idf, idf[(idf.m == m) & (idf.selected == 1)]
            on = lambda d: d[d.t.isin(idf.t)].sort_values("t")
            note = f"{len(idf)} months" if len(idf) < len(d_sel) else ""
            if note:
                add(f"{lab}: $(\\eta,\\lsy)$ selected, on the months of the next row", on(d_sel), note=note)
            add(f"{lab}: identity-shaped floor, $(\\eta,\\lsy)$ selected", idf, on(d_sel), note=note)
            add(f"{lab}: identity-shaped floor, $\\eta=1$, $\\lsy=0$", on(_sub(idf_all, m, 1.0, 0.0)), on(d_sel), note=note)
        rows.append(None)
    # block 2: the weight and the shape of the floor together (scalar weight, solver, every fourth window)
    fr_g, fr_i = R.load("scalar_floor_gram"), R.load("scalar_floor_identity")
    if fr_g is not None and fr_i is not None:
        fr_g, fr_i = fr_g[fr_g.m == MF].assign(rank_C=np.nan), fr_i[fr_i.m == MF].assign(rank_C=np.nan)
        tf = sorted(set(fr_g[fr_g.selected == 1].t) & set(fr_i[fr_i.selected == 1].t))
        on = lambda d: d[d.t.isin(tf)].sort_values("t")
        ref = on(lin_sel[MF])
        note = f"{len(tf)} months"
        add(f"linear $m={MF}$, Gram-whitened weight, floor $\\eta\\hat{{\\bm F}}$, $(\\eta,\\lsy)$ selected", ref, note=note)
        idl = R.load("idfloor_linear")
        if idl is not None:
            add(f"linear $m={MF}$, Gram-whitened weight, identity-shaped floor, $(\\eta,\\lsy)$ selected",
                on(idl[(idl.m == MF) & (idl.selected == 1)]), ref, note=note)
        add(f"linear $m={MF}$, scalar weight $N_t^{{-1/2}}\\bm I$, floor $\\eta\\hat{{\\bm F}}$, $(\\eta,\\lsy)$ selected",
            on(fr_g[fr_g.selected == 1]), ref, note=note)
        add(f"linear $m={MF}$, scalar weight $N_t^{{-1/2}}\\bm I$, identity-shaped floor, $(\\eta,\\lsy)$ selected",
            on(fr_i[fr_i.selected == 1]), ref, note=note)
        add(f"linear $m={MF}$, scalar weight $N_t^{{-1/2}}\\bm I$, $\\eta=0$, $\\lsy=0$", on(fr_g[(fr_g.gamma == 0) & (fr_g.thr == 0)]),
            ref, note=note)
        rows.append(None)
    # block 3: the weight
    br = R.load("browne_linear_mf")
    if br is not None:
        g0 = _sub(R.lin_all, MF, 0.0, 0.0)
        add(f"linear $m={MF}$, Gram-whitened weight, $\\eta=0$, $\\lsy=0$", g0)
        add(f"linear $m={MF}$, Browne weight, $\\eta=0$, $\\lsy=0$", br[(br.m == MF) & (br.lam == 0)], g0)
    fr1, fr100 = R.load("frobenius_k1"), R.load("frobenius_k100")
    m2 = R.cfg.m_lin[1]
    if fr1 is not None and fr100 is not None:
        tf = sorted(set(fr1.t) & set(fr100.t))
        on = lambda d: d[d.t.isin(tf)]
        ref2 = on(lin_sel[m2])
        add(f"linear $m={m2}$, Gram-whitened weight, $(\\eta,\\lsy)$ selected", ref2, note=f"{len(tf)} months")
        add(f"linear $m={m2}$, Gram-whitened weight, $\\eta=0$, $\\lsy=0$", on(_sub(R.lin_all, m2, 0.0, 0.0)), ref2)
        add(f"linear $m={m2}$, Frobenius weight, $\\kappa=1$", on(fr1[fr1.m == m2]), ref2)
        add(f"linear $m={m2}$, Frobenius weight, $\\kappa=100$", on(fr100[fr100.m == m2]), ref2)
        br2 = R.load("browne_linear_small")
        if br2 is not None:
            add(f"linear $m={m2}$, Browne weight, $\\eta=0$, $\\lsy=0$", on(br2[(br2.m == m2) & (br2.lam == 0)]), ref2)
        rows.append(None)
    # block 4: the protocol; the rows without re-estimation get the benchmark of the 108 months of the same test month

    def rebench(nr, ref):
        b = ref.drop_duplicates("t").set_index("t").score_bm
        nr = nr.copy()
        nr["score_bm"] = nr.t.map(b).values
        return nr
    nr_lin = R.cut(load_run(R.work, "cmk_linear", refit=False))
    nr_lin = rebench(nr_lin[nr_lin.selected == 1], lin_sel[MF])
    add(f"no re-estimation, linear $m={MF}$", nr_lin[nr_lin.m == MF], lin_sel[MF])
    nr_g = pd.concat([rebench(R.cut(load_run(R.work, "cmk_gauss_l0.5", False)), R.gsel).assign(run="l0.5"),
                      rebench(R.cut(load_run(R.work, "cmk_gauss_l1", False)), R.gsel).assign(run="l1"),
                      nr_lin[nr_lin.m == MF].assign(run="linear")], ignore_index=True)
    nr_g = nr_g[nr_g.selected == 1].sort_values(["t", "run", "m"])
    nr_g = nr_g.loc[nr_g.groupby("t")["val_score"].idxmin()]
    add("no re-estimation, linear + Gaussian selected", nr_g, R.gsel)
    ipn = R.load("ipca_K60_norefit", refit=None)
    if ipn is not None:
        ipn = rebench(ipn, R.ip[R.ip.m == 60])
        add("no re-estimation, IPCA $K=60$", ipn[ipn.m == 60], R.ip[R.ip.m == 60])
    for name, lab in [("cmk_linear_val36", f"36 validation months, 72 training months, linear $m={MF}$"),
                      ("cmk_linear_val1", f"1 validation month, 107 training months, linear $m={MF}$"),
                      ("cmk_linear_large80", f"smallest 20\\% removed, linear $m={MF}$")]:
        d = R.load(name)
        if d is not None:
            d = d[(d.selected == 1) & (d.m == MF)]
            add(lab, d, lin_sel[MF] if "large80" not in name else None)
    rows.append(None)
    # block 5: the one-step estimator by the solver against the CMK estimator, eta = 1, lambda = 0
    jt = R.load("onestep_linear_mf")
    if jt is not None:
        jt = jt[jt.m == MF]
        cf = _sub(R.lin_all, MF, 1.0, 0.0)
        cf = cf[cf.t.isin(jt.t)]
        add(f"CMK estimator, $\\eta=1$, $\\lsy=0$, linear $m={MF}$", cf, note=f"{len(cf)} months")
        add(f"one-step estimator by the solver, $\\eta=1$, $\\lsy=0$, linear $m={MF}$", jt[jt.t.isin(cf.t)], cf)
        rows[-1]["nfac"] = np.nan
    if rows and rows[-1] is None:
        rows.pop()
    cols = ["model", "nfac", "R2", "R22", "Score", "TotR2", "SR_umve", "SR_cmve", "d_score", "t_score", "p_LW"]
    hdr = ["specification", "\\# factors", "$R^2$ (\\%)", "$R^{2,2}$ (\\%)", "score", "total $R^2$ (\\%)", "SR UMVE", "SR CMVE",
           "diff.\\ score", "$t$", "$p$ SR"]
    tex_table(rows, cols, hdr, [0, 1, 2, 2, 0, 1, 2, 2, 1, 1, 2], out + ".tex")
    pd.DataFrame(csv).to_csv(out + ".csv", index=False)
    return pd.DataFrame(csv)


# ------------------------------------------------------------------------------------------------- Table D.3
def table_floorshape(R: Results, out: str):
    MF, MA = R.cfg.MF, R.cfg.MA
    idl, idg = R.load("idfloor_linear"), R.load("idfloor_gauss_l1")
    if idl is None or idg is None:
        return None
    rows, csv = [], []

    def grid_row(label, d, m):
        d = d[(d.m == m) & (d.thr == 0)]
        r = dict(model=label, months=d.t.nunique())
        for g_ in [0.0, 0.5, 1.0, 2.0]:
            x = d[d.gamma == g_]
            r[f"s{g_}"] = (x.score_bm - x.score).mean()
            r[f"f{g_}"] = x.rank_C.mean()
            r[f"r{g_}"] = 100 * (1 - x.r22_num.sum() / x.r22_den.sum())
        rows.append(r)
        csv.append(r)
    grid_row(f"linear $m={MF}$, floor $\\eta\\hat{{\\bm F}}$", R.lin_all, MF)
    grid_row(f"linear $m={MF}$, identity-shaped floor", idl, MF)
    rows.append(None)
    grid_row(f"linear + Gaussian ($\\ell=1\\times$) $m={MA}$, floor $\\eta\\hat{{\\bm F}}$", R.g1_all[R.g1_all.t.isin(idg.t)], MA)
    grid_row(f"linear + Gaussian ($\\ell=1\\times$) $m={MA}$, identity-shaped floor", idg, MA)
    gs = [0.0, 0.5, 1.0, 2.0]
    cols = ["model", "months"] + [f"{k}{g_}" for g_ in gs for k in ("s", "f", "r")]
    hdr = ["specification", "months"] + [f"{k} ({g_:g})" for g_ in gs for k in ("score", "factors", "$R^{2,2}$")]
    tex_table(rows, cols, hdr, [0, 0] + [0, 1, 2] * 4, out + ".tex")
    pd.DataFrame(csv).to_csv(out + ".csv", index=False)
    return pd.DataFrame(csv)


# ------------------------------------------------------------------------------------------------- Table D.4
def gmv_summary(R: Results, from_month: str = "1985-01"):
    """GMV volatility, calibration and the split of the score, over all common test months and from `from_month` on."""
    MF, MA = R.cfg.MF, R.cfg.MA
    models = {f"linear m={m}": R.lin_sel[m] for m in R.cfg.m_lin}
    models["additive (l,m) selected"] = R.gsel
    models[f"additive l=1x m={MA}"] = R.g1[R.g1.m == MA]
    models[f"additive l=0.5x m={MA}"] = R.g05[R.g05.m == MA]
    for K in R.Ks:
        models[f"IPCA K={K}"] = R.ip[R.ip.m == K]
    if any("r_gmv" not in d.columns or d.r_gmv.isna().any() for d in models.values()):
        return None
    months = sorted(set.intersection(*[set(d.t) for d in models.values()]))
    ym = R.lin_ref.loc[months].ym.values
    ref_lin, ref_ipca = f"linear m={MF}", "IPCA K=60"
    out = []
    for lab, sel in (("FULL SAMPLE", np.ones(len(months), bool)), (f"RETURNS FROM {from_month}", ym >= from_month)):
        ms = [t for t, k in zip(months, sel) if k]
        D = {k: d[d.t.isin(ms)].sort_values("t").set_index("t") for k, d in models.items()}
        rl, ri = D[ref_lin], D.get(ref_ipca)
        sq = lambda r: (r - r.mean()) ** 2
        rows = []
        for k, d in D.items():
            sb, s0, s = d.score_bm, d.score0, d.score
            zg = d.r_gmv / np.sqrt(d.pv_gmv)
            zs = d.ret_sdf / np.sqrt(d.pv_sdf)
            dv = (sq(d.r_gmv) - sq(rl.r_gmv)).values
            r = dict(model=k, months=len(d), gmv_sd=100 * np.sqrt(12) * d.r_gmv.std(ddof=1), gmv_dvar=12 * 1e4 * dv.mean(),
                     gmv_t=tstat(dv) if k != ref_lin else np.nan, gmv_varz=zg.var(ddof=1), sdf_varz=zs.var(ddof=1),
                     score=(sb - s).mean(), cov=(sb - s0).mean(), mean=(s0 - s).mean())
            for tag, ref in (("ipca60", ri), ("lin95", rl)):
                if ref is None:
                    continue
                same = d is ref
                r[f"dscore_{tag}"] = r["score"] - (ref.score_bm - ref.score).mean()
                r[f"dcov_{tag}"] = r["cov"] - (ref.score_bm - ref.score0).mean()
                r[f"dmean_{tag}"] = r["mean"] - (ref.score0 - ref.score).mean()
                r[f"t_dscore_{tag}"] = np.nan if same else tstat(((sb - s) - (ref.score_bm - ref.score)).values)
                r[f"t_dcov_{tag}"] = np.nan if same else tstat(((sb - s0) - (ref.score_bm - ref.score0)).values)
                r[f"t_dmean_{tag}"] = np.nan if same else tstat(((s0 - s) - (ref.score0 - ref.score)).values)
            rows.append(r)
        ew = rl.ret_ew
        dv = (sq(ew) - sq(rl.r_gmv)).values
        rows.append(dict(model="equally weighted", months=len(rl), gmv_sd=100 * np.sqrt(12) * ew.std(ddof=1),
                         gmv_dvar=12 * 1e4 * dv.mean(), gmv_t=tstat(dv)))
        out.append(pd.DataFrame(rows).assign(sample=lab))
    return pd.concat(out, ignore_index=True)


def table_gmv(R: Results, out: str):
    t = gmv_summary(R)
    if t is None:
        return None
    t.to_csv(out + ".csv", index=False)
    MF = R.cfg.MF
    full = t[t["sample"] == "FULL SAMPLE"].set_index("model")
    late = t[t["sample"] != "FULL SAMPLE"].set_index("model")
    rows = [(f"linear m={m}", f"{LAB_LIN}, $m={m}$") for m in R.cfg.m_lin] + [("additive (l,m) selected", LAB_SEL)] + \
        [(f"IPCA K={K}", f"IPCA, $K={K}$") for K in R.Ks] + [("equally weighted", "equally weighted")]
    seps = {f"linear m={MF}", "additive (l,m) selected", f"IPCA K={R.Ks[-1]}"}

    def f(x, d=2, sign=False):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        if round(x, d) == 0:
            x = 0.0
        s = f"{x:+.{d}f}" if sign and x != 0 else f"{x:.{d}f}"
        return f"${s}$" if sign or x < 0 else s
    lines = [r"\begin{tabular}{lrrrrrrrrrr}", r"\toprule",
             r" & \multicolumn{4}{c}{GMV volatility} & \multicolumn{2}{c}{Variance ratio} & "
             rf"\multicolumn{{4}}{{c}}{{Score, difference to linear $m={MF}$}} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}\cmidrule(lr){8-11}",
             rf" & {_years(R, True)} & $t$ & {_years(R, False)} & $t$ & GMV & UMVE & Cov. & $t$ & Mean & $t$ \\", r"\midrule"]
    for key, lab in rows:
        a, b = full.loc[key], late.loc[key]
        ref = key == f"linear m={MF}"
        g = lambda c, d=2, s=False: "" if (ref or c not in a or pd.isna(a[c])) else f(a[c], d, s)
        cells = [lab, f(a.gmv_sd), "" if ref else f(a.gmv_t, 1), f(b.gmv_sd), "" if ref else f(b.gmv_t, 1),
                 f(a.gmv_varz) if not pd.isna(a.get("gmv_varz", np.nan)) else "",
                 f(a.sdf_varz) if not pd.isna(a.get("sdf_varz", np.nan)) else "",
                 g("dcov_lin95", 1, True), g("t_dcov_lin95", 1), g("dmean_lin95", 1, True), g("t_dmean_lin95", 1)]
        lines.append(" & ".join(cells) + r" \\")
        if key in seps:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(out + ".tex", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return t


def _years(R, full):
    y0 = R.lin_ref.ym.min()[:4]
    y1 = R.lin_ref.ym.max()[:4]
    return f"{y0}--{y1}" if full else f"1985--{y1}"


# ------------------------------------------------------------------------------------------------- Table D.5
def table_jkp(R: Results, t1: pd.DataFrame, t2: pd.DataFrame, out: str, cost_bp: float = 25.0):
    """Table D.5: the statistical measures of Table 1, the UMVE Sharpe ratio and drawdown of Table 2, and turnover and Sharpe
    ratios before and after costs of every specification."""
    p2, sw = pass2_frames(R)
    t1 = t1.set_index("model")
    t2 = t2.set_index("model")

    def costs(g):
        g = g.sort_values("t")
        tu = g.sdf_turnover_unit.fillna(0)
        ru = g.sdf_ret_unit
        return dict(turn=tu.mean(), net0=sr(ru), net=sr(ru - cost_bp * 1e-4 * tu), months_cost=len(g))
    runs = {LAB_LIN: "linear", LAB_ADD["l1"]: "l1", LAB_ADD["l0.5"]: "l0.5", "IPCA": "ipca"}
    rows, csv = [], []
    for model, r in t1.iterrows():
        if model == f"{LAB_LIN}, $m$ selected":
            continue
        if rows and ((model.startswith("CMK, linear + Gaussian") and not rows[-1]["model"].startswith("CMK, linear + Gaussian"))
                     or (model.startswith("IPCA") and not rows[-1]["model"].startswith("IPCA"))):
            rows.append(None)
        row = dict(model=model, **r.to_dict())
        if model == LAB_SEL:
            row.update(costs(sw))
        else:
            run = next(v for k, v in runs.items() if model.startswith(k + ","))
            m = int(model.rsplit("=", 1)[1].strip("$"))
            row.update(costs(p2[(p2.run == run) & (p2.m == m)]))
        rows.append(row)
        csv.append(row)
    rows.append(None)
    ew = dict(model="$1/N_t$", SR_umve=t2.loc["$1/N_t$", "SR_umve"], MDD=t2.loc["$1/N_t$", "MDD"])
    rows.append(ew)
    csv.append(ew)
    cols = ["model", "nfac", "R2", "R22", "Score", "TotR2", "TotR2adj", "d_score", "t_score", "d_score_g", "t_g", "SR_umve", "MDD",
            "turn", "net0", "net"]
    hdr = ["model", "\\# factors", "$R^2$ (\\%)", "$R^{2,2}$ (\\%)", "score", "total $R^2$ (\\%)", "adj.\\ total $R^2$ (\\%)",
           "diff.\\ lin.", "$t$", "diff.\\ Gauss.", "$t$", "SR UMVE", "max DD (\\%)", "turnover", "SR net 0", f"SR net {cost_bp:g}"]
    tex_table(rows, cols, hdr, [0, 1, 2, 2, 0, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 2], out + ".tex")
    pd.DataFrame(csv).to_csv(out + ".csv", index=False)


# ------------------------------------------------------------------------------------------------- all
def build_tables(work: str, cfg: PanelConfig, out: str, ff5: str | None, log=print):
    os.makedirs(out, exist_ok=True)
    R = Results(work, cfg, ff5)
    log(f"{cfg.name}: {len(R.common)} common test months; (bandwidth, m) selection "
        f"{R.gsel.groupby(['run', 'm']).size().to_dict()}")
    P = lambda n: os.path.join(out, n)
    if cfg.name == "jkp":
        t1 = table1(R, P("jkp_table1_stat"))
        t2 = table2(R, P("jkp_table2_ap"), mdd=True)
        table_jkp(R, t1, t2, P("tableD5_jkp"))
        decomposition(R, P("tableD6_decomp_jkp"))
        table_periods(R, P("jkp_tableD1_periods"))
        table_robust(R, P("jkp_tableD2_robust"))
        table_floorshape(R, P("jkp_tableD3_floorshape"))
        table_gmv(R, P("jkp_tableD4_gmv"))
    else:
        table1(R, P("table1_stat"))
        table2(R, P("table2_ap"))
        decomposition(R, P("table3_decomp"))
        table_periods(R, P("tableD1_periods"))
        table_robust(R, P("tableD2_robust"))
        table_floorshape(R, P("tableD3_floorshape"))
        table_gmv(R, P("tableD4_gmv"))
    log(f"tables written to {out}")
    return R
