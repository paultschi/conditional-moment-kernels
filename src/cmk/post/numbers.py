"""The numbers quoted in the text of Section 7 and of Online Appendix D that are not in a table.

Each line starts with the section of the paper it belongs to. Written to <out>/numbers.txt by `cmk tables`.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .common import in_period, sharpe_diff_test, sr, tstat


def _pd(a, b):
    """Paired score difference a - b (indexed by t) and its NW t-statistic."""
    a = a.set_index("t") if "t" in a.columns else a
    b = b.set_index("t") if "t" in b.columns else b
    ix = a.index.intersection(b.index)
    d = ((a.score_bm - a.score) - (b.score_bm - b.score)).loc[ix].values
    return (d.mean() if len(d) else np.nan), tstat(d), len(ix)


def panel_numbers(work, cfg, P):
    """Section 7.1 and Online Appendix D: the panel, the riskless rate, the bandwidth, the ranks of the feature matrices."""
    from ..stats import Stats
    z = np.load(os.path.join(work, "stats", "stats_linear.npz"))
    N, d0 = z["N"], z["date0"]
    test = z["t"] >= cfg.first_test
    k = int(np.argmax(N))
    P(f"7.1  panel: {len(N)} months, N_t from {int(N.min())} to {int(N.max())} ({str(d0[k])[:4]}), mean {N.mean():.0f}, "
      f"mean in the test months {N[test].mean():.0f}; {int(N.sum())} stock-months")
    for bw in ("1", "0.5"):
        f = os.path.join(work, "stats", f"stats_gauss_l{bw}.npz")
        if os.path.exists(f):
            g = np.load(f)
            P(f"7.1  additive kernel, bandwidth {bw} x ell_0: ell_0^2 = {float(g['ell2_median']):.1f} "
              f"({float(g['ell2_median']) / (int(g['d_lin']) - 1):.2f} per characteristic), the landmarks capture "
              f"{100 * float(g['nystrom_share']):.0f}% of the trace of the Gaussian kernel matrix of the landmark pool and "
              f"{100 * (1 - float(np.mean(g['nres']))):.0f}% averaged over the months of the panel")
    st = Stats(os.path.join(work, "stats", "stats_linear.npz"))
    ranks = np.array([int((ev > 1e-9 * ev.max()).sum()) for ev in np.linalg.eigvalsh(st.G)])
    ym = np.array([str(x)[:7] for x in st.date1])
    full = st.d
    first_full = ym[np.argmax(np.array([np.all(ranks[i:] == full) for i in range(len(ranks))]))]
    early = ranks[ym < first_full]
    P(f"7.1  rank of [1, z_t]: {full} in every return month from {first_full} on"
      + (f", {early.min()} to {early.max()} before" if len(early) else ""))
    low = []
    for t0 in range(0, st.T - 108 + 1):
        ev = np.linalg.eigvalsh(st.G[t0:t0 + 108].sum(0))
        r = int((ev > 1e-9 * ev.max()).sum())
        if r < full:
            low.append((t0, r))
    if low:
        P(f"D.1  pooled Gram matrix of the 108-month windows: rank {min(r for _, r in low)} to {max(r for _, r in low)} in the "
          f"{len(low)} windows up to the one ending {str(st.date0[low[-1][0] + 107])[:7]}, {full} afterwards")


def selection_numbers(R, P):
    """Sections 7.2 and 7.4: hyper-parameter selection, positive months, composition of the portfolios."""
    cfg = R.cfg
    MF = cfg.MF
    for lab, d_all, d in [("linear", R.lin_all, R.lin), ("additive l0", R.g1_all, R.g1), ("additive l0/2", R.g05_all, R.g05)]:
        for m in sorted(d.m.unique()):
            g = d[d.m == m]
            g00 = d_all[(d_all.m == m) & (d_all.gamma == 0) & (d_all.thr == 0)]
            P(f"7.2  {lab} m = {m}: eta > 0 in {100 * (g.gamma > 0).mean():.0f}% of the windows, lambda > 0 in "
              f"{100 * (g.thr > 0).mean():.0f}%, penalty at the top of its grid in {100 * (g.thr == g.thr.max()).mean():.0f}%; "
              f"factors {g.rank_C.mean():.1f} (without regularizers {g00.rank_C.mean():.1f})")
    d = R.gsel.score_bm - R.gsel.score
    yearly = d.groupby(R.gsel.ym.str[:4]).mean()
    P(f"7.2  selected additive kernel: score differential positive in {100 * (d > 0).mean():.0f}% of the {len(d)} test months "
      f"and in {int((yearly > 0).sum())} of {len(yearly)} calendar years")
    P("D.1  (bandwidth, m) selection: " + ", ".join(f"{k}: {100 * v / len(R.gsel):.0f}%" for k, v in R.gsel.groupby(["run", "m"]).size().items()))
    lj = R.lin_joint
    P("7.2  linear kernel, m selected: " + ", ".join(f"m = {int(k)} in {100 * v / len(lj):.0f}%" for k, v in lj.groupby("m").size().items()))
    g0 = R.lin_all[(R.lin_all.m == MF) & (R.lin_all.gamma == 0) & (R.lin_all.thr == 0)]
    dd, t, _ = _pd(R.lin_sel[MF], g0)
    P(f"7.2  linear m = {MF}: regularizers add {dd:.1f} points (t = {t:.1f})")
    for lab, g in [(f"linear m = {MF}", R.lin_sel[MF]), ("additive selected", R.gsel)]:
        P(f"7.2  {lab}: systematic share of the conditional variance "
          f"{100 * _sys_share(g).mean():.0f}%, rho_f {100 * g.rho_f.mean():.0f}%")
    # R^{2,2} with the weight N_t^{-2}
    for lab, g in [(f"linear m = {MF}", R.lin_sel[MF]), ("additive selected", R.gsel)] + \
            [(f"IPCA K = {K}", R.ip[R.ip.m == K]) for K in R.Ks]:
        r22 = 100 * (1 - g.r22_num.sum() / g.r22_den.sum())
        r22n2 = 100 * (1 - (g.r22_num / g.N).sum() / (g.r22_den / g.N).sum())
        P(f"D.1  R^(2,2) of {lab}: {r22:.2f}% with the weight 1/N_t, {r22n2:.2f}% with 1/N_t^2")
    # pass 2: composition
    from .tables import pass2_frames
    p2, sw = pass2_frames(R)
    if len(p2):
        ps = p2.merge(R.gsel[["t", "run", "m"]], on=["t", "run", "m"])
        P(f"7.4  UMVE portfolio of the selected additive kernel: {ps.sdf_nlong.mean():.0f} long and {ps.sdf_nshort.mean():.0f} short "
          f"positions on average, largest weight {ps.sdf_maxw.mean():.2f} of the gross exposure")
        if sw is not None:
            nsw = int((sw[["run", "m"]].shift() != sw[["run", "m"]]).any(axis=1).sum()) - 1
            P(f"7.4  selected specification: {nsw} switches between models in {len(sw)} test months; turnover "
              f"{sw.sdf_turnover_unit.mean():.2f} with the switches, {ps.sdf_turnover_unit.mean():.2f} within the models")


def _sys_share(g):
    a = g.u_id * (g.rho_f * g.N - g.rank_phi) / (1 - g.rho_f)
    return a / (a + g.u_id * g.N)


def ipca_numbers(R, P):
    """Section 7.3: the linear kernel and the selected additive kernel against IPCA."""
    MF = R.cfg.MF
    lf = R.lin_ref
    for K in R.Ks:
        g = R.ip[R.ip.m == K].set_index("t").sort_index()
        d, t, _ = _pd(lf, g)
        d2, t2, _ = _pd(R.gsel_i, g)
        ix = lf.index.intersection(g.index)
        p = sharpe_diff_test(lf.ret_sdf.loc[ix].values, g.ret_sdf.loc[ix].values)[2]
        p2 = sharpe_diff_test(R.gsel_i.ret_sdf.loc[ix].values, g.ret_sdf.loc[ix].values)[2]
        P(f"7.3  IPCA K = {K}: score {(g.score_bm - g.score).mean():.0f}, SR {sr(g.ret_sdf):.2f}; linear m = {MF} minus IPCA "
          f"{d:.1f} (t = {t:.1f}), p(SR) = {p:.2f}; selected additive minus IPCA {d2:.1f} (t = {t2:.1f}), p(SR) = {p2:.2f}")
    ms = R.load("moment_step_linear")
    if ms is not None:
        ms = ms[ms.m == MF].set_index("t")
        i60 = R.ip[R.ip.m == 60].set_index("t")
        for lab, f in (("before 1985", lambda d: d[d.ym < "1985"]), ("from 1985", lambda d: d[d.ym >= "1985"])):
            d, t, n = _pd(f(ms), f(i60 if 95 not in R.Ks else R.ip[R.ip.m == 95].set_index("t")))
            if not n:
                continue
            P(f"7.3  moment step in the full space minus IPCA K = {95 if 95 in R.Ks else 60}, {lab}: {d:.1f} (t = {t:.1f}, {n} months)")
    for K in (20, 60):
        if K in R.Ks:
            g = R.ip[R.ip.m == K]
            for pn, a, b in R.cfg.periods[1:]:
                gg = g[in_period(g.ym, a, b)].set_index("t")
                s = R.gsel[in_period(R.gsel.ym, a, b)].set_index("t")
                d, t, n = _pd(s, gg)
                if not n:
                    continue
                P(f"7.3  selected additive minus IPCA K = {K}, {pn}: {d:.1f} (t = {t:.1f}, {n} months)")


def nw_lag_numbers(R, P, lags=(12, 24, 36)):
    """Online Appendix D.1, Newey-West lags."""
    MF, MA = R.cfg.MF, R.cfg.MA
    specs = {f"linear m={m}": R.lin_sel[m].set_index("t") for m in R.cfg.m_lin}
    specs[f"additive l0 m={MA}"] = R.g1[R.g1.m == MA].set_index("t")
    specs["additive selected"] = R.gsel_i
    for K in (20, 40, 60):
        if K in R.Ks:
            specs[f"IPCA K={K}"] = R.ip[R.ip.m == K].set_index("t")
    ref_c, ref_g = specs[f"linear m={MF}"], specs["additive selected"]
    dif = lambda g, r: ((g.score_bm - g.score) - (r.score_bm - r.score).reindex(g.index)).values
    for name, g in specs.items():
        line = f"D.1  NW lags {lags}: {name}: "
        if name != f"linear m={MF}":
            line += "vs linear t = " + "/".join(f"{tstat(dif(g, ref_c), l):.1f}" for l in lags) + "; "
        if name != "additive selected":
            line += "vs additive selected t = " + "/".join(f"{tstat(dif(g, ref_g), l):.1f}" for l in lags) + "; "
        line += "mean UMVE return t = " + "/".join(f"{tstat(g.ret_sdf.values, l):.1f}" for l in lags)
        P(line)
    d = dif(ref_g, specs["IPCA K=60"])
    P(f"D.1  NW lags {lags}: selected additive vs IPCA K=60: d = {d.mean():.1f}, t = " + "/".join(f"{tstat(d, l):.1f}" for l in lags))
    ym = ref_g.ym
    for pn, a, b in R.cfg.periods[1:]:
        ok = in_period(ym, a, b)
        d = dif(ref_g, ref_c)[ok]
        if len(d) < 2:
            continue
        P(f"D.1  NW lags {lags}: {pn}, selected additive vs linear m={MF}: d = {d.mean():.1f}, t = "
          + "/".join(f"{tstat(d, l):.1f}" for l in lags))


def robustness_numbers(R, P):
    """Online Appendix D.1: the extended penalty grid, the one-step estimator, the idiosyncratic specification."""
    MF, MA = R.cfg.MF, R.cfg.MA
    for bw, base_all in (("l1", R.g1_all), ("l0.5", R.g05_all)):
        ext = R.load(f"ext_gauss_{bw}")
        if ext is None:
            continue
        base = base_all[(base_all.m == MA) & base_all.t.isin(ext.t)]
        ext = ext[ext.m == MA]
        ref = base[base.selected == 1].set_index("t").sort_index()
        un = pd.concat([base.assign(src="grid"), ext[ext.thr > 3e-3].assign(src="ext")], ignore_index=True)
        new = un.loc[un.groupby("t")["val_score"].idxmin()].set_index("t").sort_index()
        d = (new.score_bm - new.score) - (ref.score_bm - ref.score)
        p = sharpe_diff_test(new.ret_sdf.values, ref.ret_sdf.values)[2]
        P(f"D.1  extended penalty grid, {bw}, m = {MA}, {len(ref)} windows: added values selected in {100 * (new.src == 'ext').mean():.0f}%, "
          f"factors {ref.rank_C.mean():.0f} -> {new.rank_C.mean():.0f}, score {d.mean():+.1f} (t = {tstat(d.values):.1f}), "
          f"SR {sr(ref.ret_sdf):.2f} -> {sr(new.ret_sdf):.2f} (p = {p:.2f})")
    jt = R.load("onestep_linear_mf")
    if jt is not None:
        jt = jt[jt.m == MF].set_index("t")
        cf = R.lin_all[(R.lin_all.m == MF) & (R.lin_all.gamma == 1) & (R.lin_all.thr == 0)].set_index("t").reindex(jt.index)
        for lab, msk in [("before 1985", jt.ym < "1985"), ("from 1985", jt.ym >= "1985")]:
            j, c = jt[msk], cf[msk]
            if not len(j):
                continue
            ds = (j.score_bm - j.score) - (c.score_bm - c.score)
            P(f"D.1  one-step minus CMK estimator, eta = 1, lambda = 0, {lab} ({len(j)} months): median score difference "
              f"{ds.median():.1f}, u^id ratio {np.median(j.u_id / c.u_id):.5f}")
    idm, iso = R.load("idio_linear", None), R.load("frobpath_linear", None)
    if idm is not None and iso is not None:
        iso = iso[iso.lam == 0]
        g = idm[(idm["mode"] == "lam0") & (idm.m == 20)].set_index("t").sort_index()
        h = iso[iso.m == 20].set_index("t").reindex(g.index)
        r22 = lambda x: 100 * (1 - x.r22_num.sum() / x.r22_den.sum())
        r2 = lambda x: 100 * (1 - x.r2_num.sum() / x.r2_den.sum())
        diff = ((h.r22_num - g.r22_num) / h.r22_den.mean()).values
        U = g[[c for c in g.columns if c.startswith("Uid")]].mean()
        var = lambda s_, v: (U["Uid_1_1"] + 2 * U["Uid_1_mvel1"] * s_ + 2 * U["Uid_1_idiovol"] * v + U["Uid_mvel1_mvel1"] * s_ ** 2
                             + 2 * U["Uid_mvel1_idiovol"] * s_ * v + U["Uid_idiovol_idiovol"] * v ** 2)
        rk = g.rank_id.value_counts(normalize=True)
        P(f"D.1  idiosyncratic specification (m = 20, lambda = 0, {len(g)} months): monthly idiosyncratic volatility "
          f"{100 * np.sqrt(var(0, 0)):.0f}% (median stock), {100 * np.sqrt(var(1, -1)):.0f}% (largest, lowest past volatility), "
          f"{100 * np.sqrt(var(-1, 1)):.0f}% (smallest, highest); rank of U^id two in {100 * rk.get(2, 0):.0f}% and three in "
          f"{100 * rk.get(3, 0):.0f}% of the windows; R^(2,2) {r22(h):.2f}% -> {r22(g):.2f}% (paired t = {tstat(diff):.1f}); "
          f"R^2 {r2(h):.2f}% -> {r2(g):.2f}%")


def tikhonov_numbers(R, P):
    """Online Appendix D.1, "Tikhonov regularization": coordinates without the regularization (pseudoinverse, and the
    truncation at the level of the regularization) against the CMK estimator."""
    MF, MA = R.cfg.MF, R.cfg.MA
    for name, ref_all, m, lab in [("pinv_linear", R.lin_all, MF, "pseudoinverse"), ("trunc_linear", R.lin_all, MF, "truncation"),
                                  ("pinv_gauss_l1", R.g1_all, MA, "pseudoinverse"), ("trunc_gauss_l1", R.g1_all, MA, "truncation")]:
        a = R.load(name)
        if a is None:
            continue
        a = a[(a.m == m) & (a.selected == 1)].set_index("t")
        b = ref_all[(ref_all.m == m) & (ref_all.selected == 1)].set_index("t").reindex(a.index)
        for per, msk in (("all", np.ones(len(a), bool)), ("before 1985", (a.ym < "1985").values), ("from 1985", (a.ym >= "1985").values)):
            aa, bb = a[msk], b[msk]
            if not len(aa):
                continue
            d = ((aa.score_bm - aa.score) - (bb.score_bm - bb.score)).values
            p = sharpe_diff_test(aa.ret_sdf.values, bb.ret_sdf.values)[2]
            P(f"D.1  {lab}, {name.split('_', 1)[1]} m = {m}, {per} ({len(aa)} months): score {(aa.score_bm - aa.score).mean():.1f} vs "
              f"{(bb.score_bm - bb.score).mean():.1f} with Tikhonov, difference {d.mean():.1f} (t = {tstat(d):.1f}); SR "
              f"{sr(aa.ret_sdf):.2f} vs {sr(bb.ret_sdf):.2f} (p = {p:.2f}); R^(2,2) {100 * (1 - aa.r22_num.sum() / aa.r22_den.sum()):.2f}%")


def holdout_numbers(R, P, start="2022-01", end="2025-12"):
    """Online Appendix D.2: the test months after the end of the main panel."""
    MF = R.cfg.MF
    cut = lambda d: d[in_period(d.ym, start, end)].set_index("t").sort_index()
    sel, lf = cut(R.gsel), cut(R.lin_sel[MF])
    i60 = cut(R.ip[R.ip.m == 60])
    if not len(sel):
        return
    d1, t1, n = _pd(sel, lf)
    d2, t2, _ = _pd(lf, i60)
    P(f"D.2  return months {start} to {end} ({n} months): additive minus linear {d1:.1f} (t = {t1:.1f}); linear minus IPCA K = 60 "
      f"{d2:.1f} (t = {t2:.1f}); SR {sr(sel.ret_sdf):.2f} (selected additive), {sr(lf.ret_sdf):.2f} (linear), "
      f"{sr(i60.ret_sdf):.2f} (IPCA K = 60, p = {sharpe_diff_test(lf.ret_sdf.values, i60.ret_sdf.values)[2]:.2f} against linear); "
      f"R^2 " + ", ".join(f"{100 * (1 - g.r2_num.sum() / g.r2_den.sum()):.2f}%" for g in (sel, lf, i60)))


def gmv_numbers(R, P):
    from .tables import gmv_summary
    t = gmv_summary(R)
    if t is None:
        return
    f = t[t["sample"] == "FULL SAMPLE"].set_index("model")
    MF = R.cfg.MF
    if "IPCA K=60" in f.index:
        P(f"7.3  score split, linear m = {MF} minus IPCA K = 60: covariance part {-f.loc['IPCA K=60', 'dcov_lin95']:+.1f} "
          f"(t = {-f.loc['IPCA K=60', 't_dcov_lin95']:.1f}), mean part {-f.loc['IPCA K=60', 'dmean_lin95']:+.1f} "
          f"(t = {-f.loc['IPCA K=60', 't_dmean_lin95']:.1f})")
    s = f.loc["additive (l,m) selected"]
    P(f"7.3  Gaussian features: covariance part {s.dcov_lin95:+.1f} (t = {s.t_dcov_lin95:.1f}), mean part {s.dmean_lin95:+.1f} "
      f"(t = {s.t_dmean_lin95:.1f}); GMV volatility {s.gmv_sd:.1f}% against {f.loc[f'linear m={MF}', 'gmv_sd']:.1f}%")


def write_numbers(R, out, work):
    lines = []
    P = lambda s: lines.append(s)
    P(f"# numbers quoted in the text, panel {R.cfg.name} ({len(R.common)} test months)")
    for fn in (lambda: panel_numbers(work, R.cfg, P), lambda: selection_numbers(R, P), lambda: ipca_numbers(R, P),
               lambda: gmv_numbers(R, P), lambda: nw_lag_numbers(R, P), lambda: robustness_numbers(R, P),
               lambda: tikhonov_numbers(R, P), lambda: holdout_numbers(R, P)):
        try:
            fn()
        except FileNotFoundError as e:          # a run that was not computed: its numbers are left out
            P(f"# skipped: {e}")
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return lines
