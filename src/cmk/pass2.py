"""Pass 2 over the raw panel: the portfolio weights of the selected models, their turnover and composition (Section 7.4).

For every test month the selected model of a run (the hyper-parameters of its `selected` refit row) is re-estimated on the
108 months before the test month, exactly as in the run, and evaluated on the stocks of the month: the moments
mu = phi b, Sigma = phi C phi' + u I, the UMVE weights Q^{-1} mu and the CMVE weights Sigma^{-1} mu (Proposition 6.1), their
returns, gross exposure, long and short positions, the largest weight and the turnover against the previous month
(union of the two cross sections, weights at unit gross exposure as well). The score recomputed from the raw panel agrees
with the run (a built-in check). The selected additive specification changes (bandwidth, m) from month to month; its
pass chains the weights across these switches, so that its turnover contains the trades induced by the switches.

Output: <work>/pass2/<run>_pass.csv.gz (one row per test month and m), ipca_pass.csv.gz, switch_pass.csv.gz.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .data import excess_returns, load_month, read_datefile
from .features import additive_features, linear_features
from .jobs import STATS_FILES

RUNS = {"cmk_linear": "linear", "cmk_gauss_l1": "gauss_l1", "cmk_gauss_l0.5": "gauss_l0.5"}
SEL_RUN = {"linear": "cmk_linear", "l1": "cmk_gauss_l1", "l0.5": "cmk_gauss_l0.5"}


def evaluate_raw(x, phi, b, V, u, s2_bm):
    """Exact evaluation of the isotropic model on the cross section (phi = Phi A); returns (dict, UMVE weights, CMVE weights)."""
    N, m = phi.shape
    I = np.eye(m)
    C = V - np.outer(b, b)
    C = 0.5 * (C + C.T)
    mu = phi @ b
    r = x - mu
    x2 = float(x @ x)
    w = 1.0 / N
    M = phi.T @ phi
    M = 0.5 * (M + M.T)
    B = M / u                                                    # phi' D^-1 phi with D = u I
    K_C = np.linalg.solve(I + C @ B, C)
    K_C = 0.5 * (K_C + K_C.T)

    def sig_inv(y):
        Dy = y / u
        return Dy - (phi / u) @ (K_C @ (phi.T @ Dy))

    sign, ld = np.linalg.slogdet(I + B @ C)
    logdet = float(N * np.log(u) + ld)
    score = logdet + float(r @ sig_inv(r))
    Mp = np.linalg.pinv(M, rcond=1e-10, hermitian=True)
    f = Mp @ (phi.T @ x)
    Sm = sig_inv(mu)
    theta = float(mu @ Sm)
    w_cmve = Sm
    w_sdf = Sm / (1.0 + theta)
    out = dict(N=N, r2_num=w * float(r @ r), r2_den=w * x2, score=score,
               score_bm=N * np.log(s2_bm) + x2 / s2_bm, tot_r2=1.0 - float(np.sum((x - phi @ f) ** 2)) / x2,
               theta=theta, ret_sdf=float(w_sdf @ x), ret_cmve=float(w_cmve @ x), ret_ew=float(x.mean()))
    return out, w_sdf, w_cmve


def port_stats(wv, x, prev, perm):
    """Gross exposure, positions, largest weight, return at unit gross exposure and turnover against `prev`."""
    lev = float(np.abs(wv).sum())
    levs = lev if lev > 0 else 1.0
    rec = dict(lev=lev, long=float(wv[wv > 0].sum()), short=float(-wv[wv < 0].sum()), nlong=int((wv > 0).sum()),
               nshort=int((wv < 0).sum()), maxw=float(np.abs(wv).max()), ret_unit=float(wv @ x) / levs)
    if prev is not None:
        p_perm, p_w, p_lev = prev
        idx = pd.Index(perm).union(pd.Index(p_perm))
        cur = pd.Series(wv, index=perm).reindex(idx, fill_value=0.0)
        old = pd.Series(p_w, index=p_perm).reindex(idx, fill_value=0.0)
        rec["turnover"] = float(np.abs(cur - old).sum())
        rec["turnover_unit"] = float(np.abs(cur / levs - old / p_lev).sum())
    else:
        rec["turnover"] = np.nan
        rec["turnover_unit"] = np.nan
    return rec, (perm, wv, levs)


def _features_fn(work, key):
    if key == "linear":
        return linear_features
    z = np.load(os.path.join(work, "stats", STATS_FILES[key]))
    zeta, B, ell2 = z["landmarks"], z["B"], float(z["ell2"])
    return lambda Z: additive_features(Z, zeta, B, ell2)


def _model_source(work, key):
    """A function (t_index, m, eta, theta) -> (A, P, s2) re-estimating the CMK estimator on the statistics."""
    from .runs import refit_model
    from .stats import Stats
    st = Stats(os.path.join(work, "stats", STATS_FILES[key]))
    return st, (lambda k, m, eta, theta: refit_model(st, k, m, eta, theta))


def _pass_task(work, panel, name, plan, label):
    """plan: list of (t, m, kind, spec) in time order; kind 'cmk' with spec (stats key, eta, theta) or 'ipca' with spec
    (A, b, V, u, s2). Weights are chained across consecutive rows of the plan."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    tt, d0, _ = read_datefile(panel)
    date_of = dict(zip(tt.tolist(), d0.tolist()))
    feats, sources = {}, {}
    prev = {"sdf": None, "cmve": None}
    rows = []
    for t, m, kind, spec in plan:
        dat = load_month(panel, t)
        x = excess_returns(dat["R"], dat["bond"])
        perm = dat["permno"]
        if kind == "cmk":
            key, eta, theta = spec
            if key not in sources:
                sources[key] = _model_source(work, key)
                feats[key] = _features_fn(work, key)
            st, src = sources[key]
            k = int(np.where(st.t == t)[0][0])
            A, P, s2 = src(k, m, eta, theta)
            phi = feats[key](dat["Z"]) @ A
            b, V, u = P.b, P.V, P.u
        else:
            A, b, V, u, s2 = spec
            phi = linear_features(dat["Z"]) @ A
        ev, w_sdf, w_cmve = evaluate_raw(x, phi, b, V, u, s2)
        rec = dict(t=t, date=date_of.get(t, ""), name=label, m=m, **ev)
        for pf, wv in (("sdf", w_sdf), ("cmve", w_cmve)):
            ps, prev[pf] = port_stats(wv, x, prev[pf], perm)
            rec.update({f"{pf}_{k2}": v for k2, v in ps.items()})
        rows.append(rec)
    return name, pd.DataFrame(rows)


def selection(work, cfg):
    """The selected (bandwidth, m) of every test month: the lowest validation score among the selected refit rows of the
    two additive runs and the linear run at its full feature space (Table 1, 'linear + Gaussian, (l,m) selected')."""
    parts = []
    for run, label in (("cmk_gauss_l0.5", "l0.5"), ("cmk_gauss_l1", "l1"), ("cmk_linear", "linear")):
        d = pd.read_csv(os.path.join(work, "runs", f"{run}_refit_monthly.csv.gz"))
        d = d[(d.selected == 1) & (d.refit == 1)]
        if label == "linear":
            d = d[d.m == cfg.MF]
        parts.append(d.assign(run=label))
    allg = pd.concat(parts, ignore_index=True).sort_values(["t", "run", "m"])
    return allg.loc[allg.groupby("t")["val_score"].idxmin()]


def run_pass2(work, panel, cfg, workers=16, log=print):
    out = os.path.join(work, "pass2")
    os.makedirs(out, exist_ok=True)
    tasks = []
    for run, key in RUNS.items():
        d = pd.read_csv(os.path.join(work, "runs", f"{run}_refit_monthly.csv.gz"))
        d = d[(d.selected == 1) & (d.refit == 1)].sort_values("t")
        for m in sorted(d.m.unique()):
            g = d[d.m == m]
            plan = [(int(r.t), int(m), "cmk", (key, float(r.gamma), float(r.thr))) for r in g.itertuples()]
            tasks.append((f"{run}.m{m}", plan, "refit_sel"))
    for K in cfg.ipca_pass2:
        z = np.load(os.path.join(work, "runs", f"ipca_K{K}_models.npz"))
        ts = sorted(int(k.split("_t")[1].split("_")[0]) for k in z.files if k.startswith("sel_") and k.endswith("_b"))
        plan = [(t, K, "ipca", (z[f"A_m{K}_t{t}"].astype(np.float64), z[f"sel_m{K}_t{t}_b"], z[f"sel_m{K}_t{t}_V"],
                                float(z[f"sel_m{K}_t{t}_u"][0]), float(z[f"s2_t{t}"][0]))) for t in ts]
        tasks.append((f"ipca.K{K}", plan, "refit_sel"))
    sel = selection(work, cfg)
    sel.to_csv(os.path.join(out, "selection.csv"), index=False)
    refit = {r: pd.read_csv(os.path.join(work, "runs", f"{SEL_RUN[r]}_refit_monthly.csv.gz")) for r in SEL_RUN}
    plan = []
    for r in sel.sort_values("t").itertuples():
        g = refit[r.run]
        row = g[(g.t == r.t) & (g.m == r.m) & (g.selected == 1)].iloc[0]
        plan.append((int(r.t), int(r.m), "cmk", (RUNS[SEL_RUN[r.run]], float(row.gamma), float(row.thr))))
    tasks.append(("switch", plan, "switch"))
    log(f"pass 2: {len(tasks)} tasks")
    t0 = time.time()
    res = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_pass_task, work, panel, name, plan, label) for name, plan, label in tasks]
        for f in futs:
            name, df = f.result()
            res[name] = df
            log(f"  {name}: {len(df)} months, {time.time() - t0:.0f}s")
    for run in RUNS:
        df = pd.concat([v.assign(run=run) for k, v in res.items() if k.startswith(run + ".")], ignore_index=True)
        df.sort_values(["t", "m"], kind="stable").to_csv(os.path.join(out, f"{run}_pass.csv.gz"), index=False)
        chk = df.merge(pd.read_csv(os.path.join(work, "runs", f"{run}_refit_monthly.csv.gz")).query("selected == 1"),
                       on=["t", "m"], suffixes=("", "_run"))
        log(f"{run}: max |score - score of the run| = {np.abs(chk.score - chk.score_run).max():.2e} over {len(chk)} months")
    ip = pd.concat([v for k, v in res.items() if k.startswith("ipca.")], ignore_index=True)
    ip.sort_values(["t", "m"], kind="stable").to_csv(os.path.join(out, "ipca_pass.csv.gz"), index=False)
    sw = res["switch"].merge(sel[["t", "run", "m"]], on=["t", "m"])
    sw.sort_values("t").to_csv(os.path.join(out, "switch_pass.csv.gz"), index=False)
