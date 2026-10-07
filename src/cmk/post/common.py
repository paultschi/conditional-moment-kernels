"""Statistics and table helpers shared by the tables, the figure and the quoted numbers.

Definitions (Section 7.2):
    R^2      = 1 - sum_t N_t^{-1} ||x - mu||^2 / sum_t N_t^{-1} ||x||^2           (pooled over the test months)
    R^{2,2}  = 1 - sum_t N_t^{-1} ||xx' - Q||_F^2 / sum_t N_t^{-1} ||xx' - u^bm I||_F^2
    score    = mean_t [S(x, 0, u^bm I) - S(x, mu, Sigma)]                         (the score differential)
    SR pred. = sqrt(12 thetabar / (1 - thetabar)), thetabar = mean_t theta_t / (1 + theta_t)    (Proposition 6.1)
    t-statistics: Newey-West with 12 lags; p-values of Sharpe ratio differences: Ledoit and Wolf (2008), HAC.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ..config import CRASHES


def nw_var_mean(x, lags=12):
    """Newey-West variance of the sample mean of x (Bartlett kernel)."""
    x = np.asarray(x, float)
    n = len(x)
    xc = x - x.mean()
    s = xc @ xc / n
    for k in range(1, lags + 1):
        s += 2.0 * (1.0 - k / (lags + 1.0)) * (xc[k:] @ xc[:-k]) / n
    return s / n


def tstat(x, lags=12):
    """Newey-West t-statistic of the mean of x (nan for a constant series, e.g. a model compared with itself)."""
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.nan
    v = nw_var_mean(x, lags)
    return x.mean() / np.sqrt(v) if v > 0 else np.nan


def sharpe_diff_test(a, b, lags=12):
    """Ledoit and Wolf (2008) test of equal Sharpe ratios with a HAC covariance: (difference, s.e., p), monthly units."""
    from scipy.stats import norm
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = len(a)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    Y = np.column_stack([a, b, a * a, b * b])
    Yc = Y - Y.mean(0)
    S = Yc.T @ Yc / n
    for k in range(1, lags + 1):
        G = Yc[k:].T @ Yc[:-k] / n
        S += (1.0 - k / (lags + 1.0)) * (G + G.T)
    ga, gb = va + ma ** 2, vb + mb ** 2
    grad = np.array([ga / (ga - ma ** 2) ** 1.5, -gb / (gb - mb ** 2) ** 1.5,
                     -0.5 * ma / (ga - ma ** 2) ** 1.5, 0.5 * mb / (gb - mb ** 2) ** 1.5])
    diff = ma / np.sqrt(va) - mb / np.sqrt(vb)
    se = np.sqrt(max(grad @ S @ grad / n, 0.0))
    p = 2 * (1 - norm.cdf(abs(diff) / se)) if se > 0 else np.nan
    return diff, se, p


def ols_nw(y, X, lags=12):
    """OLS with Newey-West standard errors: (beta, s.e., adjusted R^2, residuals)."""
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for j in range(1, lags + 1):
        G = Xe[j:].T @ Xe[:-j]
        S += (1.0 - j / (lags + 1.0)) * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    r2 = 1 - e @ e / ((y - y.mean()) @ (y - y.mean()))
    return beta, np.sqrt(np.diag(V)), 1 - (1 - r2) * (n - 1) / (n - k), e


def sr(x):
    """Annualized Sharpe ratio of monthly returns."""
    x = np.asarray(x, float)
    return np.sqrt(12) * x.mean() / x.std(ddof=1) if len(x) > 2 and x.std(ddof=1) > 0 else np.nan


def max_drawdown(x, vol=0.10):
    """Maximum drawdown (negative) of the monthly returns x in time order, rescaled to the annual volatility vol, compounded."""
    x = np.asarray(x, float)
    if len(x) < 3 or not x.std(ddof=1) > 0:
        return np.nan
    w = np.concatenate([[1.0], np.cumprod(1.0 + x * vol / np.sqrt(12) / x.std(ddof=1))])
    return float((w / np.maximum.accumulate(w) - 1.0).min())


def return_month(date0):
    """The return month (the month after the covariate date) as 'YYYY-MM'."""
    d = pd.to_datetime(pd.Series(date0)) + pd.offsets.MonthBegin(1)
    return d.dt.strftime("%Y-%m").values


def in_period(ym, start, end):
    ym = pd.Series(np.asarray(ym))
    if start == "crash":
        return np.any([((ym >= a) & (ym <= b)).values for a, b in CRASHES], axis=0)
    ok = np.ones(len(ym), bool)
    if start:
        ok &= (ym >= start).values
    if end:
        ok &= (ym <= end).values
    return ok


def load_run(work, name, refit=None):
    """A run of <work>/runs: refit=True the re-estimated rows, False the training-window rows, None the monthly file."""
    f = os.path.join(work, "runs", f"{name}_{'refit_' if refit else ''}monthly.csv.gz")
    d = pd.read_csv(f)
    if refit is not None and "refit" in d.columns:
        d = d[d.refit == (1 if refit else 0)]
    if "nfac_ex" in d.columns:
        d = d.assign(rank_C=d.nfac_ex)
    d["ym"] = return_month(d["date"])
    return d


def exists(work, name, refit=None):
    return os.path.exists(os.path.join(work, "runs", f"{name}_{'refit_' if refit else ''}monthly.csv.gz"))


def pick(d, crit, by):
    """Rows minimizing `crit` within the groups `by` (ties: the smallest m, then lambda)."""
    d = d.sort_values(by + ["m", "lam"])
    return d.loc[d.groupby(by)[crit].idxmin()]


def metrics(g):
    thetabar = g.theta_frac.mean()
    return dict(months=len(g), R2=1 - g.r2_num.sum() / g.r2_den.sum(), R22=1 - g.r22_num.sum() / g.r22_den.sum(),
                Score=(g.score_bm - g.score).mean(), TotR2=g.tot_r2.mean(), rho_f=g.rho_f.mean(),
                nfac=g.rank_C.mean(), u_id=g.u_id.mean(), SR_pred=np.sqrt(12 * thetabar / (1 - thetabar)),
                SR_umve=sr(g.ret_sdf), SR_cmve=sr(g.ret_cmve), SR_ew=sr(g.ret_ew),
                t_NW_umve=g.ret_sdf.mean() / np.sqrt(nw_var_mean(g.ret_sdf.values, 12)))


def stat(g):
    m = metrics(g)
    return dict(nfac=m["nfac"], R2=100 * m["R2"], R22=100 * m["R22"], Score=m["Score"], TotR2=100 * m["TotR2"],
                rho=100 * m["rho_f"], u=1e4 * m["u_id"], SR_umve=m["SR_umve"], SR_cmve=m["SR_cmve"], SR_ew=m["SR_ew"],
                SR_pred=m["SR_pred"], t_NW=m["t_NW_umve"], months=len(g), MDD=100 * max_drawdown(g.sort_values("t").ret_sdf))


def paired(g, ref):
    """Paired comparison of g with ref (both indexed by t): score difference with its NW t, and the LW p-value."""
    r = ref.reindex(g.index)
    d = ((g.score_bm - g.score) - (r.score_bm - r.score)).values
    return dict(d_score=d.mean(), t_score=tstat(d), p_LW=sharpe_diff_test(g.ret_sdf.values, r.ret_sdf.values, 12)[2])


def ff5_regression(ret, ym, ff5, lags=12):
    """Regression of the return, scaled to 10% annual volatility, on the five Fama-French factors."""
    F = ff5.reindex(ym)
    ok = ~F["Mkt-RF"].isna().values
    y = np.asarray(ret, float)[ok]
    y = y / (np.sqrt(12) * y.std(ddof=1)) * 0.10
    X = np.column_stack([np.ones(ok.sum()), F.loc[ok, ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]].values])
    beta, se, adj, e = ols_nw(y, X, lags)
    return dict(alpha=100 * 12 * beta[0], t_alpha=beta[0] / se[0], adjR2=100 * adj)


def fmt(v, nd=3):
    if isinstance(v, str):
        return v
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return f"{v:.{nd}f}"


def tex_table(rows, cols, headers, nd, path, align=None):
    """A booktabs tabular; None in rows is a \\midrule."""
    align = align or ("l" + "r" * (len(cols) - 1))
    lines = [r"\begin{tabular}{" + align + "}", r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
    for r in rows:
        if r is None:
            lines.append(r"\midrule")
            continue
        lines.append(" & ".join(fmt(r.get(c, np.nan), n) for c, n in zip(cols, nd)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
