"""Figure 1: rolling out-of-sample performance over 60-month windows (UMVE Sharpe ratio, score differential, score
difference to IPCA with K = 60) of the selected additive kernel, the linear kernel at the full feature space and IPCA."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CRASHES


def rolling_figure(R, out: str, K: int = 60, win: int = 60):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = R.gsel.set_index("t").sort_index()
    lf = R.lin_ref
    ip = R.ip[R.ip.m == K].set_index("t").sort_index()
    t = g.index.intersection(lf.index).intersection(ip.index)
    g, lf, ip = g.loc[t], lf.loc[t], ip.loc[t]
    dates = pd.to_datetime(g.ym.values) + pd.offsets.MonthBegin(1)
    roll_sr = lambda x: pd.Series(x).rolling(win).apply(lambda v: np.sqrt(12) * v.mean() / v.std(ddof=1)).values
    roll_mean = lambda x: pd.Series(x).rolling(win).mean().values
    series = [("CMK, linear + Gaussian (selected)", g, "#2a78d6"), (f"CMK, linear, $m={R.cfg.MF}$", lf, "#1baf7a"),
              (f"IPCA, $K={K}$", ip, "#eb6834")]
    fig, ax = plt.subplots(3, 1, figsize=(7.2, 8.4), sharex=True)
    for lab, d, col in series:
        ax[0].plot(dates, roll_sr(d.ret_sdf.values), color=col, lw=1.4, label=lab)
        ax[1].plot(dates, roll_mean((d.score_bm - d.score).values), color=col, lw=1.4)
    ax[0].plot(dates, roll_sr(g.ret_ew.values), color="#6b7280", lw=1.0, ls="--", label="$1/N_t$")
    for lab, d, col in series[:2]:
        ax[2].plot(dates, roll_mean(((d.score_bm - d.score) - (ip.score_bm - ip.score)).values), color=col, lw=1.4,
                   label=lab.replace("CMK, ", "") + " minus IPCA")
    ax[0].set_ylabel("Sharpe ratio, UMVE portfolio")
    ax[1].set_ylabel("score differential")
    ax[2].set_ylabel("score difference to IPCA")
    ax[0].legend(frameon=False, fontsize=8, ncol=2, loc="lower left", bbox_to_anchor=(0.0, 1.0))
    ax[2].legend(frameon=False, fontsize=8, loc="upper left")
    for axx in ax:
        axx.spines[["top", "right"]].set_visible(False)
        axx.grid(axis="y", color="#e5e7eb", lw=0.6)
        axx.axhline(0, color="#9ca3af", lw=0.6)
        for a0, b0 in CRASHES:
            axx.axvspan(pd.Timestamp(a0), pd.Timestamp(b0), color="#f3f4f6", zorder=0)
    ax[2].set_xlabel(f"end of the {win}-month rolling window (shaded: crash episodes)")
    fig.tight_layout()
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=130)
    plt.close(fig)
