"""A synthetic panel in the folder layout of cmk.data, so that the whole pipeline runs without licensed data.

The panel has the layout of the main panel (94 characteristics in the order of Gu, Kelly and Xiu (2020), then the eight
Welch-Goyal predictors) with a persistent population of firms that enter and leave. Characteristics are rank-transformed
AR(1) processes with 10% missing values (set to zero). Returns follow a conditional factor model whose loadings are linear
in the characteristics plus one nonlinear term, so that the additive kernel has something to find:
    x_{t+1} = [1, z] b + [1, z] Lam g_{t+1} + 0.05 tanh(2 z_1) h_{t+1} + sqrt(u) eps,   g ~ N(0, I_3), h ~ N(0, 1).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .data import GKX_NAMES
from .panel.common import rank_unit, write_month, write_summary


def write_synthetic_panel(out: str, T: int = 140, nmin: int = 300, nmax: int = 450, r: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    os.makedirs(out, exist_ok=True)
    p = 1 + len(GKX_NAMES)
    b = rng.normal(size=p) * 0.002
    Lam = rng.normal(size=(p, r)) * 0.01
    u = 0.10 ** 2
    pool = np.arange(10000, 10000 + nmax * 3)
    latent = rng.normal(size=(len(pool), len(GKX_NAMES)))
    active = np.sort(rng.choice(len(pool), size=rng.integers(nmin, nmax + 1), replace=False))
    ends = pd.date_range("1990-01-31", periods=T + 1, freq="ME")
    summary = []
    for t in range(1, T + 1):
        active = np.setdiff1d(active, rng.choice(active, size=rng.integers(0, 10), replace=False))
        cand = np.setdiff1d(np.arange(len(pool)), active)
        active = np.union1d(active, rng.choice(cand, size=rng.integers(0, 12), replace=False))
        N = len(active)
        latent[active] = 0.9 * latent[active] + np.sqrt(1 - 0.81) * rng.normal(size=(N, len(GKX_NAMES)))
        raw = latent[active].copy()
        raw[rng.random(raw.shape) < 0.1] = np.nan
        Z = np.column_stack([rank_unit(raw[:, j]) for j in range(raw.shape[1])])
        Phi = np.hstack([np.ones((N, 1)), Z])
        g = rng.normal(size=r)
        x = Phi @ (b + Lam @ g) + 0.05 * np.tanh(2 * Z[:, 0]) * rng.normal() + np.sqrt(u) * rng.normal(size=N)
        rf = 0.003
        write_month(out, t, 1.0 + rf + x, Z, rng.random(8), pool[active],
                    pd.DataFrame(columns=["permno", "dlstcd", "dlret"]), 1.0 / (1.0 + rf))
        summary.append(dict(i=t, date=ends[t], t_end=ends[t - 1], t1_end=ends[t], names=N, common=N, kept=N, delisted=0,
                            dl_cause=0, crashes=int((x < -0.2).sum())))
    write_summary(out, pd.DataFrame(summary))
    np.savez(os.path.join(out, "truth.npz"), b=b, Lam=Lam, u=u)
