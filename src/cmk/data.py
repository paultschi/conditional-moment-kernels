"""Reading a monthly stock panel in the folder layout written by `cmk.panel` (and by the original R scripts).

Folder layout, one set of files per month t = 1..T:

    out_<t>.txt       N_t rows, whitespace separated: the gross return over month t+1, then the covariates observed at the
                      end of month t: the characteristics rank-transformed to (-1, 1) within the month (missing -> 0),
                      then the eight Welch-Goyal predictors (constant within a month; not used by the paper's features)
    out_perm_<t>.txt  the CRSP permno of each row of out_<t>.txt
    out_dlst_<t>.txt  permno, delisting code, delisting return of the rows that delisted in month t+1 (reference only)
    bond_<t>.txt      the price at the end of month t of one dollar paid at the end of month t+1 (one-month bill)
    datefile.txt      lines "t  end of month t  end of month t+1"
    columns.txt       optional: the column names of out_<t>.txt (the return first); without it the panel has the
                      layout of Gu, Kelly and Xiu (2020): 94 characteristics in the order of their datashare.csv

The paper's features drop the last N_MACRO covariates (the Welch-Goyal predictors), so a panel must end with them.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

N_MACRO = 8
MACRO_NAMES = ["dp", "ep", "bm", "ntis", "tbl", "tms", "dfy", "svar"]

# datashare.csv column order of Gu, Kelly and Xiu (2020), June-2021 release (permno, DATE and sic2 removed)
GKX_NAMES = (
    "mvel1 beta betasq chmom dolvol idiovol indmom mom1m mom6m mom12m mom36m pricedelay turn "
    "absacc acc age agr bm bm_ia cashdebt cashpr cfp cfp_ia chatoia chcsho chempia chinv chpmia "
    "convind currat depr divi divo dy egr ep gma grcapx grltnoa herf hire invest lev lgr mve_ia "
    "operprof orgcap pchcapx_ia pchcurrat pchdepr pchgm_pchsale pchquick pchsale_pchinvt "
    "pchsale_pchrect pchsale_pchxsga pchsaleinv pctacc ps quick rd rd_mve rd_sale realestate roic "
    "salecash saleinv salerec secured securedind sgr sin sp tang tb aeavol cash chtx cinvest ear "
    "nincr roaq roavol roeq rsup stdacc stdcf ms baspread ill maxret retvol std_dolvol std_turn "
    "zerotrade"
).split()
assert len(GKX_NAMES) == 94

# GKX names used by the code (micro-cap filter, idiosyncratic block) and their counterparts in the JKP panel
JKP_EQUIV = {"mvel1": "market_equity", "idiovol": "ivol_capm_252d", "retvol": "rvol_21d"}


def panel_columns(folder: str) -> list[str]:
    """Names of the columns of out_<t>.txt (the gross return first)."""
    f = os.path.join(folder, "columns.txt")
    if os.path.exists(f):
        cols = [ln.strip() for ln in open(f) if ln.strip()]
    else:
        cols = ["gross"] + GKX_NAMES + MACRO_NAMES
    if cols[-N_MACRO:] != MACRO_NAMES:
        raise ValueError(f"{folder}: the last {N_MACRO} columns are {cols[-N_MACRO:]}, not the Welch-Goyal predictors "
                         f"{MACRO_NAMES}; the features of the paper drop the last {N_MACRO} covariates")
    return cols


def char_index(folder: str, name: str) -> int:
    """0-based column of the characteristic `name` (a GKX name or its JKP counterpart) in the covariate matrix Z."""
    cols = panel_columns(folder)[1:]
    if name in cols:
        return cols.index(name)
    return cols.index(JKP_EQUIV[name])


def read_datefile(folder: str):
    """(t, end of month t, end of month t+1) as arrays of int, str, str."""
    rows = [ln.split() for ln in open(os.path.join(folder, "datefile.txt")) if ln.strip()]
    return (np.array([int(r[0]) for r in rows]), np.array([r[1] for r in rows]), np.array([r[2] for r in rows]))


def month_available(folder: str, t: int) -> bool:
    return os.path.exists(os.path.join(folder, f"out_{t}.txt"))


def read_matrix(path: str) -> np.ndarray:
    """A whitespace-separated numeric file as a 2-d float64 array (correctly rounded parsing)."""
    if os.path.getsize(path) == 0:
        return np.zeros((0, 0))
    return pd.read_csv(path, sep=r"\s+", header=None, dtype=np.float64, engine="c",
                       float_precision="round_trip").to_numpy()


def load_month(folder: str, t: int) -> dict:
    """The data of month t: gross returns R (N), covariates Z (N x d), permno (N), bond price."""
    X = read_matrix(os.path.join(folder, f"out_{t}.txt"))
    ncol = len(panel_columns(folder))
    if X.shape[1] != ncol:
        raise ValueError(f"out_{t}.txt has {X.shape[1]} columns, the panel layout has {ncol}")
    bond = float(open(os.path.join(folder, f"bond_{t}.txt")).read().split()[0])
    pfile = os.path.join(folder, f"out_perm_{t}.txt")
    permno = np.loadtxt(pfile, dtype=np.int64, ndmin=1) if os.path.exists(pfile) else None
    return dict(R=X[:, 0].copy(), Z=X[:, 1:].copy(), permno=permno, bond=bond)


def excess_returns(R: np.ndarray, bond: float) -> np.ndarray:
    """Excess return over month t+1: gross return minus the gross riskless return 1/bond."""
    return R - 1.0 - (1.0 / bond - 1.0)


def load_ff5(path: str) -> pd.DataFrame:
    """Kenneth French's monthly five-factor file (CSV as downloaded) as decimals, indexed by 'YYYY-MM'."""
    lines = open(path, encoding="latin-1").read().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip().startswith(",Mkt-RF"))
    rows = []
    for ln in lines[start + 1:]:
        parts = ln.split(",")
        if len(parts) < 7 or not parts[0].strip().isdigit() or len(parts[0].strip()) != 6:
            break
        rows.append([parts[0].strip()] + [float(p) for p in parts[1:7]])
    df = pd.DataFrame(rows, columns=["ym", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]).set_index("ym")
    df.index = [f"{s[:4]}-{s[4:]}" for s in df.index]
    return df / 100.0
