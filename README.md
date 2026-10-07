# Conditional Moment Kernels: replication package

Python code that reproduces every table, figure and quoted number of

> Damir Filipović and Paul Schneider, *Conditional Moment Kernels: Joint Modeling of Expected Returns and Covariances in
> Unbalanced Panels* (2026).

The package covers the whole chain: building the monthly stock panels from the raw sources, reducing them to per-month
sufficient statistics, the rolling-window estimation of the CMK estimator and of every benchmark and robustness
specification, the portfolio pass over the raw panel, and the tables, figures and numbers of the paper. The simulation
study of Online Appendix C needs no data.

## Contents

| path | content |
|---|---|
| `src/cmk/` | the Python package (module overview in `src/cmk/__init__.py`, every module has a docstring with the formulas and the paper references) |
| `src/cmk/panel/` | data preparation: the main panel (Gu, Kelly and Xiu characteristics with CRSP returns) and the second panel (Jensen, Kelly and Pedersen characteristics) |
| `tests/` | unit tests: closed-form evaluation against dense algebra, the solver and the CMK estimator against an interior-point solver, the rank transform |
| `data/FF5.csv` | the Fama-French five factors and the riskless rate from Kenneth French's data library (FF5 regressions of Table 2, riskless-rate check) |
| `scripts/reproduce.sh` | the whole replication in one script |
| `docs/outputs.md` | which output file is which table, figure or number of the paper, and which runs feed it |
| `docs/verification.md` | how this package was checked against the original results |

## Installation

Python 3.11 or later.

```
python -m venv .venv && source .venv/bin/activate
pip install -e ".[data,test]"          # numpy, scipy, pandas, matplotlib; pyarrow and psycopg for the data step; pytest, cvxpy for the tests
pytest                                  # about 2 seconds
```

`requirements.txt` lists the versions with which the package was tested.

## Data

| source | used for | access |
|---|---|---|
| CRSP monthly stock file, delisting file, security history and one-month bill (`crsp.msf`, `crsp.msedelist`, `crsp.stksecurityinfohist`, `crsp.tfz_mth_rf`, `crsp.mcti`) | returns, sample, riskless rate | WRDS subscription (licensed, not redistributable) |
| the 94 firm characteristics of Gu, Kelly and Xiu (2020), `datashare.csv` | main panel | public: Dacheng Xiu's website (`datashare.zip`, June-2021 release) |
| the stock-level characteristics of Jensen, Kelly and Pedersen (2023), `contrib.global_factor` | second panel | WRDS (pulled by `cmk panel jkp`) |
| the monthly predictors of Welch and Goyal (2008), `goyal_Monthly.csv` | written into the panels, not used by the estimator | public: Amit Goyal's website |
| Fama-French five factors and RF | Table 2, riskless-rate check | public: Kenneth French's data library (`data/FF5.csv`) |

WRDS access uses the PostgreSQL server of WRDS. The user name is passed with `--wrds-user` or the environment variable
`WRDS_USER`; the password is read from `~/.pgpass` (see the WRDS documentation on connecting from Python). Every query result
is cached as a CSV file in the folder given by `--crsp-cache`, so WRDS is contacted once.

## Running the replication

All commands take a *work folder* per panel, which receives the statistics (`stats/`), the runs (`runs/`), the portfolio pass
(`pass2/`) and the outputs (`output/tables`, `output/figures`, `output/numbers.txt`). `--workers` caps the number of parallel
processes (default 16, one BLAS thread each). Every step skips work that is already done, so an interrupted run resumes.

```
# 1. data preparation (needs WRDS)
cmk panel gkx --datashare datashare.zip --goyal goyal_Monthly.csv --crsp-cache crsp --out panel_gkx
cmk panel jkp --parquet jkp_usa.parquet --goyal goyal_Monthly.csv --crsp-cache crsp --out panel_jkp

# 2. pass 1: sufficient statistics (a minute per panel)
cmk stats --panel-dir panel_gkx --work work/gkx

# 3. the estimation runs of the paper (see `cmk run --config gkx --work work/gkx --list`)
cmk run --config gkx --work work/gkx

# 4. pass 2: portfolio turnover and composition on the raw panel
cmk pass2 --config gkx --panel-dir panel_gkx --work work/gkx

# 5. tables, Figure 1 and the quoted numbers
cmk tables --config gkx --work work/gkx

# the same for the second panel (--config jkp), and the simulation study
cmk simulate --out work/simulation
```

`cmk all --config gkx --panel-dir panel_gkx --work work/gkx` runs steps 2 to 5, and `scripts/reproduce.sh` runs everything.
Without WRDS access the whole chain can be exercised on a synthetic panel in a few minutes:

```
cmk panel synthetic --out panel_synth
cmk all --config synthetic --panel-dir panel_synth --work work/synth
```

### Run times

Measured on the second panel (one core per run): pass 1 takes a minute. The CMK estimator takes about one CPU hour in the
linear feature space and about 13 CPU hours per bandwidth in the additive feature space. IPCA takes 2 to 5 hours per
number of factors; it cannot be split, because every window starts from the previous one. The robustness runs that need
the iterative solver dominate: the Browne weight at the full linear feature space takes about 75 CPU hours, the one-step
estimator 17, the scalar weight with the floor 11 per floor shape. All runs of one panel take about 175 CPU hours, about
11 hours with 16 workers. The headline runs alone
(`--only cmk_linear cmk_gauss_l1 cmk_gauss_l0.5 ipca_K20 ipca_K40 ipca_K60 ipca_K80 ipca_K95`) take about 40 CPU hours,
5 hours with 16 workers (the longest IPCA run), and give Tables 1, D.1 and D.4 and Figure 1. Tables 2 and 3 also need
pass 2 (an hour) and `moment_step_linear`.

## Reproducibility

Every run is deterministic. The CMK estimator is a closed form, and its runs reproduce the original results to
floating-point accuracy (relative differences below 1e-9). The iterative solver reproduces them to its tolerance (1e-6).
IPCA is the exception: its least-squares objective is not convex, the L-BFGS optimum depends on the floating-point
environment (BLAS, numpy version, number of threads), and the warm start carries a different local optimum forward from
window to window. Its rows move by a few tenths of a score point between machines; no conclusion depends on it.
`docs/verification.md` lists the checks.

## Notes on the data

* The main panel is built with the CRSP CIZ security history (`crsp.stksecurityinfohist`), with the conditions that
  replicate the legacy share codes 10 and 11. CRSP revises its files, so a panel built later can differ slightly.
* The Welch-Goyal predictors are written into the panels for completeness. The features of the paper do not use them.
* The second panel uses JKP's return, which includes the delisting return; `out_dlst_<t>.txt` is for reference only.

## License

The code is released under the MIT license (`LICENSE`). The data are subject to the terms of their providers; CRSP and
WRDS data are not included and cannot be redistributed.
