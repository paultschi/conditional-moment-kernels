# Outputs and the paper

`cmk tables --config gkx --work W` writes into `W/output/`, `cmk tables --config jkp` into the work folder of the second
panel, `cmk simulate --out S` into `S/`. The runs are those of `cmk run --list` (module docstring of `src/cmk/jobs.py`).

| paper | output file | runs (work folder `runs/`) |
|---|---|---|
| Table 1, statistical performance | `tables/table1_stat.tex` | `cmk_linear`, `cmk_gauss_l1`, `cmk_gauss_l0.5`, `ipca_K{20,40,60,80,95}` |
| Table 2, portfolios | `tables/table2_ap.tex` | as Table 1, plus pass 2 (`pass2/*_pass.csv.gz`) and `data/FF5.csv` |
| Table 3, decomposition of the gain over IPCA | `tables/table3_decomp.tex` | `ipca_K60`, `ipca_K95`, `moment_step_linear`, `cmk_linear` |
| Figure 1, rolling performance | `figures/figure1_rolling.pdf` | `cmk_linear`, `cmk_gauss_*`, `ipca_K60` |
| Figure C.1, Table C.1, simulation study | `S/figureC1_sim.pdf`, `S/tableC1_sim.tex` | none (`cmk simulate`) |
| Table D.1, sub-periods | `tables/tableD1_periods.tex` | as Table 1 |
| Table D.2, robustness | `tables/tableD2_robust.tex` | `moment_step_*`, `idfloor_*`, `scalar_floor_*`, `browne_*`, `frobenius_k*`, `cmk_linear_val36`, `cmk_linear_val1`, `cmk_linear_large80`, `ipca_K60_norefit`, `onestep_linear_mf`, and the training-window rows of the headline runs |
| Table D.3, shape of the floor | `tables/tableD3_floorshape.tex` | `cmk_linear`, `cmk_gauss_l1`, `idfloor_linear`, `idfloor_gauss_l1` |
| Table D.4, covariance forecasts | `tables/tableD4_gmv.tex` | as Table 1 (the GMV columns `r_gmv`, `pv_gmv`, `pv_sdf`, `score0`) |
| Table D.5, second panel | `tables/tableD5_jkp.tex` (jkp) | the runs of Table 1 on the second panel, pass 2 |
| Table D.6, decomposition on the second panel | `tables/tableD6_decomp_jkp.tex` (jkp) | as Table 3 on the second panel |
| numbers in the text of Section 7 and Online Appendix D | `numbers.txt` | all runs; every line starts with the section it belongs to |

For the second panel, `cmk tables --config jkp` also writes its versions of Tables 1, 2 and D.1 to D.4 (`jkp_table*`),
from which the numbers of Online Appendix D.2 are quoted.

Numbers of the text that come from pass 1 (printed by `cmk stats` and repeated in `numbers.txt`): the size of the
cross sections, the agreement of the panel's riskless rate with Kenneth French's (Online Appendix D.1), the median
heuristic ell_0^2, the trace share captured by the 200 landmarks, and the ranks of the feature matrices.
Note on the landmark trace share: the main-panel paragraph of the paper (77% at ell_0, 17% at ell_0/2) reports the share
averaged over the months of the panel, the second-panel paragraph (89%, 35%) the share on the landmark pool;
`numbers.txt` prints both definitions for both panels.

The monthly result files (`runs/*_monthly.csv.gz`) have one row per test month, model and grid point; the columns are
documented in `cmk.model.evaluate_month` (`score`, `score_bm`, `r2_num`, `r2_den`, `r22_num`, `r22_den`, `tot_r2`,
`rho_f`, `theta`, `ret_sdf` (UMVE portfolio), `ret_cmve`, `ret_ew`, ...) and `cmk.runs` (`m`, `gamma` = the floor multiplier
eta, `thr` = theta = lambda / 2, `nfac_ex` = number of factors, `val_score`, `selected`, `refit`). `t` is the month number
of the panel and `date` the end of the month of the covariates; the returns are those of the following month.
