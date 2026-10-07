# Verification

The package is a rewrite of the authors' research code. The numerical core (pass 1, the CMK estimator, the solver,
the weights, IPCA, the idiosyncratic block, the evaluation) keeps the arithmetic of that code. It was checked
against the original panels and the stored results of the paper (October 2026) without rerunning every run of the paper;
`scripts/compare_original.py` contains the comparisons and can be run against the authors' result archive.

## Data preparation

* Main panel (`cmk panel gkx`): without WRDS access, the characteristic columns were rebuilt from `datashare.zip` for the
  cross sections of the original panel (months 1, 100, 300, 500 and 720). The rank-transformed characteristics agree with
  the original `out_<t>.txt` to the printed precision (5e-16), and the Welch-Goyal columns to 4e-16 with the release of
  `goyal_Monthly.csv` that was current when the panel was built (November 2023; the September 2026 release revises
  them by up to 2e-4, which does not matter, since the paper's features do not use them). The CRSP part (sample filter,
  returns, delistings, bill) is a line-by-line port of the original R script and was not run, because it needs WRDS
  credentials.
* Second panel (`cmk panel jkp`): run on the original `jkp_usa.parquet`, with the riskless rate taken from the original
  panel instead of WRDS, the builder reproduces the original panel: the same 755 months, 121 characteristics and stocks
  in every month, `datefile.txt` and the first months byte-identical, all values equal to 1e-15.

## Pass 1

`cmk stats` on the original panels reproduces the numbers of the paper: main panel N_t up to 6,848 (1997), on average
3,731; riskless rate against Kenneth French's: correlation 0.995, mean absolute difference 1.8 basis points; ell_0^2 = 36.4;
second panel ell_0^2 = 67.6, landmarks capturing 89% and 35% of the trace on the pool.

## Estimation

| check | result |
|---|---|
| CMK estimator, main panel, linear (m = 20, 40, 95), additive l0 (m = 152, 295), additive l0/2 (m = 202), windows 18, 200, 400, 611, against the stored runs of the paper | every column equal to relative 1e-15 (linear) to 1e-7 (the additive features are stored in float32); selected hyper-parameters and number of factors identical |
| solver: scalar weight with floor, Frobenius weight, Browne weight, one-step estimator, against the stored second-panel runs | equal to the solver tolerance (relative 1e-16 to 1e-6), same selections |
| Frobenius path and idiosyncratic block, second panel | relative 1e-12 to 1e-16 |
| IPCA, second panel, first windows of the warm-start chain | training objective equal to 1e-4; out-of-sample scores to 1e-3 (different local optima of the non-convex objective, see README) |
| pass 2, second panel, linear and additive models | score to 1e-10, turnover and weights to 1e-8 (the original stored the window basis in float32) |
| simulation study (Online Appendix C), complete rerun | Table C.1 identical to the paper; all simulation outputs equal to relative 1e-10 |

## Tables and numbers

* From the stored second-panel results, `cmk tables --config jkp` writes Tables D.5 and D.6 byte-identical to the tables of
  the paper, and the numbers of the paragraph on the second panel in Online Appendix D.2 agree with the text. Its
  versions of Tables 1, 2 and D.1 to D.3 agree with the original ones to relative 1e-7 or better.
* From the stored main-panel runs of the CMK estimator and IPCA, Table 1 is identical to the paper in all CMK rows and
  in the IPCA rows with K = 80 and 95. The rows with K = 20, 40 and 60 differ by at most 0.3 score points, because the
  stored IPCA runs with these K are themselves a rerun of the paper's (documented by the authors). Table D.4 is identical.
* Unit tests (`pytest`): the closed-form evaluation against dense N x N algebra, the solver and the projection against
  the interior-point solver Clarabel, the CMK estimator as the solution of its reduced program (Proposition 4.2) and its
  unregularized limit, the rank transform against R.
* An end-to-end run on the synthetic panel (`cmk all --config synthetic`) exercises every job, pass 2 and all tables.

## Not reproduced exactly

* The sub-grid of windows of the Tikhonov check in the additive feature space ("every second window to 1990 and every
  sixth window from 1985", Online Appendix D.1) is defined in `cmk.jobs` from that description; it has 173 reported test
  months against the 174 of the original run, so these few text numbers can differ in the last digit.
* IPCA, as explained in the README.
