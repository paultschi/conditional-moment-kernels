"""Replication code for "Conditional Moment Kernels: Joint Modeling of Expected Returns and Covariances in Unbalanced Panels".

The package follows the pipeline of the paper:

    cmk.panel     data preparation: the monthly stock panels (Gu-Kelly-Xiu characteristics with CRSP returns, and the
                  Jensen-Kelly-Pedersen characteristics) in a common folder layout
    cmk.stats     pass 1: per-month sufficient statistics of the panel (Lemma 3.3 of the paper)
    cmk.model     conditional moments, scoring rule, factor and UMVE portfolios of a fitted moment kernel (Sections 3, 6, 7)
    cmk.cmk       the CMK estimator, the two-stage closed form of Section 4.5
    cmk.solver    the accelerated projected gradient solver of Appendix B (one-step estimator, other weights)
    cmk.weights   the weights of the loss (Gram-whitened, Frobenius, scalar, Browne)
    cmk.idio      the characteristic-driven idiosyncratic block (Online Appendix D)
    cmk.ipca      the IPCA benchmark of Kelly, Pruitt and Su (2019)
    cmk.runs      the rolling-window protocol of Section 7.1 for every estimator
    cmk.jobs      the list of every run of the paper and a parallel scheduler
    cmk.pass2     pass 2 over the raw panel: turnover and composition of the portfolios
    cmk.post      tables, figures and the numbers quoted in the text
    cmk.simulation  the simulation study of Online Appendix C

`python -m cmk --help` (or the `cmk` console script) lists the commands.
"""
__version__ = "1.0.0"
