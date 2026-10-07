"""The two panels of the paper and their settings.

gkx  the main panel (Section 7): Gu, Kelly and Xiu (2020) characteristics, CRSP returns, January 1962 to December 2021
     (720 months), 94 characteristics, linear kernel m in {20, 40, 95}, additive kernel m in {152, 202, 295};
     594 test months from July 1972 (the first 18 months have fewer stocks than the largest model has features)
jkp  the second panel (Online Appendix D.2): Jensen, Kelly and Pedersen (2023) characteristics, February 1963 to December
     2025 (755 months), 121 characteristics, m in {20, 40, 122} and {179, 229, 322}; 647 test months from February 1972
"""
from __future__ import annotations

from dataclasses import dataclass, field

CRASHES = [("1987-08", "1987-12"), ("2000-03", "2002-10"), ("2007-10", "2009-03"), ("2020-02", "2020-03")]


@dataclass
class PanelConfig:
    name: str
    m_lin: list            # the last entry is the full linear feature space
    m_add: list            # the last entry is the full additive feature space
    first_test: int        # first test month (month number t of the panel) of the tables
    periods: list          # (label, first return month, last return month) of Table D.1
    ipca_K: list           # IPCA runs (re-estimated on training + validation months)
    ipca_pass2: list       # IPCA runs whose portfolios enter pass 2 (turnover, costs)
    tikhonov_checks: bool  # the runs without Tikhonov regularization (Online Appendix D.1)
    label: str = ""
    train: int = 96
    nval: int = 12
    extra: dict = field(default_factory=dict)

    @property
    def MF(self):
        return self.m_lin[-1]

    @property
    def MA(self):
        return self.m_add[-1]


GKX = PanelConfig(
    name="gkx", label="Gu, Kelly and Xiu (2020) characteristics, 1962-2021",
    m_lin=[20, 40, 95], m_add=[152, 202, 295], first_test=127,
    periods=[("full", None, None), ("1972--1989", "1971-01", "1989-12"), ("1990--2009", "1990-01", "2009-12"),
             ("2010--2021", "2010-01", "2021-12"), ("crashes", "crash", None)],
    ipca_K=[20, 40, 60, 80, 95], ipca_pass2=[20, 40, 60], tikhonov_checks=True)

JKP = PanelConfig(
    name="jkp", label="Jensen, Kelly and Pedersen (2023) characteristics, 1963-2025",
    m_lin=[20, 40, 122], m_add=[179, 229, 322], first_test=109,
    periods=[("full", None, None), ("1972--1989", "1971-01", "1989-12"), ("1990--2009", "1990-01", "2009-12"),
             ("2010--2025", "2010-01", "2025-12"), ("crashes", "crash", None)],
    ipca_K=[20, 40, 60], ipca_pass2=[20, 40, 60], tikhonov_checks=False)

# the synthetic panel of cmk.synthetic (140 months from 1990): small feature spaces, for a quick run of the whole pipeline
SYNTHETIC = PanelConfig(
    name="synthetic", label="synthetic panel", m_lin=[5, 10, 20], m_add=[25, 40, 60], first_test=109,
    periods=[("full", None, None), ("first half", None, "1999-12"), ("second half", "2000-01", None),
             ("crashes", "crash", None)],
    ipca_K=[20, 60], ipca_pass2=[20, 60], tikhonov_checks=False)

PANELS = {"gkx": GKX, "jkp": JKP, "synthetic": SYNTHETIC}
