"""Data preparation helpers: the rank transform of the R scripts and the output format."""
import numpy as np

from cmk.panel.common import _fmt, rank_unit


def test_rank_unit_matches_r():
    # R: x <- c(3, NA, 1, 3, 2); -1 + 2 * rank(x[!is.na(x)]) / (sum(!is.na(x)) + 1)  ->  0.4 0 -0.6 0.4 -0.2
    out = rank_unit(np.array([3.0, np.nan, 1.0, 3.0, 2.0]))
    assert np.allclose(out, [0.4, 0.0, -0.6, 0.4, -0.2])
    assert np.all(np.abs(rank_unit(np.random.default_rng(0).normal(size=1000))) < 1)


def test_number_format_matches_write_table():
    assert _fmt(0.833333) == "0.833333"
    assert _fmt(-0.6920698034018108) == "-0.692069803401811"
    assert _fmt(1.0) == "1"
    assert _fmt(0.0) == "0"
    assert _fmt(np.nan) == "NA"
    assert _fmt(12345) == "12345"
