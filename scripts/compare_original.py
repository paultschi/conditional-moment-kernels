#!/usr/bin/env python3
"""Compare this package with the original results of the paper (docs/verification.md).

    python scripts/compare_original.py --original <repository of the paper> --jkp-archive <unzipped results_jkp.zip> \
        --work-gkx work/gkx --work-jkp work/jkp

--original: the authors' repository (results/rev16 with the stored runs of the main panel, tables/ with the tables of the
paper). --jkp-archive: the result archive of the second panel (results/, with the file names of the original code).
The work folders need the statistics of pass 1 (cmk stats). Every check reruns a few windows and prints the largest
relative differences.
"""
from __future__ import annotations

import argparse
import gzip
import os
import shutil
import tempfile

import numpy as np
import pandas as pd

from cmk import runs
from cmk.config import JKP
from cmk.idio import IdStats
from cmk.stats import Stats

OLD_NAMES = {"rev8/floor3_lin": "cmk_linear", "rev8/floor3_l1": "cmk_gauss_l1", "rev8/floor3_l0.5": "cmk_gauss_l0.5",
             "rev8/former_lin": "moment_step_linear", "rev8/former_l1": "moment_step_gauss_l1",
             "rev8/former_l0.5": "moment_step_gauss_l0.5", "rev8/idfloor_lin": "idfloor_linear", "rev8/idfloor_l1": "idfloor_gauss_l1",
             "rev8/floor3_lin_val36": "cmk_linear_val36", "rev8/floor3_lin_val1": "cmk_linear_val1",
             "rev8/floor3_lin_micro": "cmk_linear_large80", "rev8/frobfloor_gram": "scalar_floor_gram",
             "rev8/frobfloor_identity": "scalar_floor_identity", "rev8/frob_k1_m40": "frobenius_k1",
             "rev8/frob_k100_m40": "frobenius_k100", "browne/browne_lin122all": "browne_linear_mf",
             "browne/browne_lin2040": "browne_linear_small", "floor/joint_lin122_g1": "onestep_linear_mf",
             "rev8/ext_l1": "ext_gauss_l1", "rev8/ext_l0.5": "ext_gauss_l0.5", "ipca_raw_useval_K20": "ipca_K20",
             "ipca_raw_useval_K40": "ipca_K40", "ipca_raw_useval_K60": "ipca_K60", "ipca_raw_K60": "ipca_K60_norefit",
             "cos_val12": "frobpath_linear", "cos_id3": "idio_linear"}


def report(name, new, old, key, cols):
    new = pd.DataFrame(new)
    j = new.merge(old, on=key, suffixes=("", "_old"))
    assert len(j) == len(new), f"{name}: {len(new) - len(j)} rows without a match"
    w = {c: float(np.max(np.abs(j[c] - j[c + "_old"]) / np.maximum(np.abs(j[c + "_old"]), 1e-12))) for c in cols if c in new}
    sel = bool((j.selected == j.selected_old).all()) if "selected" in new else "n/a"
    print(f"{name}: {len(j)} rows, selections identical: {sel}; largest relative differences "
          + ", ".join(f"{k} {v:.1e}" for k, v in w.items()), flush=True)


def read(path):
    return pd.read_csv(path if os.path.exists(path) else path + ".gz")


def check_gkx(original, work):
    r16 = os.path.join(original, "results", "rev16")
    key = ["t", "m", "gamma", "thr"]
    for stats, run, ms in [("linear", "floor3_lin", [20, 40, 95]), ("gauss_l1", "floor3_l1", [152, 295]),
                           ("gauss_l0.5", "floor3_l0.5", [202])]:
        st = Stats(os.path.join(work, "stats", f"stats_{stats}.npz"))
        rows, refit = runs.run_cmk(st, [18, 200, 400, 611], ms, gmv=True)
        for new, f in ((rows, "monthly"), (refit, "refit_monthly")):
            report(f"main panel, CMK estimator, {run} {f}", new, read(os.path.join(r16, f"{run}_{f}.csv.gz")), key,
                   ["score", "val_score", "ret_sdf", "u_id", "r22_num", "r_gmv", "score0"])


def check_jkp(archive, work):
    R = os.path.join(archive, "results")
    st = Stats(os.path.join(work, "stats", "stats_linear.npz"))
    S = dict(train=96, nval=12, refit=True)
    cols = ["score", "val_score", "ret_sdf", "u_id", "r22_num"]
    r, rf = runs.run_solver(st, [0, 324], [122], weight="frob", month_weight="invN", kappa=100.0, lams=[0.0, 2e-4, 6e-4, 2e-3, 6e-3],
                            cone_shifts=[0.0, 0.5, 1.0, 2.0], floor_shape="gram", ridge=1e-3, **S)
    report("second panel, scalar weight with floor", rf, read(f"{R}/rev8/frobfloor_gram_refit_monthly.csv"), ["t", "m", "gamma", "thr"], cols)
    r, rf = runs.run_solver(st, [0, 300], [40], weight="frob", kappa=100.0, lams=[0.0], **S)
    report("second panel, Frobenius weight", rf, read(f"{R}/rev8/frob_k100_m40_refit_monthly.csv"), ["t", "m", "lam"], cols)
    r, rf = runs.run_solver(st, [0, 400], [20, 40], weight="browne", lams=[0.0], **S)
    report("second panel, Browne weight", rf, read(f"{R}/browne/browne_lin2040_refit_monthly.csv"), ["t", "m", "lam"], cols)
    r, rf = runs.run_solver(st, [0, 324], [122], weight="gram", equal_months=True, ridge=1e-3, kappa=100.0, u_in_feature_block=True,
                            comp_scale="months", lams=[0.0], cone_shift=1.0, **S)
    report("second panel, one-step estimator", rf, read(f"{R}/floor/joint_lin122_g1_refit_monthly.csv"), ["t", "m", "lam"], cols)
    r = runs.run_path(st, [0, 300], [5, 20])
    path = read(f"{R}/cos_val12_monthly.csv")
    report("second panel, Frobenius path", r, path, ["t", "m", "lam"], cols)
    lam_sel = {(int(x.t), int(x.m)): float(x.lam) for x in path.query("selected == 1").itertuples()}
    r = runs.run_idio(st, IdStats(os.path.join(work, "stats", "stats_id.npz")), [0, 1], [5, 20], lam_sel)
    report("second panel, idiosyncratic block", r, read(f"{R}/cos_id3_monthly.csv"), ["t", "m", "mode"], ["r22_num", "loss_id", "u00"])
    r = runs.run_ipca(st, [0, 1, 2], 20)
    report("second panel, IPCA K = 20 (non-convex, see README)", r, read(f"{R}/ipca_raw_useval_K20_monthly.csv"), ["t", "m"],
           ["score", "train_loss", "u_id"])


def jkp_tables(archive, original, ff5):
    """Tables D.5 and D.6 from the archived runs of the second panel, compared with the tables of the paper."""
    from cmk.post.tables import build_tables
    W = tempfile.mkdtemp()
    os.makedirs(f"{W}/runs")
    os.makedirs(f"{W}/pass2")
    R = os.path.join(archive, "results")

    def gz(src, dst):
        with open(src, "rb") as a, gzip.open(dst, "wb") as b:
            shutil.copyfileobj(a, b)
    for o, n in OLD_NAMES.items():
        for k in ("monthly", "refit_monthly"):
            if os.path.exists(f"{R}/{o}_{k}.csv"):
                gz(f"{R}/{o}_{k}.csv", f"{W}/runs/{n}_{k}.csv.gz")
    for o, n in [("floor3_lin", "cmk_linear"), ("floor3_l1", "cmk_gauss_l1"), ("floor3_l0.5", "cmk_gauss_l0.5"), ("ipca", "ipca"),
                 ("switch", "switch")]:
        gz(f"{R}/rev8/pass2/{o}_pass.csv", f"{W}/pass2/{n}_pass.csv.gz")
    build_tables(W, JKP, f"{W}/tables", ff5, log=lambda s: None)
    for new, old in (("tableD5_jkp", "rev11_tab_jkp"), ("tableD6_decomp_jkp", "rev19_tab_decomp_jkp")):
        same = open(f"{W}/tables/{new}.tex").read() == open(os.path.join(original, "tables", f"{old}.tex")).read()
        print(f"{new}.tex identical to the paper's {old}.tex: {same}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", required=True)
    ap.add_argument("--jkp-archive", required=True)
    ap.add_argument("--work-gkx", required=True)
    ap.add_argument("--work-jkp", required=True)
    ap.add_argument("--ff5", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "FF5.csv"))
    a = ap.parse_args()
    check_gkx(a.original, a.work_gkx)
    check_jkp(a.jkp_archive, a.work_jkp)
    jkp_tables(a.jkp_archive, a.original, a.ff5)


if __name__ == "__main__":
    main()
