"""Command-line interface: `cmk <command> ...` (or `python -m cmk <command> ...`).

    cmk panel gkx  --datashare datashare.zip --goyal goyal_Monthly.csv --crsp-cache crsp/ --out panel_gkx/
    cmk panel jkp  --parquet jkp_usa.parquet --goyal goyal_Monthly.csv --crsp-cache crsp/ --out panel_jkp/
    cmk panel synthetic --out panel_synth/
    cmk stats      --panel-dir panel_gkx/ --work work/gkx
    cmk run        --config gkx --work work/gkx [--only cmk_linear ...] [--list]
    cmk pass2      --config gkx --work work/gkx --panel-dir panel_gkx/
    cmk tables     --config gkx --work work/gkx
    cmk simulate   --out work/simulation
    cmk all        --config gkx --panel-dir panel_gkx/ --work work/gkx       (stats, run, pass2, tables)

A work folder holds stats/ (pass 1), runs/ (the estimation runs), pass2/ and output/ (tables, figure, numbers.txt).
--workers caps the number of processes (default 16); every process uses one BLAS thread.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from .config import PANELS  # noqa: E402

FF5 = str(Path(__file__).resolve().parents[2] / "data" / "FF5.csv")


def _log(s):
    print(time.strftime("%H:%M:%S"), s, flush=True)


def cmd_panel(a):
    if a.which == "gkx":
        from .panel.gkx import build_gkx
        build_gkx(a.datashare, a.goyal, a.crsp_cache, a.out, a.wrds_user, log=_log)
    elif a.which == "jkp":
        from .panel.jkp import build_jkp
        build_jkp(a.parquet, a.goyal, a.crsp_cache, a.out, a.wrds_user, a.last_eom, log=_log)
    else:
        from .synthetic import write_synthetic_panel
        write_synthetic_panel(a.out, T=a.T, seed=a.seed)
        _log(f"synthetic panel of {a.T} months -> {a.out}")


def cmd_stats(a):
    from .jobs import STATS_FILES
    from .stats import build_stats
    files = list(STATS_FILES.values()) + ["stats_id.npz"]
    if not getattr(a, "force", False) and all(os.path.exists(os.path.join(a.work, "stats", f)) for f in files):
        _log(f"statistics in {a.work}/stats exist, pass 1 skipped (--force to rebuild)")
        return
    build_stats(a.panel_dir, os.path.join(a.work, "stats"), workers=a.workers, ff5=a.ff5 if os.path.exists(a.ff5) else None, log=_log)


def cmd_run(a):
    from .jobs import job_list, run_jobs, shards_of
    cfg = PANELS[a.config]
    if a.list:
        import numpy as np
        z = np.load(os.path.join(a.work, "stats", "stats_linear.npz"))
        for j in job_list(cfg, len(z["t"]), z["date0"]):
            n = len(shards_of(j, len(z["t"])))
            print(f"{j.name:26s} {j.kind:7s} {j.stats:11s} m={j.ms} step={j.step} shards={n}")
        return
    ok = run_jobs(a.work, cfg, workers=a.workers, only=a.only, skip=a.skip or (), log=_log)
    sys.exit(0 if ok else 1)


def cmd_pass2(a):
    from .pass2 import run_pass2
    run_pass2(a.work, a.panel_dir, PANELS[a.config], workers=a.workers, log=_log)


def cmd_tables(a):
    from .post.figures import rolling_figure
    from .post.numbers import write_numbers
    from .post.tables import build_tables
    out = a.out or os.path.join(a.work, "output")
    cfg = PANELS[a.config]
    R = build_tables(a.work, cfg, os.path.join(out, "tables"), a.ff5 if os.path.exists(a.ff5) else None, log=_log)
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    name = "figure1_rolling" if cfg.name == "gkx" else f"{cfg.name}_figure1_rolling"
    rolling_figure(R, os.path.join(out, "figures", name))
    lines = write_numbers(R, os.path.join(out, "numbers.txt"), a.work)
    _log(f"figure -> {out}/figures/{name}.pdf, {len(lines)} quoted numbers -> {out}/numbers.txt")


def cmd_simulate(a):
    from .simulation import run_simulation
    os.makedirs(a.out, exist_ok=True)
    run_simulation(os.path.join(a.out, "sim"), os.path.join(a.out, "figureC1_sim"), os.path.join(a.out, "tableC1_sim.tex"),
                   reps=a.reps, log=_log)


def cmd_all(a):
    cmd_stats(a)
    from .jobs import run_jobs
    if not run_jobs(a.work, PANELS[a.config], workers=a.workers, only=a.only, skip=a.skip or (), log=_log):
        sys.exit(1)
    cmd_pass2(a)
    a.out = None
    cmd_tables(a)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cmk", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("panel", help="build a panel folder (data preparation)")
    p.add_argument("which", choices=["gkx", "jkp", "synthetic"])
    p.add_argument("--out", required=True)
    p.add_argument("--datashare", help="datashare.csv or the zip containing it (gkx)")
    p.add_argument("--parquet", help="jkp_usa.parquet (jkp; pulled from WRDS when missing)")
    p.add_argument("--goyal", help="goyal_Monthly.csv (Welch-Goyal predictors)")
    p.add_argument("--crsp-cache", help="folder of the cached WRDS query results")
    p.add_argument("--wrds-user", default=None, help="WRDS user name (default: $WRDS_USER); password from ~/.pgpass")
    p.add_argument("--last-eom", default=None, help="jkp: last information month, e.g. 2025-11-30 (default: the data decide)")
    p.add_argument("--T", type=int, default=140, help="synthetic: number of months")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_panel)

    def common(p, panel=False, config=True):
        p.add_argument("--work", required=True, help="work folder of the panel")
        p.add_argument("--workers", type=int, default=16)
        p.add_argument("--ff5", default=FF5, help="Kenneth French's five-factor file (default: data/FF5.csv)")
        if config:
            p.add_argument("--config", required=True, choices=sorted(PANELS))
        if panel:
            p.add_argument("--panel-dir", required=True, help="the panel folder")

    p = sub.add_parser("stats", help="pass 1: the sufficient statistics")
    common(p, panel=True, config=False)
    p.add_argument("--force", action="store_true", help="rebuild existing statistics")
    p.set_defaults(fn=cmd_stats)
    p = sub.add_parser("run", help="the estimation runs of the paper")
    common(p)
    p.add_argument("--only", nargs="*", default=None, help="run only these jobs (names: --list)")
    p.add_argument("--skip", nargs="*", default=None)
    p.add_argument("--list", action="store_true", help="list the jobs and exit")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("pass2", help="pass 2: portfolio turnover and composition on the raw panel")
    common(p, panel=True)
    p.set_defaults(fn=cmd_pass2)
    p = sub.add_parser("tables", help="tables, figure and quoted numbers")
    common(p)
    p.add_argument("--out", default=None, help="output folder (default: <work>/output)")
    p.set_defaults(fn=cmd_tables)
    p = sub.add_parser("simulate", help="the simulation study of Online Appendix C")
    p.add_argument("--out", required=True)
    p.add_argument("--reps", type=int, default=100)
    p.set_defaults(fn=cmd_simulate)
    p = sub.add_parser("all", help="stats, run, pass2 and tables in one command")
    common(p, panel=True)
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--skip", nargs="*", default=None)
    p.set_defaults(fn=cmd_all)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
