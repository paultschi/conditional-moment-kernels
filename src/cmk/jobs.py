"""Every estimation run of the paper as a job list, and a parallel scheduler.

The runs of a panel (see `job_list`) are written to <work>/runs/<name>_monthly.csv.gz (and *_refit_monthly.csv.gz for runs
with re-estimation); the IPCA runs used in pass 2 also write <name>_models.npz. Windows are independent for all runs except
IPCA and the idiosyncratic block (warm starts from the previous window), so the other runs are split into shards of
windows (and of m), executed in a process pool and concatenated. Finished runs are skipped when the scheduler is started
again, so an interrupted computation resumes.

Run names and their use in the paper (MF, MA: the full linear and additive feature spaces):
    cmk_linear, cmk_gauss_l1, cmk_gauss_l0.5     the CMK estimator, Tables 1, 2, 3, D.1-D.5, Figure 1
    moment_step_*                                the moment step of IPCA in the full feature space (F^ in every direction)
    idfloor_linear, idfloor_gauss_l1             the identity-shaped floor (gauss: every fourth window), Tables D.2, D.3
    scalar_floor_gram, scalar_floor_identity     the scalar weight with the two floors, solver, every fourth window
    frobenius_k1, frobenius_k100                 the Frobenius weight at m = 40 (the middle linear m), every second window
    browne_linear_mf, browne_linear_small        the Browne weight at m = MF and at m = 20, 40
    onestep_linear_mf                            the one-step estimator by the solver, every fourth window
    cmk_linear_val36, cmk_linear_val1            other splits of the 108 months; cmk_linear_large80: smallest 20% removed
    ext_gauss_l1, ext_gauss_l0.5                 the extended penalty grid at m = MA, every fourth window
    ipca_K<K>, ipca_K60_norefit                  IPCA
    frobpath_linear, idio_linear                 the characteristic-driven idiosyncratic specification
    pinv_linear, pinv_gauss_l1, trunc_linear, trunc_gauss_l1   the Tikhonov check (main panel)
"""
from __future__ import annotations

import json
import math
import os
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import runs
from .cmk import ETAS, THETAS
from .config import PanelConfig

STATS_FILES = {"linear": "stats_linear.npz", "gauss_l1": "stats_gauss_l1.npz", "gauss_l0.5": "stats_gauss_l0.5.npz"}


@dataclass
class Job:
    name: str
    kind: str                          # cmk | solver | ipca | path | idio
    stats: str = "linear"              # key of STATS_FILES
    universe: str = "full"
    ms: list = field(default_factory=list)
    opts: dict = field(default_factory=dict)
    step: int = 1
    windows: list | None = None        # explicit window starts (overrides step)
    chunk: int = 10 ** 6               # windows per shard
    split_m: bool = False              # one shard set per m
    minutes: float = 30.0              # rough CPU minutes per shard (scheduling order only)
    depends: list = field(default_factory=list)


def job_list(cfg: PanelConfig, T: int, dates=None) -> list[Job]:
    """The runs of the paper for a panel with T months (dates: end of month of every month index, for the date-based
    window sets of the Tikhonov check)."""
    MF, MA, ML, MAs = cfg.MF, cfg.MA, cfg.m_lin, cfg.m_add
    tr, nv = cfg.train, cfg.nval
    G = dict(etas=ETAS, thetas=THETAS, refit=True, gmv=True, train=tr, nval=nv)
    J = []
    # ---- headline runs: Tables 1, 2, D.1, D.4 (GMV), Figure 1
    J.append(Job("cmk_linear", "cmk", "linear", ms=ML, opts=G, chunk=160, minutes=40))
    for bw in ("l1", "l0.5"):
        J.append(Job(f"cmk_gauss_{bw}", "cmk", f"gauss_{bw}", ms=MAs, opts=G, split_m=True,
                     chunk={MAs[0]: 200, MAs[1]: 100, MAs[2]: 50}, minutes=40))
    # ---- Tables 3 and D.2: regularizers, floor shape, weights, protocol, universe
    J.append(Job("moment_step_linear", "cmk", "linear", ms=ML, opts=dict(G, former=True), chunk=320, minutes=20))
    for bw in ("l1", "l0.5"):
        J.append(Job(f"moment_step_gauss_{bw}", "cmk", f"gauss_{bw}", ms=MAs, opts=dict(G, former=True), split_m=True,
                     chunk={MAs[0]: 320, MAs[1]: 320, MAs[2]: 160}, minutes=25))
    J.append(Job("idfloor_linear", "cmk", "linear", ms=[MF], opts=dict(G, shape="identity"), chunk=160, minutes=40))
    J.append(Job("idfloor_gauss_l1", "cmk", "gauss_l1", ms=[MA], opts=dict(G, shape="identity"), step=4, chunk=60, minutes=40))
    J.append(Job("cmk_linear_val36", "cmk", "linear", ms=ML, opts=dict(G, train=72, nval=36), chunk=160, minutes=40))
    J.append(Job("cmk_linear_val1", "cmk", "linear", ms=ML, opts=dict(G, train=107, nval=1), chunk=160, minutes=40))
    J.append(Job("cmk_linear_large80", "cmk", "linear", universe="micro", ms=[MF], opts=G, chunk=320, minutes=40))
    S = dict(train=tr, nval=nv, refit=True)
    for sh in ("gram", "identity"):
        J.append(Job(f"scalar_floor_{sh}", "solver", "linear", ms=[MF], step=4, chunk=12, minutes=45,
                     opts=dict(S, weight="frob", month_weight="invN", kappa=100.0, lams=[0.0, 2e-4, 6e-4, 2e-3, 6e-3],
                               cone_shifts=[0.0, 0.5, 1.0, 2.0], floor_shape=sh, ridge=1e-3)))
    for k in (1.0, 100.0):                      # at the middle linear feature dimension (m = 40 in the paper)
        J.append(Job(f"frobenius_k{k:g}", "solver", "linear", ms=[ML[1]], step=2, chunk=320, minutes=25,
                     opts=dict(S, weight="frob", kappa=k, lams=[0.0])))
    J.append(Job("browne_linear_mf", "solver", "linear", ms=[MF], chunk=10, minutes=45, opts=dict(S, weight="browne", lams=[0.0])))
    J.append(Job("browne_linear_small", "solver", "linear", ms=ML[:2], chunk=80, minutes=40, opts=dict(S, weight="browne", lams=[0.0])))
    J.append(Job("onestep_linear_mf", "solver", "linear", ms=[MF], step=4, chunk=12, minutes=45,
                 opts=dict(S, weight="gram", equal_months=True, ridge=1e-3, kappa=100.0, u_in_feature_block=True,
                           comp_scale="months", lams=[0.0], cone_shift=1.0)))
    # ---- footnote of Online Appendix D.1: the extended penalty grid at the full additive feature space
    for bw in ("l1", "l0.5"):
        J.append(Job(f"ext_gauss_{bw}", "cmk", f"gauss_{bw}", ms=[MA], step=4, chunk=60, minutes=30,
                     opts=dict(G, thetas=[3e-3, 1e-2, 3e-2, 1e-1], gmv=False)))
    # ---- IPCA
    for K in cfg.ipca_K:
        J.append(Job(f"ipca_K{K}", "ipca", "linear", ms=[K], minutes=K * 6,
                     opts=dict(train=tr, nval=nv, use_val=True, gmv=True, models=K in cfg.ipca_pass2)))
    J.append(Job("ipca_K60_norefit", "ipca", "linear", ms=[60], minutes=360, opts=dict(train=tr, nval=nv, use_val=False)))
    # ---- the characteristic-driven idiosyncratic specification
    J.append(Job("frobpath_linear", "path", "linear", ms=[5, 10, 20, 40], chunk=80, minutes=40, opts=dict(train=tr, nval=nv)))
    J.append(Job("idio_linear", "idio", "linear", ms=[5, 10, 20, 40], minutes=150, depends=["frobpath_linear"],
                 opts=dict(train=tr, nval=nv)))
    # ---- the Tikhonov check (Online Appendix D.1), main panel
    if cfg.tikhonov_checks:
        J.append(Job("pinv_linear", "cmk", "linear", ms=ML, opts=dict(G, coords="pinv", gmv=False), chunk=160, minutes=40))
        J.append(Job("trunc_linear", "cmk", "linear", ms=[MF], opts=dict(G, coords="pinv", pinv_tol=1e-3, pinv_rel="mean",
                                                                     gmv=False), chunk=320, minutes=40))
        n = T - tr - nv
        test_date = lambda t0: str(dates[t0 + tr + nv])
        sub = sorted({t0 for t0 in range(n) if (t0 % 2 == 0 and test_date(t0) < "1991") or
                      (t0 % 6 == 0 and test_date(t0) >= "1986")}) if dates is not None else list(range(0, n, 2))
        J.append(Job("pinv_gauss_l1", "cmk", "gauss_l1", ms=[MAs[0], MA], windows=sub, split_m=True, chunk=30, minutes=40,
                     opts=dict(G, coords="pinv", gmv=False)))
        J.append(Job("trunc_gauss_l1", "cmk", "gauss_l1", ms=[MA], step=6, chunk=30, minutes=40,
                     opts=dict(G, coords="pinv", pinv_tol=1e-3, pinv_rel="mean", gmv=False)))
    return J


# ------------------------------------------------------------------------------------------------- execution
_CACHE = {}


def _stats(work, key, universe):
    from .stats import Stats
    k = (key, universe)
    if k not in _CACHE:
        if len(_CACHE) >= 2:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[k] = Stats(os.path.join(work, "stats", STATS_FILES[key]), universe)
    return _CACHE[k]


def _task(work, job: Job, ms, t0s, shard_path):
    """Execute one shard of a job; writes <shard_path>_monthly.csv(.gz) / _refit_monthly.csv / _models.npz."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    t_start = time.time()
    logf = open(shard_path + ".log", "w")
    log = lambda s: (logf.write(f"{time.time() - t_start:8.0f}s {s}\n"), logf.flush())
    try:
        st = _stats(work, job.stats, job.universe)
        o = dict(job.opts)
        refit_rows = []
        if job.kind == "cmk":
            rows, refit_rows = runs.run_cmk(st, t0s, ms, log=log, **o)
        elif job.kind == "solver":
            rows, refit_rows = runs.run_solver(st, t0s, ms, log=log, **o)
        elif job.kind == "ipca":
            models = {} if o.pop("models", False) else None
            rows = runs.run_ipca(st, t0s, ms[0], models=models, log=log, **o)
            if models is not None:
                np.savez_compressed(shard_path + "_models.npz", **models)
        elif job.kind == "path":
            rows = runs.run_path(st, t0s, ms, log=log, **o)
        elif job.kind == "idio":
            from .idio import IdStats
            path = pd.read_csv(os.path.join(work, "runs", "frobpath_linear_monthly.csv.gz"))
            lam_sel = {(int(r.t), int(r.m)): float(r.lam) for r in path[path.selected == 1].itertuples()}
            ids = IdStats(os.path.join(work, "stats", "stats_id.npz"), (0, 1, 2))
            rows = runs.run_idio(st, ids, t0s, ms, lam_sel, log=log, **o)
        else:
            raise ValueError(job.kind)
        pd.DataFrame(rows).to_csv(shard_path + "_monthly.csv", index=False)
        if refit_rows:
            pd.DataFrame(refit_rows).to_csv(shard_path + "_refit_monthly.csv", index=False)
        log("done")
        return shard_path, None
    except Exception:                                   # noqa: BLE001 -- reported by the scheduler
        tb = traceback.format_exc()
        log(tb)
        return shard_path, tb
    finally:
        logf.close()


def shards_of(job: Job, T: int):
    """(ms, t0s, shard name) of every shard of the job."""
    all_t0 = job.windows if job.windows is not None else runs.windows(T, job.opts.get("train", 96), job.opts.get("nval", 12), job.step)
    out = []
    groups = [[m] for m in job.ms] if job.split_m else [job.ms]
    for ms in groups:
        chunk = job.chunk[ms[0]] if isinstance(job.chunk, dict) else job.chunk
        if job.kind in ("ipca", "idio"):
            chunk = len(all_t0)
        n = max(1, math.ceil(len(all_t0) / chunk))
        for i in range(n):
            t0s = all_t0[i * chunk:(i + 1) * chunk]
            if t0s:
                tag = f"m{ms[0]}_" if job.split_m else ""
                out.append((ms, t0s, f"{job.name}.{tag}s{i}"))
    return out


def merge(work, job: Job, T: int) -> bool:
    """Concatenate the shards of a finished job into <work>/runs/<name>_*.csv.gz."""
    sh = os.path.join(work, "runs", "shards")
    parts = shards_of(job, T)
    for kind in ("monthly", "refit_monthly"):
        fs = [os.path.join(sh, n + f"_{kind}.csv") for _, _, n in parts]
        have = [f for f in fs if os.path.exists(f)]
        if kind == "monthly" and len(have) < len(fs):
            return False
        if not have:
            continue
        d = pd.concat([pd.read_csv(f) for f in have if os.path.getsize(f) > 1], ignore_index=True)
        d = d.sort_values(["t", "m"], kind="stable").reset_index(drop=True)
        d.to_csv(os.path.join(work, "runs", f"{job.name}_{kind}.csv.gz"), index=False)
    mf = [os.path.join(sh, n + "_models.npz") for _, _, n in parts]
    if all(os.path.exists(f) for f in mf):
        models = {}
        for f in mf:
            z = np.load(f)
            models.update({k: z[k] for k in z.files})
        np.savez_compressed(os.path.join(work, "runs", f"{job.name}_models.npz"), **models)
    json.dump(dict(name=job.name, kind=job.kind, stats=job.stats, universe=job.universe, ms=job.ms, step=job.step,
                   windows=None if job.windows is None else len(job.windows), opts=job.opts),
              open(os.path.join(work, "runs", f"{job.name}_config.json"), "w"), indent=1, default=str)
    return True


def done(work, name) -> bool:
    return os.path.exists(os.path.join(work, "runs", f"{name}_monthly.csv.gz"))


def run_jobs(work, cfg: PanelConfig, workers: int = 16, only=None, skip=(), log=print):
    """Execute every job of the panel (or the jobs named in `only`) with at most `workers` processes."""
    os.makedirs(os.path.join(work, "runs", "shards"), exist_ok=True)
    z = np.load(os.path.join(work, "stats", STATS_FILES["linear"]))
    T, dates = len(z["t"]), z["date0"]
    jobs = [j for j in job_list(cfg, T, dates) if (only is None or j.name in only) and j.name not in skip]
    byname = {j.name: j for j in jobs}
    pending = {j.name for j in jobs if not done(work, j.name)}
    for j in jobs:
        if j.name not in pending:
            log(f"{j.name}: done, skipped")
    remaining = {}                                        # job -> shards still to run
    for j in jobs:
        if j.name in pending:
            sh = [s for s in shards_of(j, T) if not os.path.exists(os.path.join(work, "runs", "shards", s[2] + "_monthly.csv"))]
            remaining[j.name] = sh
            log(f"{j.name}: {len(sh)} shards to run")
    started, failed = set(), set()
    queue = []

    def ready(j):
        return all(d not in pending for d in j.depends)

    def enqueue():
        for name in list(remaining):
            j = byname[name]
            if name not in started and ready(j):
                started.add(name)
                for ms, t0s, sname in remaining[name]:
                    queue.append((j.minutes * (10 if j.kind in ("ipca", "idio") else 1), j, ms, t0s, sname))
        queue.sort(key=lambda q: -q[0])

    def finish(name):
        if merge(work, byname[name], T):
            pending.discard(name)
            log(f"{name}: merged -> runs/{name}_monthly.csv.gz")
        else:
            log(f"!! {name}: shards missing after the run")
            failed.add(name)
            pending.discard(name)

    for name in [n for n, sh in remaining.items() if not sh]:     # all shards present from an earlier run
        finish(name)
        remaining.pop(name)
    enqueue()
    open_shards = {n: len(sh) for n, sh in remaining.items()}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {}
        while queue or futs:
            while queue and len(futs) < workers:
                _, j, ms, t0s, sname = queue.pop(0)
                f = ex.submit(_task, work, j, ms, t0s, os.path.join(work, "runs", "shards", sname))
                futs[f] = j.name
            fin, _ = wait(futs, return_when=FIRST_COMPLETED)
            for f in fin:
                name = futs.pop(f)
                path, err = f.result()
                if err:
                    log(f"!! {os.path.basename(path)} failed:\n{err}")
                    failed.add(name)
                open_shards[name] -= 1
                if open_shards[name] == 0:
                    if name in failed:
                        pending.discard(name)
                    else:
                        finish(name)
                    enqueue()
            log(f"[{(time.time() - t0) / 60:6.1f} min] running {len(futs)}, queued {len(queue)}, "
                f"jobs left {len(pending)}")
    if failed:
        log(f"failed jobs: {sorted(failed)}")
    return not failed
