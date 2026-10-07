"""The rolling-window protocol of Section 7.1 for every estimator of the paper.

A window starts at month index t0: training months [t0, t0 + train), validation months [t0 + train, t0 + train + nval),
test month t0 + train + nval (features at the end of that month, returns over the following month). For every window
the runners record one row per model and grid point with the out-of-sample quantities of cmk.model.SCALARS on the test
month and the validation score (the average score on the validation months, lower is better); `selected` marks the grid
point with the lowest validation score. With refit, the model is re-estimated, window basis included, on the
train + nval months with the hyper-parameters of every grid point, and evaluated on the same test month (the protocol of
the paper; rows with refit = 1, written to the *_refit rows).

Runners (each takes the list of window starts it should process, so that runs can be split into shards):
    run_cmk        the CMK estimator (cmk.cmk) on the (eta, theta) grid; also the moment step of IPCA (former), the
                   identity-shaped floor, and the pseudoinverse / truncated coordinates of the Tikhonov check
    run_solver     the CMK program by the solver of Appendix B: the Frobenius and scalar weights (with or without floor),
                   the Browne weight, and the one-step estimator (Gram-whitened loss with u in the feature block)
    run_ipca       IPCA with K factors (sequential: every window starts from the loadings of the previous one)
    run_path       the Frobenius loss on a lambda grid (the isotropic reference of the idiosyncratic specification)
    run_idio       the characteristic-driven idiosyncratic block (sequential warm starts)
"""
from __future__ import annotations

import sys

import numpy as np

from .cmk import ETAS, THETAS, floor_matrix, stage1, stage1_pinv, stage2
from .model import GMV_COLUMNS, SCALARS, benchmark_sigma2, evaluate_month, month_features, score_month, window_basis
from .solver import FistaQP
from .weights import browne_features, gram_features, month_loss, month_loss_aug, plain_features


def windows(T: int, train: int = 96, nval: int = 12, step: int = 1, first: int = 0, last: int | None = None):
    """Window starts t0 with a test month t0 + train + nval < T (last: exclusive bound on t0)."""
    n = T - train - nval
    return list(range(first, n if last is None else min(last, n), step))


def _rec(st, t_te, **kw):
    return dict(t=int(st.t[t_te]), date=st.date0[t_te], **kw)


# ------------------------------------------------------------------------------------------------- CMK estimator
def run_cmk(st, t0s, ms, etas=ETAS, thetas=THETAS, train=96, nval=12, ridge=1e-3, refit=True, former=False,
            shape="gram", coords="ridge", pinv_tol=1e-9, pinv_rel="max", gmv=False, log=None):
    """The CMK estimator on the (eta, theta) grid (theta = lambda / 2); returns (rows, refit rows)."""
    SC = SCALARS + (GMV_COLUMNS if gmv else [])
    grid = [(0.0, 0.0)] if former else [(g, th) for g in etas for th in thetas]
    s1 = (lambda A, idx: stage1(st, A, idx, ridge)) if coords == "ridge" else \
        (lambda A, idx: stage1_pinv(st, A, idx, pinv_tol, pinv_rel))
    rows, rows_refit = [], []
    for t0 in t0s:
        idx_tr = list(range(t0, t0 + train))
        idx_val = list(range(t0 + train, t0 + train + nval))
        t_te = t0 + train + nval
        s2 = benchmark_sigma2(st, idx_tr)
        for m in ms:
            A, ev = window_basis(st, idx_tr, m)
            resid_trace = float(ev[m:].sum() / ev.sum())
            b, u, Abar, Xi = s1(A, idx_tr)
            if not (u > 1e-12):
                print(f"  t={int(st.t[t_te])} m={m}: u^id = 0, skipped", file=sys.stderr)
                continue
            feats_val = [month_features(st, A, t) for t in idx_val]
            recs = []
            for g, th in grid:
                P = stage2(b, u, Abar, Xi, g, th, former, shape)
                vs = float(np.mean([score_month(P, f) for f in feats_val]))
                ev_te = evaluate_month(P, A, st, t_te, s2, gmv=gmv)
                rec = _rec(st, t_te, m=m, lam=2 * th, gamma=g, thr=th, val_score=vs, resid_trace=resid_trace,
                           status="closed-form", nfac_ex=P.nfac_ex)
                rec.update({k: ev_te[k] for k in SC})
                recs.append(rec)
            best = int(np.argmin([r["val_score"] for r in recs]))
            for k, rec in enumerate(recs):
                rec["selected"] = int(k == best)
                rec["refit"] = 0
                rows.append(rec)
            if refit:
                idx_all = idx_tr + idx_val
                A2, ev2 = window_basis(st, idx_all, m)
                resid2 = float(ev2[m:].sum() / ev2.sum())
                b2, u2, Abar2, Xi2 = s1(A2, idx_all)
                if not (u2 > 1e-12):
                    print(f"  t={int(st.t[t_te])} m={m}: refit u^id = 0, skipped", file=sys.stderr)
                    continue
                s2_all = benchmark_sigma2(st, idx_all)
                for k, (g, th) in enumerate(grid):
                    P2 = stage2(b2, u2, Abar2, Xi2, g, th, former, shape)
                    ev2_te = evaluate_month(P2, A2, st, t_te, s2_all, gmv=gmv)
                    rec2 = _rec(st, t_te, m=m, lam=2 * th, gamma=g, thr=th, val_score=recs[k]["val_score"],
                                resid_trace=resid2, status="closed-form", nfac_ex=P2.nfac_ex)
                    rec2.update({kk: ev2_te[kk] for kk in SC})
                    rec2["selected"] = int(k == best)
                    rec2["refit"] = 1
                    rows_refit.append(rec2)
        if log:
            log(f"test month {int(st.t[t_te])} ({st.date0[t_te]})")
    return rows, rows_refit


def refit_model(st, t_te, m, eta, theta, train=96, nval=12, ridge=1e-3):
    """The CMK estimator re-estimated on the train + nval months before month index t_te (pass 2); returns (A, P, s2)."""
    idx_all = list(range(t_te - train - nval, t_te))
    A, _ = window_basis(st, idx_all, m)
    P = stage2(*stage1(st, A, idx_all, ridge), eta, theta)
    return A, P, benchmark_sigma2(st, idx_all)


# ------------------------------------------------------------------------------------------------- the solver
class _Shifted:
    """Passes the floor S of program (krr1floor) to every solve."""

    def __init__(self, qp, S):
        self.qp, self.S = qp, S

    def solve(self, lam, **kw):
        return self.qp.solve(lam, shift=self.S, **kw)

    def loss(self, P):
        return self.qp.loss(P)


def run_solver(st, t0s, ms, weight="frob", lams=(0.0,), train=96, nval=12, kappa=100.0, ridge=1e-8, refit=True,
               month_weight="one", equal_months=False, u_in_feature_block=False, comp_scale="one", cone_shift=0.0,
               cone_shifts=None, floor_shape="gram", first_stage_w="one", log=None):
    """The CMK program by the solver of Appendix B on the lambda grid (and the floor grid cone_shifts); weights:
        frob    Omega_t = Diag(kappa, I) with month weight 1 (month_weight="one") or 1/N_t ("invN": the scalar weight
                N_t^{-1/2} I of Section 3.1 up to the first-moment weight kappa)
        browne  the Browne weight of Online Appendix D.1, first stage = Frobenius fit at lambda = 0 with kappa = 1
        gram    the Gram-whitened loss (with u_in_feature_block and equal_months: the one-step estimator (krr1floor))
    Returns (rows, refit rows)."""
    rows, rows_refit = [], []

    def fit(idx_fit, m):
        A, ev = window_basis(st, idx_fit, m)
        resid_trace = float(ev[m:].sum() / ev.sum())
        if weight == "browne":
            Nbar = float(np.mean(st.N[idx_fit]))
            w_fs = [1.0 / Nbar if first_stage_w == "one" else st.w[t] for t in idx_fit]
            w_b = {t: (1.0 / Nbar if month_weight == "one" else st.w[t]) for t in idx_fit}
            P1 = FistaQP(st, A, idx_fit, kappa=1.0, wts=w_fs).solve(0.0)
            feats_tr = [browne_features(st, A, t, P1, w_b[t]) for t in idx_fit]
            qp = FistaQP(st, A, idx_fit, kappa=1.0, feats=feats_tr)
            feats_val = lambda idx_val: [browne_features(st, A, t, P1, 1.0 / Nbar) for t in idx_val]
        elif weight == "gram":
            gam = lambda t: (1.0 / np.sqrt(st.w[t]) if comp_scale == "months" else 1.0)
            pairs = [gram_features(st, A, t, ridge=ridge, gamma=gam(t), equal_months=equal_months,
                                   u_in_feature_block=u_in_feature_block) for t in idx_fit]
            feats_tr = [f for pr in pairs for f in pr]
            qp = FistaQP(st, A, [t for t in idx_fit for _ in range(2)], kappa=kappa, feats=feats_tr)
            feats_val = lambda idx_val: [gram_features(st, A, t, ridge=ridge, gamma=gam(t), equal_months=equal_months,
                                                       u_in_feature_block=u_in_feature_block) for t in idx_val]
        elif weight == "frob":
            w_f = [1.0 if month_weight == "one" else st.w[t] for t in idx_fit]
            qp = FistaQP(st, A, idx_fit, kappa=kappa, wts=w_f)
            feats_val = lambda idx_val: [plain_features(st, A, t) for t in idx_val]
        else:
            raise ValueError(weight)
        if cone_shift > 0 or cone_shifts is not None:
            _, u1, _, Xi = stage1(st, A, idx_fit, ridge)
            Fh = floor_matrix(u1, Xi, 1.0, floor_shape)

            def shifted(g, qp0=qp):
                S = np.zeros((m + 1, m + 1))
                S[1:, 1:] = g * Fh
                return _Shifted(qp0, S)
            qp = shifted(cone_shift) if cone_shifts is None else shifted
        return A, resid_trace, qp, feats_val

    def val_loss(P, idx_val, feats_val):
        return float(np.mean([st.w[t] * (sum(month_loss(P, g, kappa) for g in f) if isinstance(f, tuple)
                                         else (month_loss_aug(P, f) if "Maug" in f else month_loss(P, f, kappa)))
                              for t, f in zip(idx_val, feats_val)]))

    for t0 in t0s:
        idx_tr = list(range(t0, t0 + train))
        idx_val = list(range(t0 + train, t0 + train + nval))
        t_te = t0 + train + nval
        tt = int(st.t[t_te])
        s2 = benchmark_sigma2(st, idx_tr)
        for m in ms:
            A, resid_trace, qp, fv = fit(idx_tr, m)
            feats_val = fv(idx_val)
            grid = [(None, lam) for lam in lams] if cone_shifts is None else [(g, lam) for g in cone_shifts for lam in lams]
            recs = []
            for g, lam in grid:
                P = qp.solve(lam) if g is None else qp(g).solve(lam)
                if P.u <= 1e-12:
                    print(f"  t={tt} m={m} lam={lam}: u^id = 0, skipped", file=sys.stderr)
                    continue
                evs = [evaluate_month(P, A, st, t, s2) for t in idx_val]
                ev_te = evaluate_month(P, A, st, t_te, s2)
                rec = _rec(st, t_te, m=m, lam=lam, val_score=float(np.mean([e["score"] for e in evs])),
                           val_gls=val_loss(P, idx_val, feats_val),
                           train_loss=(qp.loss(P) if g is None else np.nan), resid_trace=resid_trace,
                           status=P.status, iters=P.iters)
                if g is not None:                       # the number of factors: rank of the excess over the floor
                    Ex = P.V - np.outer(P.b, P.b) - qp(g).S[1:, 1:]
                    rec.update(gamma=g, thr=lam / 2,
                               nfac_ex=int(np.sum(np.linalg.eigvalsh(0.5 * (Ex + Ex.T)) > 1e-8 * max(1.0, np.abs(Ex).max()))))
                rec.update({k: ev_te[k] for k in SCALARS})
                recs.append(rec)
            if not recs:
                continue
            best = int(np.argmin([r["val_score"] for r in recs]))
            for k, rec in enumerate(recs):
                rec["selected"] = int(k == best)
                rec["refit"] = 0
                rows.append(rec)
            if refit:
                idx_all = idx_tr + idx_val
                A2, resid2, qp2, _ = fit(idx_all, m)
                s2_all = benchmark_sigma2(st, idx_all)
                lam_sel, g_sel = recs[best]["lam"], recs[best].get("gamma")
                if cone_shifts is None:
                    modes = [("refit_sel", None, lam_sel)] + ([("refit_lam0", None, 0.0)] if lam_sel != 0.0 else [])
                else:
                    modes = [("refit_sel", g_sel, lam_sel)] + [(f"refit_g{g}_l0", g, 0.0) for g in (0.0, 1.0)
                                                               if (g, 0.0) != (g_sel, lam_sel) and g in cone_shifts]
                for mode, g, lam in modes:
                    P2 = qp2.solve(lam) if g is None else qp2(g).solve(lam)
                    if P2.u <= 1e-12:
                        print(f"  t={tt} m={m} refit lam={lam}: u^id = 0, skipped", file=sys.stderr)
                        continue
                    ev2 = evaluate_month(P2, A2, st, t_te, s2_all)
                    rec2 = _rec(st, t_te, m=m, lam=lam, val_score=recs[best]["val_score"], val_gls=recs[best]["val_gls"],
                                train_loss=(qp2.loss(P2) if g is None else np.nan), resid_trace=resid2,
                                status=P2.status, iters=P2.iters)
                    rec2.update({k: ev2[k] for k in SCALARS})
                    rec2["selected"] = int(mode == "refit_sel")
                    rec2["refit"] = 1
                    if g is not None:
                        rec2.update(gamma=g, thr=lam / 2)
                    rows_refit.append(rec2)
        if log:
            log(f"test month {tt} ({st.date0[t_te]})")
    return rows, rows_refit


# ------------------------------------------------------------------------------------------------- IPCA
def run_ipca(st, t0s, K, train=96, nval=12, use_val=True, ridge=1e-4, max_iter=200, gmv=False, models=None, log=None):
    """IPCA with K factors, estimated on the train + nval months before the test month (use_val) or on the training
    months only; every window starts from the loadings of the previous window and from the KPS start. models: optional
    dict that receives the fitted models for pass 2. Returns rows."""
    from .ipca import evaluate_ipca, fit_window
    rows = []
    prev = None
    ntr = train + (nval if use_val else 0)
    for t0 in t0s:
        idx_tr = list(range(t0, t0 + ntr))
        t_te = t0 + train + nval
        s2 = benchmark_sigma2(st, idx_tr)
        P = fit_window(st, idx_tr, K, Gamma0=prev, max_iter=max_iter, cold_start=True, ridge=ridge)
        prev = P.Gamma
        ev = evaluate_ipca(P, st, t_te, s2, gmv=gmv)
        rec = _rec(st, t_te, m=K, lam=-1.0, val_score=np.nan, train_loss=P.sse, status=P.status, iters=P.iters)
        rec.update({k: ev[k] for k in SCALARS})
        rec.update(ret_tan=ev["ret_tan"], tot_r2_train=ev["tot_r2_train"], selected=1)
        if gmv:
            rec.update({k: ev[k] for k in GMV_COLUMNS})
        rows.append(rec)
        if models is not None:
            tt = rec["t"]
            models[f"A_m{K}_t{tt}"] = P.Gamma.astype(np.float32)
            models[f"s2_t{tt}"] = np.array([s2])
            models[f"sel_m{K}_t{tt}_b"] = P.b
            models[f"sel_m{K}_t{tt}_V"] = P.V
            models[f"sel_m{K}_t{tt}_u"] = np.array([P.u])
        if log:
            log(f"test month {int(st.t[t_te])} ({st.date0[t_te]}), K = {K}")
    return rows


# ------------------------------------------------------------------------------------------------- Frobenius path
LAMS_PATH = [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]


def run_path(st, t0s, ms, lams=LAMS_PATH, train=96, nval=12, kappa=1.0, log=None):
    """The Frobenius loss with month weight 1/N_t on a lambda grid, no re-estimation (the isotropic reference of the
    idiosyncratic specification). Returns rows."""
    rows = []
    for t0 in t0s:
        idx_tr = list(range(t0, t0 + train))
        idx_val = list(range(t0 + train, t0 + train + nval))
        t_te = t0 + train + nval
        s2 = benchmark_sigma2(st, idx_tr)
        for m in ms:
            A, ev = window_basis(st, idx_tr, m)
            resid_trace = float(ev[m:].sum() / ev.sum())
            qp = FistaQP(st, A, idx_tr, kappa=kappa)
            recs = []
            for lam in lams:
                P = qp.solve(lam)
                vs = float(np.mean([evaluate_month(P, A, st, t, s2)["score"] for t in idx_val]))
                ev_te = evaluate_month(P, A, st, t_te, s2)
                rec = _rec(st, t_te, m=m, lam=lam, val_score=vs, train_loss=qp.loss(P), resid_trace=resid_trace,
                           status=P.status, iters=P.iters)
                rec.update({k: ev_te[k] for k in SCALARS})
                recs.append(rec)
            best = int(np.argmin([r["val_score"] for r in recs]))
            for k, rec in enumerate(recs):
                rec["selected"] = int(k == best)
                rows.append(rec)
        if log:
            log(f"test month {int(st.t[t_te])} ({st.date0[t_te]})")
    return rows


# ------------------------------------------------------------------------------------------------- idiosyncratic block
def run_idio(st, ids, t0s, ms, lam_sel: dict, train=96, nval=12, log=None):
    """The joint program in (U^sy, U^id) at lambda^sy = 0 ("lam0") and at the lambda^sy selected for the isotropic model
    of run_path in the same window ("sel"), next to the isotropic model; no re-estimation. Returns rows."""
    from .idio import FistaQPId, r22_terms
    rows = []
    prev = {}
    for t0 in t0s:
        idx_tr = list(range(t0, t0 + train))
        t_te = t0 + train + nval
        tt = int(st.t[t_te])
        s2 = benchmark_sigma2(st, idx_tr)
        for m in ms:
            A, _ = window_basis(st, idx_tr, m)
            qp = FistaQPId(st, ids, A, idx_tr)
            qiso = FistaQP(st, A, idx_tr)
            lam_s = lam_sel.get((tt, m), 0.0)
            P0 = P0iso = None
            for mode, lam in (("lam0", 0.0), ("sel", lam_s)):
                if mode == "sel" and lam == 0.0:
                    P, Piso = P0, P0iso
                else:
                    P = qp.solve(lam, U0=prev[m][0], Uid0=prev[m][1]) if (mode == "lam0" and m in prev) else qp.solve(lam)
                    Piso = qiso.solve(lam)
                if mode == "lam0":
                    P0, P0iso = P, Piso
                    prev[m] = (P.U, P.Uid)
                ev = r22_terms(P, A, st, ids, t_te, s2)
                rec = _rec(st, t_te, m=m, mode=mode, lam=lam, status=P.status, iters=P.iters, loss_id=qp.loss(P),
                           loss_iso=qiso.loss(Piso), rank_C=P.rank_C(), rank_C_iso=Piso.rank_C(), rank_id=P.rank_id(),
                           u_iso=Piso.u, u00=P.Uid[0, 0])
                for i in range(ids.p):
                    for j in range(i, ids.p):
                        rec[f"Uid_{ids.names[i]}_{ids.names[j]}"] = P.Uid[i, j]
                rec.update(ev)
                rows.append(rec)
        if log:
            log(f"test month {tt} ({st.date0[t_te]})")
    return rows
