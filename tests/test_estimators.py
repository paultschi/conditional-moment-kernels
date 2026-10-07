"""The CMK estimator (Proposition 4.2) and the solver of Appendix B against an interior-point solver (Clarabel via cvxpy)."""
import numpy as np
import pytest

from cmk.cmk import stage1, stage2
from cmk.model import month_features, window_basis
from cmk.solver import FistaQP, project_psd_fixed00

cp = pytest.importorskip("cvxpy")
TOL = dict(tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12)


def frobenius_qp(stats, A, idx):
    """The Frobenius-weight objective of the isotropic CMK program in vectorized form (Lemma 3.3), month weight 1/N_t."""
    m = A.shape[1]
    n = m + 1
    p = n * n
    H = np.zeros((p + 1, p + 1))
    lin = np.zeros(p + 1)
    const = 0.0
    e0 = np.zeros(n)
    e0[0] = 1.0
    for t in idx:
        Mt, st, N, _, x2 = month_features(stats, A, t)
        w = stats.w[t]
        M = np.zeros((n, n))
        M[0, 0] = 1.0
        M[1:, 1:] = Mt
        s = np.concatenate(([1.0], st))
        D = M - np.outer(e0, e0)
        H[:p, :p] += w * np.kron(M, M)
        H[:p, p] += w * D.ravel()
        H[p, :p] += w * D.ravel()
        H[p, p] += w * N
        lin[:p] += -2.0 * w * np.outer(s, s).ravel()
        lin[p] += -2.0 * w * x2
        const += w * (1.0 + x2) ** 2
    T = len(idx)
    return 0.5 * (H + H.T) / T, lin / T, const / T


@pytest.mark.parametrize("lam", [0.0, 1e-3, 1e-2])
def test_solver_against_clarabel(panel, lam):
    idx = list(range(24))
    A, _ = window_basis(panel, idx, 6)
    n = A.shape[1] + 1
    H, lin, const = frobenius_qp(panel, A, idx)
    ev, V = np.linalg.eigh(H)
    L = (V * np.sqrt(np.clip(ev, 0, None))).T
    U = cp.Variable((n, n), symmetric=True)
    u = cp.Variable(nonneg=True)
    v = cp.hstack([cp.vec(U, order="F"), u])
    prob = cp.Problem(cp.Minimize(cp.sum_squares(L @ v) + lin @ v + const + lam * cp.trace(U)), [U >> 0, U[0, 0] == 1])
    prob.solve(solver="CLARABEL", **TOL)
    P = FistaQP(panel, A, idx, tol=1e-10).solve(lam)
    vv = np.concatenate([P.U.ravel(), [P.u]])
    obj = vv @ H @ vv + lin @ vv + const + lam * np.trace(P.U)
    assert abs(obj - prob.value) <= 1e-8 * max(1.0, abs(prob.value))
    assert np.abs(P.U - U.value).max() < 1e-5


def test_projection():
    """The projection onto {U PSD, U_00 = c} is the nearest point: compare with cvxpy."""
    rng = np.random.default_rng(3)
    Y = rng.normal(size=(5, 5))
    Y = 0.5 * (Y + Y.T)
    P, nu, r = project_psd_fixed00(Y, 1.0)
    X = cp.Variable((5, 5), symmetric=True)
    cp.Problem(cp.Minimize(cp.sum_squares(X - Y)), [X >> 0, X[0, 0] == 1]).solve(solver="CLARABEL", **TOL)
    assert np.abs(P - X.value).max() < 1e-6


@pytest.mark.parametrize("eta,theta", [(0.0, 0.0), (1.0, 0.0), (0.5, 1e-4), (2.0, 3e-4)])
def test_cmk_estimator_solves_the_reduced_program(panel, eta, theta):
    """Proposition 4.2: the excess is argmin_{C~ PSD} ||C~ - (Abar - b b' - eta F^)||_F^2 + lambda tr C~, lambda = 2 theta."""
    idx = list(range(24))
    A, _ = window_basis(panel, idx, 6)
    b, u, Abar, Xi = stage1(panel, A, idx, 1e-3)
    P = stage2(b, u, Abar, Xi, eta, theta)
    target = Abar - np.outer(b, b) - eta * u * Xi
    sc = np.abs(target).max()                              # solve the program in units of the target
    m = len(b)
    Ct = cp.Variable((m, m), symmetric=True)
    cp.Problem(cp.Minimize(cp.sum_squares(Ct - target / sc) + 2 * theta / sc * cp.trace(Ct)), [Ct >> 0]).solve(solver="CLARABEL", **TOL)
    assert np.abs((P.C - eta * u * Xi) / sc - Ct.value).max() < 1e-6
    assert np.allclose(P.b, b) and P.u == u


def test_cmk_estimator_without_regularization(panel):
    """For rho -> 0, eta = 0, lambda = 0 in months of full rank: b = mean f, C = P_+(mean f f' - u^ mean G^{-1} - b b')."""
    idx = list(range(24))
    A, _ = window_basis(panel, idx, 6)
    P = stage2(*stage1(panel, A, idx, 0.0), 0.0, 0.0)
    F = []
    res, dof, Gi = 0.0, 0.0, 0.0
    for t in idx:
        Mt, st, N, _, x2 = month_features(panel, A, t)
        f = np.linalg.solve(Mt, st)
        F.append(f)
        res += x2 - st @ f
        dof += N - 6
        Gi = Gi + np.linalg.inv(Mt)
    F = np.array(F)
    u = res / dof
    C = F.T @ F / len(idx) - u * Gi / len(idx) - np.outer(F.mean(0), F.mean(0))
    ev, Q = np.linalg.eigh(0.5 * (C + C.T))
    C = (Q * np.maximum(ev, 0)) @ Q.T
    assert np.allclose(P.b, F.mean(0), rtol=1e-10, atol=1e-14)
    assert abs(P.u - u) < 1e-12 * u
    assert np.abs(P.C - C).max() < 1e-10 * np.abs(C).max()
