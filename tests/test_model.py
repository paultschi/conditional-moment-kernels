"""The out-of-sample quantities computed from the sufficient statistics (Woodbury identity) against explicit N x N algebra."""
import numpy as np

from cmk.model import evaluate_month, from_moments, window_basis


def test_evaluate_month_against_dense_algebra(panel):
    rng = np.random.default_rng(0)
    A, _ = window_basis(panel, list(range(20)), 7)
    m = A.shape[1]
    b = rng.normal(scale=0.01, size=m)
    L = rng.normal(scale=0.05, size=(m, 3))
    C = L @ L.T
    u = 0.02
    P = from_moments(b, C + np.outer(b, b), u)
    t, s2 = 25, 0.03
    e = evaluate_month(P, A, panel, t, s2, gmv=True)
    Z, x = panel.Zs[t], panel.xs[t]
    N = len(x)
    phi = Z @ A
    S = phi @ C @ phi.T + u * np.eye(N)
    mu = phi @ b
    Q = S + np.outer(mu, mu)
    Si = np.linalg.inv(S)
    one = np.ones(N)
    r = x - mu
    w_gmv = Si @ one / (one @ Si @ one)
    f = np.linalg.lstsq(phi, x, rcond=None)[0]
    ref = dict(
        score=np.linalg.slogdet(S)[1] + r @ Si @ r,
        score_bm=N * np.log(s2) + x @ x / s2,
        r2_num=r @ r / N,
        r22_num=np.sum((np.outer(x, x) - Q) ** 2) / N,
        r22_den=np.sum((np.outer(x, x) - s2 * np.eye(N)) ** 2) / N,
        theta=mu @ Si @ mu,
        ret_cmve=mu @ Si @ x,
        ret_sdf=mu @ np.linalg.solve(Q, x),
        tot_r2=1 - np.sum((x - phi @ f) ** 2) / (x @ x),
        r_gmv=w_gmv @ x,
        pv_gmv=w_gmv @ S @ w_gmv,
        score0=np.linalg.slogdet(S)[1] + x @ Si @ x,
    )
    wq = np.linalg.solve(Q, mu)
    ref["pv_sdf"] = wq @ S @ wq
    for k, v in ref.items():
        assert abs(e[k] - v) <= 1e-8 * max(abs(v), 1e-12), (k, e[k], v)


def test_umve_identities(panel):
    """Proposition 6.1: the UMVE weights are Sigma^+ mu / (1 + theta)."""
    rng = np.random.default_rng(2)
    A, _ = window_basis(panel, list(range(20)), 5)
    b = rng.normal(scale=0.01, size=5)
    C = np.diag(rng.uniform(0.001, 0.01, size=5))
    e = evaluate_month(from_moments(b, C + np.outer(b, b), 0.01), A, panel, 21, 0.02)
    assert abs(e["ret_sdf"] - e["ret_cmve"] / (1 + e["theta"])) < 1e-12 * max(1.0, abs(e["ret_cmve"]))
