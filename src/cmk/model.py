"""A fitted moment kernel and its out-of-sample evaluation on the sufficient statistics (Sections 3.3, 6 and 7.2).

Notation of the paper. For the isotropic specification a parameter is U = (U^sy, u^id) with
    U^sy = [[1, b'], [b, V]]  ((m+1) x (m+1) PSD),   C = V - b b',   u = u^id >= 0,
and on a cross section with features phi_t (N_t x m) the conditional moments are
    mu_t = phi_t b,   Sigma_t = phi_t C phi_t' + u I,   Q_t = Sigma_t + mu_t mu_t'.
The features of a window are phi_t = Phi_t A for the raw feature matrix Phi_t of the statistics file and a window basis A
(d x m, orthonormal columns: the m dominant eigenvectors of the pooled Gram matrix of the window, Section 7.1), so that
    M_t = phi_t' phi_t = A' G_t A,   s_t = phi_t' x_{t+1} = A' s^raw_t.
Every quantity of Section 7 follows from (M_t, s_t, N_t, 1'x_{t+1}, ||x_{t+1}||^2) by the Woodbury identity, without forming
an N_t x N_t matrix (Online Appendix D.1, "Selection and measures").
"""
from __future__ import annotations

import numpy as np


class Params:
    """U^sy and u^id of the isotropic specification, with b, V and C = V - b b'."""

    def __init__(self, U: np.ndarray, u: float, status: str = "optimal"):
        self.U = U
        self.u = u
        self.status = status
        self.b = U[1:, 0].copy()
        self.V = U[1:, 1:].copy()
        self.C = self.V - np.outer(self.b, self.b)
        self.C = 0.5 * (self.C + self.C.T)

    def rank_C(self, rtol: float = 1e-6, atol: float = 1e-12) -> int:
        """rank(V - b b'): eigenvalues below rtol times the largest or below atol count as zero."""
        ev = np.linalg.eigvalsh(self.C)
        return int((ev > max(rtol * ev.max(), atol)).sum()) if ev.size else 0


def from_moments(b: np.ndarray, V: np.ndarray, u: float, status: str = "closed-form") -> Params:
    m = len(b)
    U = np.zeros((m + 1, m + 1))
    U[0, 0] = 1.0
    U[1:, 0] = U[0, 1:] = b
    U[1:, 1:] = V
    return Params(U, u, status)


def window_basis(stats, idx, m: int):
    """Orthonormal basis A (d x m) of the m dominant eigenvectors of sum_{t in idx} G_t, and all eigenvalues (descending)."""
    Gsum = stats.G[idx].sum(axis=0)
    ev, V = np.linalg.eigh(Gsum)
    order = np.argsort(ev)[::-1]
    return V[:, order[:m]], ev[order]


def month_features(stats, A: np.ndarray, t: int):
    """(M_t, s_t, N_t, 1'x_{t+1}, ||x_{t+1}||^2) of month index t under the window basis A."""
    Mt = A.T @ stats.G[t] @ A
    Mt = 0.5 * (Mt + Mt.T)
    st = A.T @ stats.s[t]
    return Mt, st, stats.N[t], stats.sumx[t], stats.sumx2[t]


def benchmark_sigma2(stats, idx) -> float:
    """u^bm of the purely idiosyncratic benchmark: the average of ||x_{t+1}||^2 / N_t over the months idx."""
    return float((stats.w[idx] * stats.sumx2[idx]).sum() / (stats.w[idx] * stats.N[idx]).sum())


def _sym(A):
    return 0.5 * (A + A.T)


# columns of the monthly result files written by cmk.runs (one row per test month and model)
SCALARS = ["N", "r2_num", "r2_den", "r22_num", "r22_den", "score", "score_bm", "tot_r2", "rho_f",
           "theta", "theta_frac", "pred_sr_ann", "ret_sdf", "ret_cmve", "ret_ew", "rank_C",
           "rank_phi", "sigma_min_phi", "u_id", "mean_norm2", "trC", "trHC"]
GMV_COLUMNS = ["r_gmv", "pv_gmv", "pv_sdf", "score0"]


def evaluate_month(P: Params, A: np.ndarray, stats, t: int, sigma2_bm: float, rank_rtol: float = 1e-10,
                   gmv: bool = False) -> dict:
    """The out-of-sample quantities of Section 7 for month index t (features at t, returns over t+1).

      r2_num/r2_den, r22_num/r22_den   the summands of the predictive R^2 and R^{2,2} (weight 1/N_t)
      score, score_bm                  the Dawid-Sebastiani score of the model and of the benchmark (0, u^bm I)
      tot_r2, rho_f                    total R^2 of the factor portfolios f = phi^+ x and the projected variance share
      theta, theta_frac, pred_sr_ann   mu' Sigma^+ mu, theta / (1 + theta), sqrt(12 theta) (Proposition 6.1)
      ret_sdf, ret_cmve, ret_ew        returns of the UMVE portfolio Q^+ mu, the CMVE portfolio Sigma^+ mu and 1/N_t
      rank_C, rank_phi, ...            diagnostics
    gmv = True adds the covariance forecasts of Online Appendix D.1 (needs the constant as the first raw feature):
      r_gmv = 1'S^{-1}x / 1'S^{-1}1    return of the GMV portfolio S^{-1}1 / 1'S^{-1}1
      pv_gmv = 1 / 1'S^{-1}1           its predicted variance
      pv_sdf                           predicted variance of the UMVE return
      score0                           the score with the mean set to zero (score_bm - score0 is the covariance part)
    """
    Mt, st, N, sumx, x2 = month_features(stats, A, t)
    m = A.shape[1]
    b, V, C, u = P.b, P.V, P.C, P.u
    I = np.eye(m)
    if u <= 0:
        raise ValueError("u^id = 0: conditional covariance singular, scoring rule undefined")
    w = 1.0 / N
    # first moment
    res2 = x2 - 2.0 * b @ st + b @ Mt @ b                       # ||x - mu||^2
    r2_num, r2_den = w * res2, w * x2
    # second moment
    MV = Mt @ V
    r22_num = w * (x2 ** 2 - 2.0 * st @ V @ st - 2.0 * u * x2 + np.trace(MV @ MV)
                   + 2.0 * u * np.trace(MV) + u ** 2 * N)
    r22_den = w * (x2 ** 2 - 2.0 * sigma2_bm * x2 + sigma2_bm ** 2 * N)
    # scoring rule S = logdet Sigma + r' Sigma^{-1} r, Sigma = phi C phi' + u I
    MC = Mt @ C
    K_C = np.linalg.solve(u * I + MC, I)                        # (uI + MC)^{-1}
    CK = _sym(C @ K_C)                                          # C (uI + MC)^{-1}
    sign, logdet_small = np.linalg.slogdet(u * I + MC)
    logdet = (N - m) * np.log(u) + logdet_small
    phir = st - Mt @ b                                          # phi'(x - mu)
    quad = (res2 - phir @ CK @ phir) / u
    score = logdet + quad
    score_bm = N * np.log(sigma2_bm) + x2 / sigma2_bm
    # factor portfolios f = phi^+ x
    evM = np.linalg.eigvalsh(Mt)
    rank_phi = int((evM > rank_rtol * max(evM.max(), 1e-300)).sum())
    Mp = np.linalg.pinv(Mt, rcond=1e-10, hermitian=True)
    tot_r2 = 1.0 - (x2 - st @ Mp @ st) / x2
    sigma_min_phi = float(np.sqrt(max(evM.min(), 0.0)))
    trMC = float(np.trace(MC))
    rho_f = (trMC + u * rank_phi) / (trMC + u * N)
    # conditional Sharpe ratio, UMVE and CMVE portfolios: phi' Sigma^{-1} y = (phi'y - M C (uI + MC)^{-1} phi'y) / u
    Bmat = (Mt - Mt @ CK @ Mt) / u                              # phi' Sigma^{-1} phi
    theta = float(b @ Bmat @ b)
    g_cmve = (b - CK @ Mt @ b) / u                              # Sigma^{-1} mu = phi g_cmve
    K_V = np.linalg.solve(u * I + Mt @ V, I)
    VK = _sym(V @ K_V)
    g_sdf = (b - VK @ Mt @ b) / u                               # Q^{-1} mu = phi g_sdf
    ret_cmve = float(g_cmve @ st)
    ret_sdf = float(g_sdf @ st)
    ret_ew = sumx / N
    out = dict(
        N=N, r2_num=r2_num, r2_den=r2_den, r22_num=r22_num, r22_den=r22_den,
        score=score, score_bm=score_bm, tot_r2=tot_r2, rho_f=rho_f,
        theta=theta, theta_frac=theta / (1.0 + theta), pred_sr_ann=np.sqrt(12.0 * theta),
        ret_sdf=ret_sdf, ret_cmve=ret_cmve, ret_ew=ret_ew,
        rank_C=P.rank_C(), rank_phi=rank_phi, sigma_min_phi=sigma_min_phi, u_id=u,
        mean_norm2=float(b @ b), trC=float(np.trace(C)), trHC=float(np.trace(Bmat @ C)),
        gamma_sdf=A @ g_sdf, gamma_cmve=A @ g_cmve,
    )
    if gmv:
        c = A.T @ stats.G[t][:, 0]                                # phi'1 = A' Phi_t'1 (raw feature 0 = constant)
        one_S_one = (N - c @ CK @ c) / u
        one_S_x = (sumx - c @ CK @ st) / u
        out.update(r_gmv=float(one_S_x / one_S_one), pv_gmv=float(1.0 / one_S_one),
                   pv_sdf=float(g_sdf @ (MC @ Mt + u * Mt) @ g_sdf),
                   score0=float(logdet + (x2 - st @ CK @ st) / u))
    return out


def score_month(P: Params, feats) -> float:
    """evaluate_month(...)["score"] alone, from precomputed month_features (the validation score of every grid point);
    the floating-point operations are those of evaluate_month, so the value is the same."""
    Mt, st, N, sumx, x2 = feats
    m = Mt.shape[0]
    b, C, u = P.b, P.C, P.u
    I = np.eye(m)
    if u <= 0:
        raise ValueError("u^id = 0: conditional covariance singular, scoring rule undefined")
    res2 = x2 - 2.0 * b @ st + b @ Mt @ b
    MC = Mt @ C
    K_C = np.linalg.solve(u * I + MC, I)
    CK = _sym(C @ K_C)
    sign, logdet_small = np.linalg.slogdet(u * I + MC)
    logdet = (N - m) * np.log(u) + logdet_small
    phir = st - Mt @ b
    quad = (res2 - phir @ CK @ phir) / u
    return logdet + quad
