import numpy as np
import pytest


class RawStats:
    """Sufficient statistics of a small simulated panel with a constant first feature (the interface of cmk.stats.Stats)."""

    def __init__(self, Zs, xs):
        self.Zs, self.xs = Zs, xs
        self.G = np.array([Z.T @ Z for Z in Zs])
        self.s = np.array([Z.T @ x for Z, x in zip(Zs, xs)])
        self.N = np.array([len(x) for x in xs], float)
        self.sumx = np.array([x.sum() for x in xs])
        self.sumx2 = np.array([x @ x for x in xs])
        self.T, self.d = len(xs), Zs[0].shape[1]
        self.w = 1.0 / self.N
        self.t = np.arange(1, self.T + 1)
        self.date0 = np.array([f"{2000 + k // 12}-{1 + k % 12:02d}-28" for k in range(self.T)])


@pytest.fixture(scope="session")
def panel():
    """30 months, 250 to 350 stocks, 12 features (constant + 11 characteristics), a three-factor model."""
    rng = np.random.default_rng(1)
    d = 12
    b = rng.normal(scale=0.003, size=d)
    L = rng.normal(scale=0.02, size=(d, 3))
    Zs, xs = [], []
    for t in range(30):
        N = int(rng.integers(250, 351))
        Z = np.column_stack([np.ones(N), rng.uniform(-1, 1, size=(N, d - 1))])
        x = Z @ (b + L @ rng.normal(size=3)) + 0.1 * rng.normal(size=N)
        Zs.append(Z)
        xs.append(x)
    return RawStats(Zs, xs)
