"""Active learning with a GP surrogate on a precomputed Gram matrix.

The loop is identical for every kernel: start from `n_init` random
labelled molecules, fit a GP, pick the unlabelled molecule with the
largest predictive variance (uncertainty sampling), add it, repeat.
We track the RMSE on the still-unlabelled molecules after each step,
which is the "labels-to-accuracy" curve the project is about.
"""
import numpy as np


class KernelGP:
    """Exact GP regression with a fixed Gram matrix and a noise term.

    The signal scale is fitted by maximising the marginal likelihood on a
    1-D grid; that keeps all kernels on an equal footing.
    """

    def __init__(self, K: np.ndarray, noise: float = 1e-4):
        self.K = K
        self.noise = noise

    def fit(self, idx: np.ndarray, y: np.ndarray):
        self.idx = np.asarray(idx)
        self.mu, self.sd = y.mean(), y.std() + 1e-12
        yn = (y - self.mu) / self.sd
        Kii = self.K[np.ix_(self.idx, self.idx)]
        best = None
        for scale in np.logspace(-2, 2, 17):
            A = scale * Kii + self.noise * np.eye(len(self.idx))
            try:
                L = np.linalg.cholesky(A)
            except np.linalg.LinAlgError:
                continue
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, yn))
            lml = -0.5 * yn @ alpha - np.log(np.diag(L)).sum()
            if best is None or lml > best[0]:
                best = (lml, scale, L, alpha)
        _, self.scale, self.L, self.alpha = best
        return self

    def predict(self, idx: np.ndarray):
        idx = np.asarray(idx)
        Ks = self.scale * self.K[np.ix_(idx, self.idx)]
        mean = Ks @ self.alpha
        v = np.linalg.solve(self.L, Ks.T)
        var = self.scale * np.diag(self.K)[idx] - (v ** 2).sum(0)
        var = np.clip(var, 1e-12, None)
        return mean * self.sd + self.mu, np.sqrt(var) * self.sd


def active_learning_curve(K: np.ndarray, y: np.ndarray, n_init: int = 5,
                          n_final: int = 25, seed: int = 0,
                          strategy: str = "uncertainty") -> np.ndarray:
    """Return RMSE on unlabelled molecules after each acquisition step."""
    rng = np.random.default_rng(seed)
    n = len(y)
    labelled = list(rng.choice(n, n_init, replace=False))
    rmse = []
    while True:
        unl = np.setdiff1d(np.arange(n), labelled)
        gp = KernelGP(K).fit(np.array(labelled), y[labelled])
        mean, sd = gp.predict(unl)
        rmse.append(float(np.sqrt(((mean - y[unl]) ** 2).mean())))
        if len(labelled) >= n_final:
            break
        if strategy == "uncertainty":
            pick = unl[np.argmax(sd)]
        else:  # random baseline
            pick = rng.choice(unl)
        labelled.append(int(pick))
    return np.array(rmse)
