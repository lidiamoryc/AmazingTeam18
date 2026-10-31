"""XY-mixer QAOA (Quantum Alternating Operator Ansatz, Hadfield et al. 2019).

Initial state: Dicke state |D^n_K>, the uniform superposition of every plan
with exactly K sites (Wang et al., PRA 101, 012320 (2020) use XY mixers with
W/Dicke-type states). Prep cost O(Kn) gates, depth O(n) (Baertschi &
Eidenbenz 2019); NOT included in the gate counts below.

Layer:  exp(-i gamma E)  then the XY mixer  prod exp(-i beta (XX+YY)/2).
Every XX+YY term moves one crew between two sites and never adds or removes
one, so the state never leaves the K-site subspace. We simulate inside that
C(n,K)-dimensional subspace exactly -- the same physics as a 2^n statevector,
checked against Qiskit in tests/test_quantum.py.

E is the cost to MINIMISE: E = -(Q - mean_F Q) / std_F Q over feasible plans F.
Normalising this way lets one set of angles transfer between instances,
following Shaydulin et al., Sci. Adv. 10, eadm6761 (2024).
"""
import itertools

import numpy as np
from scipy.optimize import minimize


class Subspace:
    def __init__(self, n, K, mixer="ring"):
        self.n, self.K = n, K
        C = np.array(list(itertools.combinations(range(n), K)), dtype=np.int64)
        self.C = C
        self.masks = (np.int64(1) << C).sum(1)
        order = np.argsort(self.masks)
        self.C, self.masks = self.C[order], self.masks[order]
        self.dim = len(self.masks)
        if mixer == "ring":
            even = [(i, i + 1) for i in range(0, n - 1, 2)]
            odd = [(i, i + 1) for i in range(1, n - 1, 2)]
            pairs = even + odd + ([(n - 1, 0)] if n > 2 else [])
        elif mixer == "clique":
            pairs = list(itertools.combinations(range(n), 2))
        else:
            raise ValueError(mixer)
        self.pairs = pairs
        self.maps = []
        for i, j in pairs:
            bi, bj = (self.masks >> i) & 1, (self.masks >> j) & 1
            a = np.flatnonzero((bi == 1) & (bj == 0))
            partner = self.masks[a] ^ ((np.int64(1) << i) | (np.int64(1) << j))
            b = np.searchsorted(self.masks, partner)
            self.maps.append((a, b))

    def dicke(self):
        return np.full(self.dim, 1 / np.sqrt(self.dim), dtype=complex)

    def evolve(self, E, gammas, betas, psi=None):
        psi = self.dicke() if psi is None else psi.copy()
        for g, b in zip(gammas, betas):
            psi *= np.exp(-1j * g * E)
            c, s = np.cos(b), -1j * np.sin(b)
            for a, bb in self.maps:
                pa, pb = psi[a], psi[bb]
                psi[a] = c * pa + s * pb
                psi[bb] = s * pa + c * pb
        return psi


def normalised_cost(Qvals):
    mu, sd = Qvals.mean(), Qvals.std()
    return -(Qvals - mu) / (sd if sd > 0 else 1.0)


def expectation(sub, E, x):
    p = len(x) // 2
    psi = sub.evolve(E, x[:p], x[p:])
    return float(np.abs(psi) ** 2 @ E)


def interp(x):
    """Zhou et al., PRX 10, 021067 (2020): lift depth-p angles to depth p+1."""
    p = len(x) // 2
    out = []
    for arr in (x[:p], x[p:]):
        new = np.empty(p + 1)
        for i in range(1, p + 2):
            lo = arr[i - 2] if i >= 2 else 0.0
            hi = arr[i - 1] if i <= p else 0.0
            new[i - 1] = (i - 1) / p * lo + (p - i + 1) / p * hi
        out.append(new)
    return np.concatenate(out)


def optimise(sub, E, p_max, rng, starts=8):
    """Per-instance angles for p = 1..p_max, INTERP warm-starts, L-BFGS-B."""
    fun = lambda x: expectation(sub, E, x)
    best = None
    for _ in range(starts):
        x0 = np.array([rng.uniform(0.05, 1.0), rng.uniform(0.05, 1.2)])
        r = minimize(fun, x0, method="L-BFGS-B")
        if best is None or r.fun < best.fun:
            best = r
    out = {1: best.x}
    x = best.x
    for p in range(2, p_max + 1):
        r = minimize(fun, interp(x), method="L-BFGS-B")
        x = r.x
        out[p] = x
    return out


def success_prob(sub, E, x, opt_mask):
    p = len(x) // 2
    psi = sub.evolve(E, x[:p], x[p:])
    return float((np.abs(psi) ** 2)[opt_mask].sum())


def tts_shots(p_opt):
    """Shaydulin et al. (2024): TTS = 1/p_opt expected shots to sample an optimum."""
    return 1.0 / max(p_opt, 1e-300)


def two_qubit_gates(n, K, p, n_pairs_nonzero, mixer="ring"):
    """Native 2-qubit gate count on all-to-all hardware (e.g. trapped ions).

    cost layer: one ZZ(theta) per site pair with r_ij != 0
    mixer:      one XY term = RXX + RYY = 2 two-qubit gates, per mixer pair
    Dicke-state preparation (O(Kn), Baertschi & Eidenbenz 2019) is excluded,
    so this is a lower bound.
    """
    mix_pairs = n if mixer == "ring" else n * (n - 1) // 2
    return p * (n_pairs_nonzero + 2 * mix_pairs)


def depolarised_success(p_opt, n_opt, n, N2q, eps2):
    """Global depolarising model: with fidelity F = (1-eps)^N2q the output is
    the ideal state, otherwise uniform over all 2^n bitstrings."""
    F = (1 - eps2) ** N2q
    return F * p_opt + (1 - F) * n_opt / 2 ** n, F
