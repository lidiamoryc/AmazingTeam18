"""Classical solvers for choosing exactly K sampling sites.

QUBO solvers (maximise Q(x) = v.x - lam * sum_{i<j} r_ij x_i x_j, sum x = K):
  greedy_qubo        Nemhauser-Wolsey-Fisher greedy
  local_search       best-improvement swaps from a start set
  sim_anneal         simulated annealing over swap moves (Kirkpatrick et al. 1983)
  tabu               tabu search over swap moves (Glover 1989)
  brute_qubo         exhaustive, vectorised, exact
  mip_qubo           exact MIP with HiGHS (via scipy.optimize.milp), linearised pairs

True-objective solvers (maximise D(A), weighted pair coverage):
  greedy_true        lazy greedy; carries the 1 - 1/e guarantee (D is monotone submodular)
  mip_true           exact MIP with HiGHS
  chama_true         the same model through Sandia's Chama, as the judges suggested
"""
import heapq
import itertools
import math
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, csr_matrix

from . import objective as ob


# ----------------------------------------------------------------- QUBO side
class Qubo:
    def __init__(self, v, r, lam=1.0):
        self.v = np.asarray(v, float)
        self.R = lam * np.asarray(r, float)
        self.n = len(v)

    def value(self, A):
        A = list(A)
        return float(self.v[A].sum() - np.triu(self.R[np.ix_(A, A)], 1).sum()) if A else 0.0


def greedy_qubo(q, K):
    A, g = [], np.zeros(q.n)               # g_j = sum_{k in A} R_jk
    for _ in range(K):
        gain = q.v - g
        gain[A] = -np.inf
        j = int(np.argmax(gain))
        A.append(j)
        g += q.R[:, j]
    return sorted(A)


def local_search(q, A, max_iter=10_000):
    A = list(A)
    inA = np.zeros(q.n, bool); inA[A] = True
    g = q.R[:, A].sum(1)
    for _ in range(max_iter):
        out_ids = np.where(inA)[0]
        in_ids = np.where(~inA)[0]
        # delta(i out, j in) = v_j - v_i - (g_j - R_ji) + g_i
        d = (q.v[in_ids][None, :] - q.v[out_ids][:, None]
             - (g[in_ids][None, :] - q.R[np.ix_(out_ids, in_ids)]) + g[out_ids][:, None])
        a, b = np.unravel_index(np.argmax(d), d.shape)
        if d[a, b] <= 1e-12:
            break
        i, j = out_ids[a], in_ids[b]
        inA[i], inA[j] = False, True
        g += q.R[:, j] - q.R[:, i]
    return sorted(np.where(inA)[0].tolist())


def random_set(n, K, rng):
    return sorted(rng.choice(n, size=K, replace=False).tolist())


def sim_anneal(q, K, rng, sweeps=200, T0=None, T1=None, A0=None):
    A = list(A0) if A0 is not None else random_set(q.n, K, rng)
    inA = np.zeros(q.n, bool); inA[A] = True
    g = q.R[:, A].sum(1)
    cur = q.value(A)
    scale = np.median(np.abs(q.v)) + 1e-12
    T0 = T0 if T0 is not None else 0.5 * scale
    T1 = T1 if T1 is not None else 1e-3 * scale
    steps = max(1, sweeps * q.n)
    best, best_val = A[:], cur
    for t in range(steps):
        T = T0 * (T1 / T0) ** (t / steps)
        out_ids = np.flatnonzero(inA); in_ids = np.flatnonzero(~inA)
        i = out_ids[rng.integers(len(out_ids))]
        j = in_ids[rng.integers(len(in_ids))]
        d = q.v[j] - q.v[i] - (g[j] - q.R[j, i]) + g[i]
        if d >= 0 or rng.random() < math.exp(d / T):
            inA[i], inA[j] = False, True
            g += q.R[:, j] - q.R[:, i]
            cur += d
            if cur > best_val + 1e-12:
                best_val, best = cur, np.flatnonzero(inA).tolist()
    return sorted(best)


def tabu(q, K, rng, iters=None, tenure=None, A0=None):
    A = list(A0) if A0 is not None else random_set(q.n, K, rng)
    iters = iters or 20 * q.n
    tenure = tenure or max(3, min(q.n // 4, 10))
    inA = np.zeros(q.n, bool); inA[A] = True
    g = q.R[:, A].sum(1)
    cur = q.value(A)
    best, best_val = A[:], cur
    tabu_until = np.zeros(q.n, int)
    for it in range(iters):
        out_ids = np.flatnonzero(inA); in_ids = np.flatnonzero(~inA)
        d = (q.v[in_ids][None, :] - q.v[out_ids][:, None]
             - (g[in_ids][None, :] - q.R[np.ix_(out_ids, in_ids)]) + g[out_ids][:, None])
        allowed = (tabu_until[out_ids][:, None] <= it) & (tabu_until[in_ids][None, :] <= it)
        aspire = cur + d > best_val + 1e-12
        d = np.where(allowed | aspire, d, -np.inf)
        if not np.isfinite(d).any():
            continue
        a, b = np.unravel_index(np.argmax(d), d.shape)
        i, j = out_ids[a], in_ids[b]
        inA[i], inA[j] = False, True
        g += q.R[:, j] - q.R[:, i]
        cur += d[a, b]
        tabu_until[i] = tabu_until[j] = it + tenure
        if cur > best_val + 1e-12:
            best_val, best = cur, np.flatnonzero(inA).tolist()
    return sorted(best)


def combos_array(n, K):
    return np.array(list(itertools.combinations(range(n), K)), dtype=np.int32)


def qubo_all(q, C):
    """QUBO value of every subset in C (rows of site indices), vectorised."""
    vals = q.v[C].sum(1)
    K = C.shape[1]
    for a in range(K):
        for b in range(a + 1, K):
            vals -= q.R[C[:, a], C[:, b]]
    return vals


def brute_qubo(q, K, chunk=2_000_000):
    best, best_val = None, -np.inf
    it = itertools.combinations(range(q.n), K)
    while True:
        block = np.array(list(itertools.islice(it, chunk)), dtype=np.int32)
        if block.size == 0:
            break
        vals = qubo_all(q, block)
        k = int(np.argmax(vals))
        if vals[k] > best_val:
            best_val, best = float(vals[k]), sorted(block[k].tolist())
    return best, best_val


def mip_qubo(q, K, time_limit=600.0):
    """max v.x - sum R_ij y_ij,  y_ij >= x_i + x_j - 1,  y >= 0,  sum x = K."""
    n = q.n
    I, J = np.where(np.triu(q.R, 1) > 0)
    P = len(I)
    c = -np.concatenate([q.v, -q.R[I, J]])          # milp minimises
    rows = np.arange(P)
    A = coo_matrix((np.concatenate([np.ones(P), np.ones(P), -np.ones(P)]),
                    (np.concatenate([rows, rows, rows]),
                     np.concatenate([I, J, n + rows]))), shape=(P, n + P)).tocsr()
    card = csr_matrix((np.ones(n), (np.zeros(n, int), np.arange(n))), shape=(1, n + P))
    cons = [LinearConstraint(A, -np.inf, 1.0), LinearConstraint(card, K, K)]
    integ = np.concatenate([np.ones(n), np.zeros(P)])
    t = time.perf_counter()
    res = milp(c, constraints=cons, integrality=integ,
               bounds=Bounds(np.zeros(n + P), np.concatenate([np.ones(n), np.full(P, np.inf)])),
               options={"time_limit": time_limit, "disp": False})
    dt = time.perf_counter() - t
    x = np.round(res.x[:n]).astype(int) if res.x is not None else None
    A_sel = sorted(np.flatnonzero(x).tolist()) if x is not None else None
    return A_sel, (-res.fun if res.x is not None else None), dt, res.status == 0


# ----------------------------------------------------------- true objective
def pair_table(ev):
    """Cross-source scenario pairs, aggregated by the set of sites separating them."""
    p, qq = np.triu_indices(ev.m, 1)
    keep = ev.src[p] != ev.src[qq]
    p, qq = p[keep], qq[keep]
    w = ev.w[p] * ev.w[qq]
    S = ev.O[p] != ev.O[qq]                           # (pairs, n) separates?
    keys = np.packbits(S, axis=1)
    keys = np.ascontiguousarray(keys).view(np.dtype((np.void, keys.shape[1]))).ravel()
    u, inv = np.unique(keys, return_inverse=True)
    Wu = np.bincount(inv, weights=w)
    first = np.zeros(len(u), int); first[inv[::-1]] = np.arange(len(inv))[::-1]
    Su = S[first]
    nz = Su.any(1)                                    # pairs no site can separate drop out
    return Su[nz], Wu[nz]


def greedy_true(ev, K):
    """CELF lazy greedy on D(A)."""
    S, W = pair_table(ev)
    covered = np.zeros(len(W), bool)
    heap = [(-(W[S[:, j]]).sum(), j, 0) for j in range(ev.n)]
    heapq.heapify(heap)
    A = []
    while len(A) < K and heap:
        ng, j, stamp = heapq.heappop(heap)
        if stamp == len(A):
            A.append(j); covered |= S[:, j]
            continue
        gain = W[S[:, j] & ~covered].sum()
        heapq.heappush(heap, (-gain, j, len(A)))
    return sorted(A)


def mip_true(ev, K, time_limit=600.0):
    """max sum_g W_g z_g,  z_g <= sum_{i separates g} x_i,  z <= 1,  sum x = K."""
    S, W = pair_table(ev)
    G, n = S.shape
    c = -np.concatenate([np.zeros(n), W])
    gi, ii = np.nonzero(S)
    A = coo_matrix((np.concatenate([np.ones(G), -np.ones(len(gi))]),
                    (np.concatenate([np.arange(G), gi]), np.concatenate([n + np.arange(G), ii]))),
                   shape=(G, n + G)).tocsr()
    card = csr_matrix((np.ones(n), (np.zeros(n, int), np.arange(n))), shape=(1, n + G))
    t = time.perf_counter()
    res = milp(c, constraints=[LinearConstraint(A, -np.inf, 0.0), LinearConstraint(card, K, K)],
               integrality=np.concatenate([np.ones(n), np.zeros(G)]),
               bounds=Bounds(np.zeros(n + G), np.ones(n + G)),
               options={"time_limit": time_limit, "disp": False})
    dt = time.perf_counter() - t
    x = np.round(res.x[:n]).astype(int)
    return sorted(np.flatnonzero(x).tolist()), -res.fun, dt, res.status == 0


def chama_true(ev, K, solver="appsi_highs"):
    import pandas as pd
    from chama.optimize import CoverageFormulation
    S, W = pair_table(ev)
    cov = pd.DataFrame({"Sensor": [f"s{j}" for j in range(ev.n)],
                        "Coverage": [np.flatnonzero(S[:, j]).tolist() for j in range(ev.n)]})
    ent = pd.DataFrame({"Entity": list(range(len(W))), "Weight": W})
    t = time.perf_counter()
    out = CoverageFormulation().solve(cov, entity=ent, sensor_budget=K, use_sensor_cost=False,
                                      use_entity_weight=True, mip_solver_name=solver)
    dt = time.perf_counter() - t
    sel = sorted(int(s[1:]) for s in out["Sensors"])
    return sel, float(out["Objective"]), dt
