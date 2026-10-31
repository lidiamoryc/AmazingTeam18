"""Seeded instance draws, shared by every experiment so they see the same problems."""
import numpy as np

from . import event as evm
from . import hydraulics as hy
from . import objective as ob

_CACHE = {}


def load(net="Net3"):
    if net not in _CACHE:
        E, nodes, scen, th = hy.load(f"data/{net}.npz")
        cands = evm.candidate_events(E, nodes, scen, th, min_sources=20)
        _CACHE[net] = (E, nodes, scen, th, cands)
    return _CACHE[net]


def event_pool(net="Net3"):
    E, nodes, scen, th, cands = load(net)
    pool = []
    for nd, t, m, ns in cands:
        ev = evm.informative(evm.build_event(E, nodes, scen, th, nd, t))
        if ev.n_sources >= 20 and ev.n >= 10:
            pool.append(ev)
    return pool


def split_pool(pool, seed=7, train_frac=0.3):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pool))
    k = int(len(pool) * train_frac)
    return [pool[i] for i in idx[:k]], [pool[i] for i in idx[k:]]


def draw(pool, n, count, seed):
    rng = np.random.default_rng(seed + 1000 * n)
    ok = [ev for ev in pool if ev.n >= n]
    out = []
    for c in range(count):
        ev = ok[rng.integers(len(ok))]
        sub = evm.subsample_sites(ev, n, rng)
        out.append(sub)
    return out


def qubo_of(ev, lam=1.0):
    return ob.site_values(ev), ob.redundancy(ev)
