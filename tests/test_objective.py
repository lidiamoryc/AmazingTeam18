"""v_i, r_ij and D(A) checked against literal pair-by-pair counting."""
import itertools
import sys

import numpy as np

sys.path.insert(0, ".")
from qadv import objective as ob
from qadv.event import Event


def random_event(rng, m=14, n=7, ns=5):
    O = rng.integers(0, 2, size=(m, n)).astype(np.uint8)
    src = rng.integers(0, ns, size=m)
    src[:ns] = np.arange(ns)
    w = rng.random(m); w /= w.sum()
    return Event("A", 0, 1, O, w, src, [f"s{i}" for i in range(ns)], [f"j{i}" for i in range(n)])


def brute(ev):
    pairs = [(p, q) for p, q in itertools.combinations(range(ev.m), 2) if ev.src[p] != ev.src[q]]
    pw = {pq: ev.w[pq[0]] * ev.w[pq[1]] for pq in pairs}
    sep = lambda i, p, q: ev.O[p, i] != ev.O[q, i]
    v = np.array([sum(pw[pq] for pq in pairs if sep(i, *pq)) for i in range(ev.n)])
    r = np.zeros((ev.n, ev.n))
    for i in range(ev.n):
        for j in range(ev.n):
            if i != j:
                r[i, j] = sum(pw[pq] for pq in pairs if sep(i, *pq) and sep(j, *pq))
    D = lambda A: sum(pw[pq] for pq in pairs if any(sep(i, *pq) for i in A))
    return v, r, D


def test_all(trials=40):
    rng = np.random.default_rng(0)
    for _ in range(trials):
        ev = random_event(rng)
        v0, r0, D0 = brute(ev)
        v, r = ob.site_values(ev), ob.redundancy(ev)
        assert np.allclose(v, v0), "v_i mismatch"
        assert np.allclose(r, r0), "r_ij mismatch"
        for k in range(1, ev.n + 1):
            for A in itertools.combinations(range(ev.n), k):
                d = ob.true_value(ev, A)
                assert abs(d - D0(A)) < 1e-12, "D(A) mismatch"
                q = ob.qubo_value(A, v, r, 1.0)
                assert q <= d + 1e-12, "Bonferroni lower bound violated"
                assert d <= v[list(A)].sum() + 1e-12, "union bound violated"
    print(f"objective: {trials} random events, every subset checked -- OK")


if __name__ == "__main__":
    test_all()
