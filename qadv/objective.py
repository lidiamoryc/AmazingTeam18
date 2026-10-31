"""Objectives for choosing K sampling sites.

Pair-discrimination view. Two candidate scenarios p, q with DIFFERENT source
nodes are "told apart" by a sampling plan A if at least one site in A gives a
different result under p than under q. Pairs with the same source node but
different start times never need separating: we want the source, not the time.

  true objective  D(A) = total weight of cross-source pairs told apart by A
                  Weighted coverage, hence monotone submodular.

  v_i   = weight of pairs site i alone tells apart
  r_ij  = weight of pairs BOTH i and j tell apart (the redundancy)

  QUBO  Q(A) = sum_{i in A} v_i  -  lam * sum_{i<j in A} r_ij

With lam = 1, Q(A) is exactly the second Bonferroni lower bound on D(A):
  sum v_i - sum r_ij  <=  D(A)  <=  sum v_i.
So lam = 1 is not a tuning choice; it is the principled second-order
truncation of inclusion-exclusion. It is still only an approximation: when
three or more selected sites tell apart the same pair, Q under-counts.
"""
import numpy as np


def _per_source(ev):
    S = np.zeros((ev.m, ev.n_sources))
    S[np.arange(ev.m), ev.src] = ev.w
    return S                                    # weight of scenario p, in its source column


def site_values(ev):
    O = ev.O.astype(float)
    S = _per_source(ev)                         # (m, ns)
    W = ev.w.sum()
    Ws = S.sum(0)                               # (ns,)
    W1 = ev.w @ O                               # (n,)
    W1s = S.T @ O                               # (ns, n)
    return W1 * (W - W1) - (W1s * (Ws[:, None] - W1s)).sum(0)


def redundancy(ev):
    """r_ij = cross-source pair weight separated by BOTH sites i and j."""
    O = ev.O.astype(float)
    n = ev.n
    w = ev.w

    def joint(wv, Ov):
        A1 = Ov.T @ (wv[:, None] * Ov)          # W11
        a = wv @ Ov
        A10 = a[:, None] - A1                   # o_i=1, o_j=0
        A01 = a[None, :] - A1
        A00 = wv.sum() - A1 - A10 - A01
        return A00, A01, A10, A1

    A00, A01, A10, A11 = joint(w, O)
    r = A00 * A11 + A01 * A10
    for s in range(ev.n_sources):
        idx = np.where(ev.src == s)[0]
        if len(idx) < 2:
            continue
        B00, B01, B10, B11 = joint(w[idx], O[idx])
        r -= B00 * B11 + B01 * B10
    np.fill_diagonal(r, 0.0)
    return r


def true_value(ev, A):
    """D(A) exactly, via the signature classes the plan induces."""
    A = list(A)
    W = ev.w.sum()
    Ws = np.bincount(ev.src, weights=ev.w, minlength=ev.n_sources)
    total = (W ** 2 - (Ws ** 2).sum()) / 2
    if not A:
        return 0.0
    sig = ev.O[:, A]
    keys = np.packbits(sig, axis=1)
    keys = np.ascontiguousarray(keys).view(np.dtype((np.void, keys.shape[1]))).ravel()
    _, g = np.unique(keys, return_inverse=True)
    Wg = np.bincount(g, weights=ev.w)
    Wgs = np.zeros((len(Wg), ev.n_sources))
    np.add.at(Wgs, (g, ev.src), ev.w)
    undist = ((Wg ** 2 - (Wgs ** 2).sum(1)) / 2).sum()
    return float(total - undist)


def total_pairs(ev):
    W = ev.w.sum()
    Ws = np.bincount(ev.src, weights=ev.w, minlength=ev.n_sources)
    return float((W ** 2 - (Ws ** 2).sum()) / 2)


def qubo_value(A, v, r, lam=1.0):
    A = list(A)
    if not A:
        return 0.0
    val = float(v[A].sum())
    if len(A) > 1:
        val -= lam * float(np.triu(r[np.ix_(A, A)], 1).sum())
    return val


def identification(ev, A):
    """Operational KPIs after sampling plan A.

    p_identified    probability the sample results leave exactly one candidate source
    exp_remaining   expected number of candidate source nodes still consistent
                    (= field inspections still needed to find the source)
    posterior(sig)  can be read off the same classes: P(source | results)
    """
    A = list(A)
    if not A:
        return 0.0, float(ev.n_sources)
    sig = ev.O[:, A]
    keys = np.packbits(sig, axis=1)
    keys = np.ascontiguousarray(keys).view(np.dtype((np.void, keys.shape[1]))).ravel()
    _, g = np.unique(keys, return_inverse=True)
    pid, rem = 0.0, 0.0
    for c in range(g.max() + 1):
        idx = g == c
        Wc = ev.w[idx].sum()
        k = len(set(ev.src[idx].tolist()))
        rem += Wc * k
        pid += Wc * (k == 1)
    return float(pid / ev.w.sum()), float(rem / ev.w.sum())
