"""Train fixed QAOA angles on TRAINING events only (never used for testing).

Deviations from pre-registration (both fixed before any test result existed):
  1. F1 (K = n/4) needs n divisible by 4, so it trains on n in {8,12,16};
     F2 (K = 5) trains on n in {10,12,14} as registered.
  2. Sign canonicalisation before averaging. The cost and mixer are real
     matrices, so angles (gamma, beta) and (-gamma, -beta) give identical
     output distributions (complex conjugation). Training runs land on either
     mirror copy at random, and averaging mixed copies cancels towards zero.
     Each solution is flipped so that sum(gamma) >= 0 before averaging.
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from qadv import classical as cl, instances as ins, quantum as qm

P_MAX = 8
FAMILIES = {"F1": dict(ns=[8, 12, 16], K=lambda n: n // 4),
            "F2": dict(ns=[10, 12, 14], K=lambda n: 5)}

if __name__ == "__main__":
    pool = ins.event_pool()
    train, _ = ins.split_pool(pool)
    rng = np.random.default_rng(11)
    out = {}
    for fam, cfg in FAMILIES.items():
        per_p = {p: [] for p in range(1, P_MAX + 1)}
        t = time.time()
        for n in cfg["ns"]:
            K = cfg["K"](n)
            sub = qm.Subspace(n, K, "ring")
            for ev in ins.draw(train, n, 6, seed=7):
                v, r = ins.qubo_of(ev)
                q = cl.Qubo(v, r, 1.0)
                E = qm.normalised_cost(cl.qubo_all(q, sub.C))
                angles = qm.optimise(sub, E, P_MAX, rng, starts=6)
                for p, x in angles.items():
                    if x[:p].sum() < 0:
                        x = -x
                    per_p[p].append(x.tolist())
        fixed = {p: np.mean(np.array(xs), axis=0).tolist() for p, xs in per_p.items()}
        spread = {p: np.std(np.array(xs), axis=0).tolist() for p, xs in per_p.items()}
        out[fam] = {"fixed": fixed, "std": spread, "n_train": len(per_p[1])}
        print(f"{fam}: trained on {len(per_p[1])} instances in {time.time() - t:.0f}s")
        for p in (1, 2, 4, 8):
            g = np.round(fixed[p][:p], 3); b = np.round(fixed[p][p:], 3)
            print(f"   p={p}: gamma={g.tolist()}  beta={b.tolist()}")
    json.dump(out, open("results/fixed_angles.json", "w"), indent=1)
