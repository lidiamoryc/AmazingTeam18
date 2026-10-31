"""XY-QAOA on held-out TEST instances: fixed angles (all n) and per-instance
optimised angles (n <= 20, an optimistic upper bound for QAOA)."""
import csv
import json
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from qadv import classical as cl, instances as ins, objective as ob, quantum as qm

FAM = {"F1": dict(ns=[12, 16, 20, 24, 28], K=lambda n: n // 4),
       "F2": dict(ns=[12, 16, 20, 24, 28, 32, 36, 40], K=lambda n: 5)}
PS = [1, 2, 4, 8]
N_TEST = 12

if __name__ == "__main__":
    fam_sel = sys.argv[1:] or list(FAM)
    angles = json.load(open("results/fixed_angles.json"))
    _, test = ins.split_pool(ins.event_pool())
    rng = np.random.default_rng(3)
    rows = []
    for fam in fam_sel:
        cfg = FAM[fam]
        for n in cfg["ns"]:
            K = cfg["K"](n)
            t = time.time()
            sub = qm.Subspace(n, K, "ring")
            for k, ev in enumerate(ins.draw(test, n, N_TEST, seed=2026)):
                v, r = ins.qubo_of(ev)
                q = cl.Qubo(v, r, 1.0)
                Q = cl.qubo_all(q, sub.C)
                E = qm.normalised_cost(Q)
                best = Q.max()
                opt = Q >= best - 1e-12 * max(1.0, abs(best))
                n_opt = int(opt.sum())
                tv = None
                row = dict(family=fam, n=n, K=K, inst=k, dim=sub.dim, n_opt=n_opt,
                           p_uniform=n_opt / sub.dim,
                           nnz_pairs=int((np.triu(r, 1) > 0).sum()))
                for p in PS:
                    x = np.array(angles[fam]["fixed"][str(p)])
                    row[f"fixed_p{p}"] = qm.success_prob(sub, E, x, opt)
                    x = np.array(angles[fam]["medoid"][str(p)])
                    row[f"donor_p{p}"] = qm.success_prob(sub, E, x, opt)
                if n <= 20:
                    opt_ang = qm.optimise(sub, E, max(PS), rng, starts=4)
                    for p in PS:
                        row[f"opt_p{p}"] = qm.success_prob(sub, E, opt_ang[p], opt)
                rows.append(row)
            last = rows[-1]
            med = lambda key: np.median([r_[key] for r_ in rows if r_["family"] == fam and r_["n"] == n])
            print(f"{fam} n={n:2d} K={K} dim={sub.dim:8,d}  P(opt) uniform={med('p_uniform'):.2e}  "
                  f"mean-angle p8={med('fixed_p8'):.2e}  donor p1={med('donor_p1'):.2e} p4={med('donor_p4'):.2e} p8={med('donor_p8'):.2e}"
                  + (f"  optimised p8={med('opt_p8'):.2e}" if n <= 20 else "")
                  + f"  ({time.time() - t:.0f}s)", flush=True)
            del sub
    keys = sorted({k for r_ in rows for k in r_}, key=lambda s: (s not in ("family", "n", "K", "inst"), s))
    tag = "_".join(fam_sel)
    with open(f"results/quantum_{tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
