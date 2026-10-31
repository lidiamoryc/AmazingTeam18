"""Deviation 3 (made after seeing the in-sample failure of averaged angles,
before any medoid test result): choose, for each p, the single training
solution that performs best ON AVERAGE ACROSS ALL TRAINING INSTANCES
(a "donor" angle set, cf. parameter-transfer studies). Still fixed angles: no
optimisation is done on test instances. Selection criterion: mean over
training instances of log(P_opt / P_uniform).
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from qadv import classical as cl, instances as ins, quantum as qm

FAMILIES = {"F1": dict(ns=[8, 12, 16], K=lambda n: n // 4),
            "F2": dict(ns=[10, 12, 14], K=lambda n: 5)}
PS = [1, 2, 4, 8]

if __name__ == "__main__":
    train, _ = ins.split_pool(ins.event_pool())
    rng = np.random.default_rng(11)
    out = json.load(open("results/fixed_angles.json"))
    for fam, cfg in FAMILIES.items():
        insts, sols = [], {p: [] for p in PS}
        for n in cfg["ns"]:
            K = cfg["K"](n)
            sub = qm.Subspace(n, K, "ring")
            for ev in ins.draw(train, n, 6, seed=7):
                v, r = ins.qubo_of(ev)
                Q = cl.qubo_all(cl.Qubo(v, r, 1.0), sub.C)
                E = qm.normalised_cost(Q)
                opt = Q >= Q.max() - 1e-12
                insts.append((sub, E, opt))
                ang = qm.optimise(sub, E, max(PS), rng, starts=6)
                for p in PS:
                    x = ang[p]
                    sols[p].append(-x if x[:p].sum() < 0 else x)
        out[fam]["medoid"] = {}
        for p in PS:
            score = []
            for x in sols[p]:
                s = [np.log(max(qm.success_prob(sub, E, x, opt), 1e-300) / opt.mean())
                     for sub, E, opt in insts]
                score.append(np.mean(s))
            b = int(np.argmax(score))
            out[fam]["medoid"][str(p)] = sols[p][b].tolist()
            wins = np.mean([qm.success_prob(sub, E, sols[p][b], opt) > opt.mean() for sub, E, opt in insts])
            print(f"{fam} p={p}: best donor beats uniform on {100 * wins:.0f}% of training instances, "
                  f"mean log-gain {score[b]:+.2f}  (mean-angle approach failed in-sample)")
    json.dump(out, open("results/fixed_angles.json", "w"), indent=1)
