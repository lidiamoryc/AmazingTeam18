"""Classical time-to-solution on the SAME held-out instances as e3b.

Randomised heuristics get 40 seeds per instance at a fixed per-run budget:
TTS = mean run time / success rate (expected time to first hit of an exact
optimum with restarts). Operation counts are recorded too, because Python
constant overheads flatten wall-clock curves at small n and would make the
classical side look like it scales better than it does.
"""
import csv
import math
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from qadv import classical as cl, instances as ins

FAM = {"F1": dict(ns=[12, 16, 20, 24, 28], K=lambda n: n // 4),
       "F2": dict(ns=[12, 16, 20, 24, 28, 32, 36, 40], K=lambda n: 5)}
N_TEST, SEEDS = 12, 40
SA_SWEEPS, TABU_ITERS = 30, 8


def timed(f):
    t = time.perf_counter(); out = f(); return out, time.perf_counter() - t


if __name__ == "__main__":
    fam_sel = sys.argv[1:] or list(FAM)
    _, test = ins.split_pool(ins.event_pool())
    rows = []
    for fam in fam_sel:
        cfg = FAM[fam]
        for n in cfg["ns"]:
            K = cfg["K"](n)
            C = cl.combos_array(n, K)
            t_n = time.time()
            for k, ev in enumerate(ins.draw(test, n, N_TEST, seed=2026)):
                v, r = ins.qubo_of(ev)
                q = cl.Qubo(v, r, 1.0)
                Q, t_bf = timed(lambda: cl.qubo_all(q, C))
                best = float(Q.max())
                hit = lambda A: q.value(A) >= best - 1e-12 * max(1.0, abs(best))
                row = dict(family=fam, n=n, K=K, inst=k, t_brute=t_bf,
                           ops_brute=len(C) * K * (K - 1) // 2)
                A, t = timed(lambda: cl.local_search(q, cl.greedy_qubo(q, K)))
                row.update(gls_ok=int(hit(A)), t_gls=t)
                (_, _, t_mip, ok), _ = timed(lambda: cl.mip_qubo(q, K, time_limit=120))
                row.update(t_mip=t_mip, mip_ok=int(ok))
                rng = np.random.default_rng(100 + k)
                for name, run, ops in (
                    ("sa", lambda: cl.sim_anneal(q, K, rng, sweeps=SA_SWEEPS), SA_SWEEPS * n),
                    ("tabu", lambda: cl.tabu(q, K, rng, iters=TABU_ITERS * n), TABU_ITERS * n * K * (n - K)),
                    ("rrls", lambda: cl.local_search(q, cl.random_set(n, K, rng)), None),
                ):
                    succ, ts = 0, []
                    for _ in range(SEEDS):
                        A, t = timed(run); ts.append(t); succ += hit(A)
                    pr = succ / SEEDS
                    row[f"{name}_p"] = pr
                    row[f"{name}_t"] = float(np.mean(ts))
                    row[f"tts_{name}"] = (np.mean(ts) / pr) if pr > 0 else math.inf
                    if ops is not None:
                        row[f"ops_{name}"] = (ops / pr) if pr > 0 else math.inf
                rows.append(row)
            sel = [r_ for r_ in rows if r_["family"] == fam and r_["n"] == n]
            med = lambda key: float(np.median([r_[key] for r_ in sel]))
            print(f"{fam} n={n:2d} K={K}  greedy+LS exact on {sum(r_['gls_ok'] for r_ in sel)}/{len(sel)}  "
                  f"SA hit {med('sa_p'):.2f}  tabu hit {med('tabu_p'):.2f}  RR-LS hit {med('rrls_p'):.2f}  "
                  f"TTS(s): SA {med('tts_sa'):.3g} tabu {med('tts_tabu'):.3g} RR-LS {med('tts_rrls'):.3g} "
                  f"MIP {med('t_mip'):.3g} brute {med('t_brute'):.3g}  ({time.time() - t_n:.0f}s)", flush=True)
    tag = "_".join(fam_sel)
    with open(f"results/classical_{tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
