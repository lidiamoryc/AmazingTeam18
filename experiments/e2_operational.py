"""Full-size operational benchmark (Judges 1, 3, 5, 6, 7).

Every informative site of real Net3 alarm events, K in {3, 5, 10} crews.
  * exact optimum of the TRUE objective (HiGHS MIP; Chama cross-check)
  * how good is the QUBO as a stand-in?  D(QUBO optimum) / D(true optimum)
  * greedy, greedy + local search, simulated annealing, tabu: quality + time
  * KPIs: P(source identified), expected candidate sources left, versus
    no sampling, random sites, and naive top-K ranking
"""
import csv
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from qadv import classical as cl, instances as ins, objective as ob

KS = [3, 5, 10]
LAMS = [0.25, 0.5, 1.0]
N_EVENTS = 15

if __name__ == "__main__":
    _, test = ins.split_pool(ins.event_pool())
    rng = np.random.default_rng(42)
    events = [test[i] for i in rng.choice(len(test), N_EVENTS, replace=False)]
    rows = []
    t0 = time.time()
    for e_id, ev in enumerate(events):
        v, r = ob.site_values(ev), ob.redundancy(ev)
        for K in KS:
            if K >= ev.n:
                continue
            A_opt, D_opt, t_mip, ok = cl.mip_true(ev, K, time_limit=300)
            row = dict(event=e_id, alarm=ev.alarm_node, t_alarm=ev.t_alarm_h, m=ev.m,
                       sources=ev.n_sources, n=ev.n, K=K, D_opt=D_opt, t_mip_true=t_mip,
                       mip_true_ok=int(ok))
            rel = lambda A: ob.true_value(ev, A) / D_opt
            t = time.perf_counter(); A = cl.greedy_true(ev, K); row["t_greedy_true"] = time.perf_counter() - t
            row["greedy_true"] = rel(A)
            for lam in LAMS:
                q = cl.Qubo(v, r, lam)
                Aq, vq, tq, okq = cl.mip_qubo(q, K, time_limit=20)
                # if the MIP is not proven within 20 s, use the best of its
                # incumbent and 10 tabu restarts as the QUBO optimum (flagged)
                if not okq:
                    for _ in range(10):
                        At = cl.tabu(q, K, rng, iters=8 * ev.n)
                        if Aq is None or q.value(At) > q.value(Aq) + 1e-12:
                            Aq = At
                row[f"qubo{lam}_opt_true"] = rel(Aq)
                row[f"t_mip_qubo{lam}"] = tq
                row[f"mip_qubo{lam}_proven"] = int(okq)
                t = time.perf_counter(); Ag = cl.local_search(q, cl.greedy_qubo(q, K))
                row[f"gls{lam}_t"] = time.perf_counter() - t
                row[f"gls{lam}_true"] = rel(Ag)
                row[f"gls{lam}_hits_qubo_opt"] = int(abs(q.value(Ag) - q.value(Aq)) < 1e-9)
            q = cl.Qubo(v, r, 0.5)
            for name, f in (("sa", lambda: cl.sim_anneal(q, K, rng, sweeps=30)),
                            ("tabu", lambda: cl.tabu(q, K, rng, iters=8 * ev.n))):
                t = time.perf_counter(); A = f(); row[f"{name}_t"] = time.perf_counter() - t
                row[f"{name}_true"] = rel(A)
            pid, rem = ob.identification(ev, A_opt)
            row.update(opt_pid=pid, opt_rem=rem)
            pid, rem = ob.identification(ev, cl.greedy_true(ev, K))
            row.update(greedy_pid=pid, greedy_rem=rem)
            topk = list(np.argsort(-v)[:K])
            pid, rem = ob.identification(ev, topk)
            row.update(topk_pid=pid, topk_rem=rem, topk_true=rel(topk))
            rs = [ob.identification(ev, cl.random_set(ev.n, K, rng)) for _ in range(30)]
            row.update(rand_pid=float(np.mean([a for a, _ in rs])), rand_rem=float(np.mean([b for _, b in rs])))
            row.update(none_rem=float(ev.n_sources))
            rows.append(row)
        print(f"event {e_id + 1}/{N_EVENTS}: {ev.n_sources} sources, {ev.n} sites  ({time.time() - t0:.0f}s)", flush=True)
    with open("results/operational_Net3.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    import pandas as pd
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    cols = ["greedy_true", "gls0.5_true", "sa_true", "tabu_true", "topk_true",
            "qubo0.25_opt_true", "qubo0.5_opt_true", "qubo1.0_opt_true"]
    print("\nfraction of the TRUE optimum reached (mean over events):")
    print(df.groupby("K")[cols].mean().round(4).to_string())
    print("\nseconds (median):")
    print(df.groupby("K")[["t_mip_true", "t_greedy_true", "gls0.5_t", "sa_t", "tabu_t"]].median().round(4).to_string())
    print("\nKPIs (mean): P(identified) and expected candidate sources left")
    print(df.groupby("K")[["opt_pid", "greedy_pid", "topk_pid", "rand_pid", "opt_rem", "greedy_rem",
                            "topk_rem", "rand_rem", "none_rem"]].mean().round(3).to_string())
