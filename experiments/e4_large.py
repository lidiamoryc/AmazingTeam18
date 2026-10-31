"""Dr. Mousa's scale: hundreds of candidate sources, a handful of crews.

Real ky4 network (959 junctions, 2877 scenarios). For the largest alarm
events: solve classically (lazy greedy with the Krause et al. online bound as
an optimality certificate, plus exact MIP with a time limit), then compute
what XY-QAOA would need on hardware for the same instance.
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from qadv import classical as cl, event as evm, hydraulics as hy, objective as ob, quantum as qm

# Error rates, sourced in the README:
#   Quantinuum Helios: 99.921% average 2-qubit fidelity, 98 all-to-all qubits
#   IBM Heron: ~3e-3 median 2-qubit error, 156 qubits
HW = {"Quantinuum Helios (98 q)": (98, 7.9e-4), "IBM Heron (156 q)": (156, 3e-3)}
ZEPHYR_Z15_CLIQUE = 232   # Boothby et al., Zephyr topology report: K_{16m-8}, m = 15


def online_bound(ev, A, K):
    S, W = cl.pair_table(ev)
    cov = S[:, A].any(1)
    base = W[cov].sum()
    gains = np.sort(np.array([W[S[:, j] & ~cov].sum() for j in range(ev.n) if j not in A]))[::-1]
    return base, base + gains[:K].sum()


if __name__ == "__main__":
    E, nodes, scen, th = hy.load("data/ky4.npz")
    cands = evm.candidate_events(E, nodes, scen, th, min_sources=100)
    cands.sort(key=lambda c: -c[3])
    print(f"ky4 alarm events with >= 100 candidate sources: {len(cands)}")
    seen, events = set(), []
    for nd, t, m, ns in cands:
        if nd in seen:
            continue
        ev = evm.informative(evm.build_event(E, nodes, scen, th, nd, t))
        if ev.n >= 50:
            events.append(ev); seen.add(nd)
        if len(events) == 3:
            break
    out = []
    for ev in events:
        v, r = ob.site_values(ev), ob.redundancy(ev)
        nnz = int((np.triu(r, 1) > 0).sum())
        rec = dict(alarm=ev.alarm_node, t_alarm=ev.t_alarm_h, scenarios=ev.m,
                   sources=ev.n_sources, sites=ev.n, nonzero_pairs=nnz)
        for K in (5, 10):
            t = time.perf_counter(); A = cl.greedy_true(ev, K); tg = time.perf_counter() - t
            D, UB = online_bound(ev, A, K)
            Am, Dm, tm, okm = cl.mip_true(ev, K, time_limit=120)
            pid, rem = ob.identification(ev, Am if okm else A)
            rec[f"K{K}"] = dict(greedy_s=round(tg, 3), greedy_certified_pct=round(100 * D / UB, 2),
                                greedy_vs_mip_pct=round(100 * D / Dm, 3), mip_s=round(tm, 2),
                                mip_proven=bool(okm), p_identified=round(pid, 3),
                                sources_left=round(rem, 2))
            hw = {}
            for p in (1, 4, 8):
                N2 = qm.two_qubit_gates(ev.n, K, p, nnz)
                hw[f"p{p}"] = {"two_qubit_gates": N2}
                for name, (nq, eps) in HW.items():
                    F = (1 - eps) ** N2
                    hw[f"p{p}"][name] = {"fits": ev.n <= nq, "circuit_fidelity": float(f"{F:.3g}")}
            rec[f"K{K}"]["quantum"] = hw
        rec["fits_annealer_clique"] = ev.n <= ZEPHYR_Z15_CLIQUE
        out.append(rec)
        print(json.dumps(rec, indent=1), flush=True)
    json.dump(out, open("results/large_ky4.json", "w"), indent=1)
