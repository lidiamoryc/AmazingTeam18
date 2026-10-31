"""Fit TTS scaling and apply the pre-registered decision rule mechanically.

log2(median TTS) = a + b*n per method; base = 2^b; 95% CI by bootstrap
(resample the 12 instances at each n, 2000 times). Quantum TTS is in shots,
classical in seconds and in operation counts: units differ, so only the BASES
are compared, as in Shaydulin et al. (2024).
"""
import json
import math
import sys

import numpy as np
import pandas as pd

B = 2000
rng = np.random.default_rng(0)


def fit_base(df, col, ns, invert=False, sqrt=False):
    def tts(x):
        x = np.asarray(x, float)
        if invert:
            x = 1.0 / np.maximum(x, 1e-300)
        if sqrt:
            x = np.sqrt(x)
        return x

    groups = [tts(df.loc[df.n == n, col].values) for n in ns]
    if any((~np.isfinite(g)).mean() > 0.5 for g in groups):
        return None
    def slope(gs):
        y = np.log2([np.median(g) for g in gs])
        return np.polyfit(ns, y, 1)[0]
    b = slope(groups)
    boots = [slope([g[rng.integers(len(g), size=len(g))] for g in groups]) for _ in range(B)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return 2 ** b, 2 ** lo, 2 ** hi


def main():
    res = {}
    for fam in ("F1", "F2"):
        q = pd.read_csv(f"results/quantum_{fam}.csv")
        c = pd.read_csv(f"results/classical_{fam}.csv")
        ns = sorted(q.n.unique())
        rows = []
        add = lambda name, kind, r: rows.append((name, kind, r)) if r else None
        add("random guessing (uniform over feasible)", "Q-ref", fit_base(q, "p_uniform", ns, invert=True))
        for p in (1, 2, 4, 8):
            add(f"XY-QAOA donor angles p={p}", "Q", fit_base(q, f"donor_p{p}", ns, invert=True))
        add("XY-QAOA donor p=8 + quantum min. finding", "Q", fit_base(q, "donor_p8", ns, invert=True, sqrt=True))
        add("XY-QAOA mean angles p=8", "Q", fit_base(q, "fixed_p8", ns, invert=True))
        small = [n for n in ns if n <= 20]
        if "opt_p8" in q and len(small) >= 3:
            add("XY-QAOA per-instance optimised p=8 (n<=20, no tuning cost counted)", "Q",
                fit_base(q[q.n <= 20], "opt_p8", small, invert=True))
        for col, name in (("tts_sa", "simulated annealing"), ("tts_tabu", "tabu search"),
                          ("tts_rrls", "random-restart local search"), ("t_mip", "exact MIP (HiGHS)"),
                          ("t_brute", "brute force (vectorised)")):
            add(f"{name} [seconds]", "C", fit_base(c, col, ns))
        for col, name in (("ops_sa", "simulated annealing"), ("ops_tabu", "tabu search"),
                          ("ops_brute", "brute force")):
            add(f"{name} [operations]", "C", fit_base(c, col, ns))
        res[fam] = rows
        gls = c.groupby("n").gls_ok.mean()
        print(f"\n=== {fam}  (n = {ns[0]}..{ns[-1]}, K = {'n/4' if fam == 'F1' else '5'}) ===")
        print(f"greedy + local search found the exact optimum: "
              + ", ".join(f"n={n}: {100 * gls[n]:.0f}%" for n in ns))
        print(f"{'method':72s} {'base':>7} {'95% CI':>17}")
        for name, kind, (b, lo, hi) in rows:
            print(f"{('  ' if kind != 'C' else '') + name:72s} {b:7.3f} [{lo:6.3f}, {hi:6.3f}]")
        qs = [(n, r) for n, k, r in rows if k == "Q"]
        cs = [(n, r) for n, k, r in rows if k == "C" and "operations" not in n]
        best_c = min(cs, key=lambda x: x[1][0])
        best_q = min(qs, key=lambda x: x[1][0])
        verdict = best_q[1][2] < best_c[1][1]
        print(f"best quantum: {best_q[0]}  base {best_q[1][0]:.3f}, CI upper {best_q[1][2]:.3f}")
        print(f"best classical: {best_c[0]}  base {best_c[1][0]:.3f}, CI lower {best_c[1][1]:.3f}")
        print("DECISION (pre-registered rule):",
              "EVIDENCE OF SCALING ADVANTAGE" if verdict else "NO EVIDENCE OF ADVANTAGE")
        res[fam + "_decision"] = bool(verdict)
    json.dump({k: v for k, v in res.items()}, open("results/fits.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
