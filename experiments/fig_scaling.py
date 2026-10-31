"""Main figure: relative TTS growth, every method normalised to its own value
at the smallest n. Units differ (shots vs seconds), so only slopes compare."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
for ax, fam, title in ((axes[0], "F1", "Crews grow with network (K = n/4)"),
                       (axes[1], "F2", "Fixed crews (K = 5), the operational case")):
    q = pd.read_csv(f"results/quantum_{fam}.csv")
    c = pd.read_csv(f"results/classical_{fam}.csv")
    m = pd.read_csv("results/mip_time_to_find.csv"); m = m[m.family == fam]
    ns = sorted(q.n.unique())
    med = lambda d, col, f=lambda x: x: np.array([np.median(f(d.loc[d.n == n, col].values)) for n in ns])
    series = [
        ("random guessing", med(q, "p_uniform", lambda x: 1 / x), dict(color="0.6", ls=":", marker="")),
        ("XY-QAOA, donor angles, p=8", med(q, "donor_p8", lambda x: 1 / x), dict(color="#7F77DD", marker="o")),
        ("XY-QAOA p=8 + quantum min. finding*", med(q, "donor_p8", lambda x: 1 / np.sqrt(x)), dict(color="#534AB7", marker="s")),
        ("exact MIP (HiGHS), time to find", med(m, "t_find"), dict(color="#D85A30", marker="^")),
        ("simulated annealing", med(c, "tts_sa"), dict(color="#1D9E75", marker="v")),
        ("tabu search", med(c, "tts_tabu"), dict(color="#0F6E56", marker="D")),
    ]
    for name, y, st in series:
        ax.plot(ns, y / y[0], label=name, lw=1.6, ms=5, **st)
    ax.set_yscale("log"); ax.set_xlabel("candidate sampling sites n"); ax.grid(alpha=0.3, which="both")
    ax.set_title(title, fontsize=10)
axes[0].set_ylabel("time-to-solution, relative to smallest n")
axes[1].legend(fontsize=7.5, loc="upper left")
fig.suptitle("Source-identification sampling on EPANET Net3: growth of time-to-solution "
             "(median of 12 held-out instances per size)", fontsize=10)
fig.text(0.5, 0.005, "*requires fault-tolerant hardware. Noiseless simulation. Lower slope = better scaling.",
         ha="center", fontsize=8, color="0.35")
fig.tight_layout(rect=(0, 0.03, 1, 0.95))
fig.savefig("results/scaling.png", dpi=160)
print("results/scaling.png")
