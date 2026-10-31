"""Is 'QAOA + QMF scales better than exact MIP' robust? Refit with the
smallest n dropped (Shaydulin et al. fit only N >= 28 for stability) and with
MIP measured as time-to-FIND rather than time-to-prove."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "experiments")
from e3d_fit import fit_base

for fam in ("F1", "F2"):
    q = pd.read_csv(f"results/quantum_{fam}.csv")
    c = pd.read_csv(f"results/classical_{fam}.csv")
    m = pd.read_csv("results/mip_time_to_find.csv"); m = m[m.family == fam]
    all_ns = sorted(q.n.unique())
    print(f"\n=== {fam} ===")
    print(f"{'fit range':>11} | {'QAOA p8':>22} | {'QAOA p8 + QMF':>22} | {'MIP find':>22} | {'MIP prove':>22} | {'tabu (s)':>22}")
    for nmin in (12, 16, 20):
        ns = [n for n in all_ns if n >= nmin]
        if len(ns) < 3:
            continue
        sub = lambda d: d[d.n >= nmin]
        cells = [fit_base(sub(q), "donor_p8", ns, invert=True),
                 fit_base(sub(q), "donor_p8", ns, invert=True, sqrt=True),
                 fit_base(sub(m), "t_find", ns), fit_base(sub(m), "t_prove", ns),
                 fit_base(sub(c), "tts_tabu", ns)]
        fmt = lambda r: f"{r[0]:.3f} [{r[1]:.3f},{r[2]:.3f}]"
        print(f"n={ns[0]:>2}..{ns[-1]:<2}    | " + " | ".join(f"{fmt(x):>22}" for x in cells))
        qmf, mf = cells[1], cells[2]
        print(f"{'':11s}   QMF beats MIP time-to-find by the pre-registered rule? "
              f"{'YES' if qmf[2] < mf[1] else 'no'}  (QMF upper {qmf[2]:.3f} vs MIP lower {mf[1]:.3f})")
