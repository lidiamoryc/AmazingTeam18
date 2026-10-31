"""Simulate contamination scenarios: every junction x several unknown start times."""
import sys
import time

import wntr

sys.path.insert(0, ".")
from qadv import hydraulics as hy

CONFIG = {
    "Net3": dict(inp="Net3.inp", starts=[0, 2, 4, 6, 8, 10]),
    "ky4": dict(inp="ky4.inp", starts=[0, 3, 6]),
}

if __name__ == "__main__":
    name = sys.argv[1]
    cfg = CONFIG[name]
    inp = hy.network_path(cfg["inp"])
    wn = wntr.network.WaterNetworkModel(inp)
    sources = list(wn.junction_name_list)
    meta = dict(network=name, starts=cfg["starts"], duration_h=24, inject_h=2,
                strength_mg_l=1000.0, threshold_mg_l=0.1, report_dt_s=900, quality_dt_s=300)
    print(f"{name}: {len(sources)} sources x {len(cfg['starts'])} start times", flush=True)
    t = time.time()
    E, nodes, scen, th = hy.simulate(inp, sources, cfg["starts"], duration_h=24, inject_h=2,
                                     workdir="/tmp")
    hy.save(f"data/{name}.npz", E, nodes, scen, th, meta)
    print(f"{name}: done in {time.time() - t:.0f}s, E={E.shape}", flush=True)
