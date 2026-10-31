"""Contamination scenario simulation on EPANET networks via WNTR.

A scenario is (source node, injection start time). For each one we store a
boolean exceedance tensor E[t, node]: is the concentration at `node` above the
lab detection threshold at report time t? Everything downstream -- which
scenarios are consistent with an alarm, what a crew's grab sample would show
-- is read off E. Concentrations themselves are not needed.
"""
import os

import numpy as np
import wntr

LIB = os.path.join(os.path.dirname(wntr.__file__), "library", "networks")


def network_path(name):
    return os.path.join(LIB, name)


def simulate(
    inp,
    sources,
    start_times_h,
    duration_h=24,
    inject_h=2,
    strength_mg_l=1000.0,
    threshold_mg_l=0.1,
    quality_dt_s=300,
    report_dt_s=900,
    workdir="/tmp",
    verbose=True,
):
    """Return E (n_scen, n_t, n_nodes) bool, node list, scenario list, times_h."""
    base = wntr.network.WaterNetworkModel(inp)
    nodes = list(base.junction_name_list)
    n_pat = int(np.ceil(duration_h))
    scen, blocks = [], []
    prefix = os.path.join(workdir, f"_qadv_{os.getpid()}")
    times_h = None
    total = len(sources) * len(start_times_h)
    done = 0
    for src in sources:
        for t0 in start_times_h:
            wn = wntr.network.WaterNetworkModel(inp)
            wn.options.time.duration = int(duration_h * 3600)
            wn.options.time.hydraulic_timestep = min(wn.options.time.hydraulic_timestep or 3600, 3600)
            wn.options.time.pattern_timestep = 3600
            wn.options.time.quality_timestep = quality_dt_s
            wn.options.time.report_timestep = report_dt_s
            wn.options.quality.parameter = "CHEMICAL"
            # EPANET prefix-matches its TRACE keyword, so a chemical name
            # starting with "trace" is misread and the run fails (Error 212).
            wn.options.quality.chemical_name = "CONTAM"
            mult = np.zeros(n_pat)
            mult[int(t0):int(t0 + inject_h)] = 1.0
            wn.add_pattern("inj", list(mult))
            wn.add_source("src", src, "SETPOINT", strength_mg_l, "inj")
            res = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=prefix)
            q = res.node["quality"].loc[:, nodes]
            # WNTR reports chemical quality in SI (kg/m^3); 1 mg/L = 1e-3 kg/m^3
            E = q.values > threshold_mg_l * 1e-3
            if times_h is None:
                times_h = np.asarray(q.index, dtype=float) / 3600.0
            blocks.append(E[: len(times_h)])
            scen.append((src, float(t0)))
            done += 1
            if verbose and (done % 100 == 0 or done == total):
                print(f"    {done}/{total} scenarios", flush=True)
    for ext in (".inp", ".rpt", ".bin"):
        try:
            os.remove(prefix + ext)
        except OSError:
            pass
    return np.stack(blocks).astype(bool), nodes, scen, times_h


def save(path, E, nodes, scen, times_h, meta):
    np.savez_compressed(path, E=np.packbits(E, axis=-1), shape=np.array(E.shape),
                        nodes=np.array(nodes), src=np.array([s for s, _ in scen]),
                        t0=np.array([t for _, t in scen]), times_h=times_h,
                        meta=np.array([repr(meta)]))


def load(path):
    z = np.load(path, allow_pickle=False)
    shp = tuple(z["shape"])
    E = np.unpackbits(z["E"], axis=-1, count=shp[-1]).astype(bool).reshape(shp)
    scen = list(zip(z["src"].tolist(), z["t0"].tolist()))
    return E, z["nodes"].tolist(), scen, z["times_h"]
