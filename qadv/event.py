"""Turn simulated scenarios into one source-identification decision problem.

The situation the judges and Dr. Mousa describe:
  1. A monitoring point A raises an alarm at time t_A.
  2. Many (source node, start time) scenarios are consistent with that alarm.
     These are the "candidate sources", each with a probability.
  3. K crews can each take one grab sample, arriving at t_A + delay.
  4. Which K sites should they sample so the results best pin down the source?

Output: an outcome matrix O[m, n] -- what a sample at site i would show (1 =
contaminated) if scenario p were the truth -- plus a weight per scenario and
the source node each scenario belongs to. That is the full input to both the
classical and quantum optimisers. No water knowledge is needed after this.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Event:
    alarm_node: str
    t_alarm_h: float
    t_sample_h: float
    O: np.ndarray            # (m, n) uint8: outcome of sampling site j under scenario p
    w: np.ndarray            # (m,) prior weight of each plausible scenario, sums to 1
    src: np.ndarray          # (m,) int: source-node id of each scenario
    src_names: list          # source-node id -> junction name
    site_names: list         # column j -> junction name
    scen: list = field(default_factory=list)

    @property
    def m(self):
        return self.O.shape[0]

    @property
    def n(self):
        return self.O.shape[1]

    @property
    def n_sources(self):
        return len(self.src_names)


def first_arrival(E, times_h):
    """T[s, node] = first report time the node exceeds threshold, inf if never."""
    hit = E.any(axis=1)
    idx = E.argmax(axis=1)
    T = np.where(hit, times_h[idx], np.inf)
    return T


def build_event(E, nodes, scen, times_h, alarm_node, t_alarm_h,
                tol_h=0.25, delay_h=1.0, prior=None):
    """Scenarios consistent with an alarm at `alarm_node` first seen at `t_alarm_h`."""
    a = nodes.index(alarm_node)
    T = first_arrival(E, times_h)
    plaus = np.where(np.abs(T[:, a] - t_alarm_h) <= tol_h + 1e-9)[0]
    t_s = t_alarm_h + delay_h
    k = int(np.argmin(np.abs(times_h - t_s)))
    O = E[plaus, k, :].astype(np.uint8)
    O = np.delete(O, a, axis=1)                       # no point re-sampling the alarm
    sites = [nd for i, nd in enumerate(nodes) if i != a]
    srcs = [scen[p][0] for p in plaus]
    names = sorted(set(srcs), key=srcs.index)
    sid = np.array([names.index(s) for s in srcs])
    w = np.ones(len(plaus)) if prior is None else np.array([prior(scen[p]) for p in plaus], float)
    w = w / w.sum()
    return Event(alarm_node, t_alarm_h, float(times_h[k]), O, w, sid, names, sites,
                 [scen[p] for p in plaus])


def informative(ev):
    """Drop sites whose sample gives the same result for every candidate (useless)."""
    keep = np.where(ev.O.min(0) != ev.O.max(0))[0]
    return Event(ev.alarm_node, ev.t_alarm_h, ev.t_sample_h, ev.O[:, keep], ev.w, ev.src,
                 ev.src_names, [ev.site_names[j] for j in keep], ev.scen)


def subsample_sites(ev, n, rng):
    keep = np.sort(rng.choice(ev.n, size=n, replace=False))
    return Event(ev.alarm_node, ev.t_alarm_h, ev.t_sample_h, ev.O[:, keep], ev.w, ev.src,
                 ev.src_names, [ev.site_names[j] for j in keep], ev.scen)


def candidate_events(E, nodes, scen, times_h, min_sources=10, tol_h=0.25, delay_h=1.0):
    """Every (alarm node, alarm time) whose plausible set spans >= min_sources source nodes."""
    T = first_arrival(E, times_h)
    srcs = np.array([s for s, _ in scen])
    out = []
    for a, nd in enumerate(nodes):
        col = T[:, a]
        for t in np.unique(col[np.isfinite(col)]):
            if t + delay_h > times_h[-1]:
                continue
            p = np.where(np.abs(col - t) <= tol_h + 1e-9)[0]
            ns = len(set(srcs[p]))
            if ns >= min_sources:
                out.append((nd, float(t), len(p), ns))
    return out
