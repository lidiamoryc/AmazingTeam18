# Pre-registration: quantum advantage test

Written and committed before any quantum result was computed. Any deviation
from this plan is listed in the results under "Deviations".

## Question

On the contamination **source-identification sampling** problem built from
real EPANET hydraulics, is there any regime where XY-mixer QAOA's
time-to-solution scales better than the best classical method?

## Instances

- Network: EPANET Net3 (92 junctions), 552 scenarios (every junction x 6 unknown
  start times), 24 h, 2 h injection at 1000 mg/L, 0.1 mg/L detection threshold.
- Events: (alarm node, alarm time) with >= 20 candidate source nodes.
  Crew sampling delay 1 h, alarm-time tolerance 15 min, uniform prior.
- Each instance = one event + a random subset of n of its informative sites.
  12 test instances per n, seed 2026. Training instances for QAOA angles are
  drawn from DIFFERENT events (seed 7) and never used for testing.

## Families

- F1, crews grow with the network: K = n/4, n in {12, 16, 20, 24, 28}.
- F2, fixed crews (the operational case): K = 5, n in {12, 16, ..., 40}.

## Objective

QUBO with lam = 1 (the second Bonferroni lower bound on pair coverage, so no
tuning). Ties at the optimum count as optimal.

## Quantum method

XY ring mixer, Dicke initial state, p in {1, 2, 4, 8}. Fixed angles: the mean
of per-instance optimised angles over the training set (n in {10, 12, 14},
cost normalised to zero mean and unit std over feasible plans), following
Shaydulin et al. (2024). Also per-instance optimised angles for n <= 20, as an
optimistic upper bound. Noiseless simulation.

## Metrics

- Quantum: TTS = 1/p_opt shots (Shaydulin et al.). QAOA + quantum minimum
  finding: TTS proportional to 1/sqrt(p_opt).
- Classical: simulated annealing, tabu search, random-restart local search,
  each over 40 seeds per instance with a fixed per-run budget; TTS = mean run
  time / success rate. Plus greedy + local search, exact MIP solve time
  (HiGHS), and brute force C(n, K).
- Fit log2(TTS) = a + b*n on the per-n median; 95% CI by bootstrap over
  instances. Scaling base = 2^b.

## Decision rule

"Evidence of scaling advantage" if and only if the upper end of a quantum
method's 95% CI on its base lies below the lower end of the best classical
method's CI. Otherwise: "no evidence of advantage". Results for every family,
every p and every method are reported, whichever way they come out.
