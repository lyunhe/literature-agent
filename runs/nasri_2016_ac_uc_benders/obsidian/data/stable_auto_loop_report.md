# Benders Automatic Loop Report

- Status: `auto_loop_complete`
- Stop reason: `tolerance_reached`
- Iterations: 2
- Tolerance: 0.3%
- Cumulative cuts: 6
- Unique cut subproblems: 6
- Failed subproblems seen: 0
- Runtime: 200.323 seconds

| Iteration | Lower Bound | Upper Bound Proxy | Gap % | Cuts Active | New Cuts | NLP Solves | Failed NLP | Skipped Cut/Failed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 639922.356028 | 639922.356031 | 0.00000000 | 0 | 3 | 3/3 | 0 | 0/0 |
| 2 | 639922.356028 | 639922.356028 | 0.00000000 | 3 | 3 | 3/3 | 0 | 3/0 |

This loop follows the paper's sequence at workflow level: master solve, AC subproblem solves, cut generation, and master re-solve. Failed AC NLP rows are retained as diagnostics but are excluded from cut generation and the gap proxy. It remains a simplified reproduction because only a small subset of scenario-hours is selected per iteration and the master model is still the current synthetic-data DC-UC formulation.
