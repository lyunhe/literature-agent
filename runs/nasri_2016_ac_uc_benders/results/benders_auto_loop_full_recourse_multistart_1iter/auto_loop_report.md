# Benders Automatic Loop Report

- Status: `auto_loop_complete`
- Stop reason: `tolerance_reached`
- Iterations: 1
- Tolerance: 0.3%
- Cumulative cuts: 3
- Unique cut subproblems: 3
- Failed subproblems seen: 0
- Runtime: 256.663 seconds

| Iteration | Lower Bound | Evaluated Upper Bound | Gap % | Cuts Active | New Cuts | NLP Solves | Failed NLP | Skipped Cut/Failed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 639922.356028 | 639922.355943 | 0.00000000 | 0 | 3 | 3/3 | 0 | 0/0 |

This loop follows the paper's sequence at workflow level: master solve, AC subproblem solves, cut generation, and master re-solve. Failed AC NLP rows are retained as diagnostics but are excluded from cut generation. The reported evaluated upper bound is a reconstructed paper-style proxy built from first-stage cost plus expected AC recourse over the successful NLP evaluations.
