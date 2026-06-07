# Benders Automatic Loop Report

- Status: `auto_loop_complete`
- Stop reason: `max_iterations_reached`
- Iterations: 2
- Tolerance: 0.3%
- Cumulative cuts: 6
- Runtime: 199.763 seconds

| Iteration | Lower Bound | Upper Bound Proxy | Gap % | Cuts Active | New Cuts | NLP Solves |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 639922.356028 | 639922.356031 | 0.00000000 | 0 | 3 | 3/3 |
| 2 | 639922.356028 | 770012.402376 | 16.89453909 | 3 | 3 | 2/3 |

This loop follows the paper's sequence at workflow level: master solve, AC subproblem solves, cut generation, and master re-solve. It remains a simplified reproduction because only a small subset of scenario-hours is selected per iteration and the master model is still the current synthetic-data DC-UC formulation.
