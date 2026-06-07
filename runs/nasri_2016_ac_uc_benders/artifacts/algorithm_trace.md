# Algorithm Trace: Network-Constrained AC Unit Commitment Under Uncertainty: A Benders' Decomposition Approach

## Recommended Implementation Path

1. Build deterministic UC model from the appendix.
2. Implement reduced C&CG master problem with an initial scenario set.
3. Solve robust feasibility subproblem for the current commitment.
4. If infeasible, add the returned worst-case scenario to the master.
5. If feasible, solve optimality subproblem for worst-case dispatch cost.
6. Update lower/upper bounds and stop when the relative gap meets tolerance.
7. Add LSF cutting-plane to the master to include only violated transmission constraints.
8. Treat subproblem column-generation heuristics as a later acceleration.

## Paper-to-Code Map

| Paper object | Implementation function | Inputs | Outputs | Notes |
|---|---|---|---|---|
| Deterministic UC appendix | `build_deterministic_uc()` | generator data, load, wind, LSF, reserves | MILP model | First model to validate. |
| Reduced robust master | `solve_master(scenarios, active_lines)` | scenario pool, active TLC set | commitment, dispatch recourse, lower bound | C&CG master. |
| Feasibility subproblem | `solve_feasibility_subproblem(commitment)` | commitment, uncertainty set | violating scenario or feasible flag | Add scenario if dispatch impossible. |
| Optimality subproblem | `solve_optimality_subproblem(commitment)` | commitment, uncertainty set | worst-case scenario, dispatch cost, upper bound | MILP via extreme-point/big-M reformulation. |
| LSF cutting-plane | `update_active_lines(solution)` | dispatch flows, LSF matrix, line limits | violated/critical lines | Use before subproblem column generation. |
| Experiment runner | `run_ccg(config)` | case data, solver config, tolerance | logs, solution, alignment metrics | Persist every iteration. |

## Minimum Iteration Log

| Field | Description |
|---|---|
| iteration | C&CG iteration index |
| lower_bound | Master objective |
| upper_bound | Commitment cost plus worst-case dispatch cost |
| relative_gap | Stopping metric |
| feasibility_status | Robustly feasible or violating scenario found |
| scenario_id | Worst-case scenario added or evaluated |
| active_line_count | Number of TLCs retained by LSF cutting-plane |
| master_runtime_sec | Master solve time |
| subproblem_runtime_sec | Subproblem solve time |
