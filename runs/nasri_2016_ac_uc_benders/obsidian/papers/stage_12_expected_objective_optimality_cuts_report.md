# Stage 12 - Expected Objective and Coupling-Based Optimality Cuts

## Purpose

This stage aligns the Benders loop more closely with the paper's stochastic expected-cost structure. The AC subproblem cuts now enter the master through scenario-probability-weighted eta variables, so the master objective represents an expected second-stage contribution rather than an unweighted sum of selected AC subproblem penalties.

## Implementation

Updated files:

- `src/generate_benders_cut_constraints.py`
- `src/dc_uc_baseline.py`
- `src/run_benders_auto_loop.py`

The generated cut type is now:

```text
optimality_cut
```

The cut algebra remains:

```text
eta_ac_s_t >= phi_s_t(x_bar) + sum_i beta_i * (x_i - xbar_i)
```

where `beta_i` is extracted from explicit fixed-master coupling equalities:

```text
Pg_ac[g]   - Pg_master_bar[g]   = 0
Wind_ac[w] - Wind_master_bar[w] = 0
```

For each scenario-hour cut, the header now includes:

```text
scenario_probability
eta_objective_weight
```

The master reads `eta_objective_weight` when creating each eta variable. Thus, each scenario's AC recourse approximation contributes to the objective with its probability from Table IV.

## Expected Objective Smoothing

The implementation now reflects the paper's expected-value smoothing idea at the master objective level:

```text
min first_stage_cost + expected_dispatch_cost + sum_s probability_s * eta_s
```

In the current selected-subproblem loop, only solved scenario-hours receive eta cuts. This is still a sampled/selected approximation, but the weighting is now consistent with the stochastic expected objective.

## Verification Run

Command:

```bash
/Users/yunhe/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 src/run_benders_auto_loop.py \
  --max-iterations 2 \
  --min-iterations 2 \
  --cuts-per-iteration 3 \
  --out-dir ../results/benders_auto_loop_expected_optimality_v2
```

Main outputs:

- `results/benders_auto_loop_expected_optimality_v2/auto_loop_report.md`
- `results/benders_auto_loop_expected_optimality_v2/auto_loop_iteration_log.csv`
- `results/benders_auto_loop_expected_optimality_v2/cumulative_cuts/benders_cut_constraints.csv`
- `results/benders_auto_loop_expected_optimality_v2/cumulative_cuts/benders_cut_terms.csv`

## Result Summary

| Iteration | Lower Bound | Expected AC Phi | Gap % | Active Cuts | New Cuts | NLP Solves |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 639922.356028 | 3.224844e-11 | 0.00000000 | 0 | 3 | 3/3 |
| 2 | 639922.356028 | 4.541603e-11 | 0.00000000 | 3 | 3 | 3/3 |

Generated cut headers include:

```text
cut_type = optimality_cut
scenario_probability = probability_s
eta_objective_weight = probability_s
source_constraint = fixed_master_dispatch_eq / fixed_master_wind_eq
```

## Interpretation

This is now an exact master-coupling optimality cut with respect to the implemented slack-penalty AC NLP and the local Ipopt KKT multipliers. It is no longer the earlier nodal-balance proxy.

However, it should not be described as a globally valid convex Benders cut, because the AC NLP remains nonconvex. This matches the paper's own caveat: the decomposition is justified through expected-objective smoothing, numerical behavior, and multi-start mitigation rather than a strict global convexity proof.

## Remaining Work

1. Expand from selected scenario-hours to a broader or full 40 x 24 subproblem pass.
2. Add multi-start AC NLP/decomposition runs to mirror the paper's nonconvexity mitigation.
3. Add commitment coupling or a disciplined treatment of offline units.
4. Replace the slack-penalty recourse approximation with the paper's fuller real-time operating-cost subproblem if exact numerical replication is required.
