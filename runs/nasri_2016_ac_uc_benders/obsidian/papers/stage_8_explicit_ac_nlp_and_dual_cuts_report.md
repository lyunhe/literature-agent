# Stage 8 Explicit AC NLP and Dual-Cut Draft Report

## What Changed

The AC subproblem now has an explicit constrained Ipopt formulation:

- Backend: `cyipopt_constrained`
- Interface: `cyipopt.Problem`
- Objective: minimize explicit slack penalties
- Equality constraints:
  - Active-power balance at every bus
  - Reactive-power balance at every bus
- Inequality constraints:
  - Apparent line-flow limits
- Slack variables:
  - Positive/negative active-power balance slack
  - Positive/negative reactive-power balance slack
  - Line MVA slack

This replaces the earlier black-box residual penalty when the `cyipopt_constrained` backend is selected.

## Single-Subproblem Result

Command:

```bash
python run_reproduction.py --experiment ac-subproblem --scenario-id 1 --hour 1 --solve --ac-nlp-solver cyipopt_constrained
```

Result:

- Status: `ac_constrained_nlp_solved`
- Ipopt status code: 0
- Objective: 6.439440e-10
- Variables: 204
- Constraints: 86
- Max P residual: 2.84e-12 MW
- Max Q residual: 7.16e-10 Mvar
- Sum P slack: 0.000240 MW
- Sum Q slack: 0.000240 Mvar
- Max line slack: 0.025670 MVA
- Max AC line loading: 46.887%

Output files:

- `results/ac_nlp_subproblem/case_b_ac_uc_benders_cyipopt_constrained_scenario_1_hour_1_nlp_solution.csv`
- `results/ac_nlp_subproblem/case_b_ac_uc_benders_cyipopt_constrained_scenario_1_hour_1_nlp_constraints.csv`
- `results/ac_nlp_subproblem/case_b_ac_uc_benders_cyipopt_constrained_scenario_1_hour_1_nlp_multipliers.csv`
- `results/ac_nlp_subproblem/case_b_ac_uc_benders_cyipopt_constrained_scenario_1_hour_1_nlp_summary.json`

## Worst-Reactive Batch

Command:

```bash
python run_ac_nlp_batch.py --limit 3 --selection worst-reactive --ac-nlp-solver cyipopt_constrained
```

Results:

| Scenario | Hour | Screening Q Violation | NLP P Residual | NLP Q Residual | NLP Line Loading |
|---:|---:|---:|---:|---:|---:|
| 26 | 17 | 516.798 | 0.000000 | 0.000000 | 96.455% |
| 21 | 17 | 514.950 | 0.000000 | 0.000000 | 98.549% |
| 26 | 24 | 514.595 | 0.000392 | 0.000000 | 100.000% |

All three selected subproblems solved successfully with explicit constraints and Ipopt multipliers.

## Dual-Cut Coefficient Draft

First-pass cut coefficients were generated from Ipopt multipliers:

- `results/benders_cuts/case_b_dual_cut_coefficients.csv`
- `results/benders_cuts/case_b_dual_cut_coefficients.md`

Rows: 174

| Family | Count | Min | Max | Mean |
|---|---:|---:|---:|---:|
| fixed_dispatch_active_power | 96 | -3.405177e-06 | 7.832955e-05 | 1.419122e-06 |
| fixed_wind_active_power | 6 | -3.405687e-06 | 6.212335e-10 | -1.134968e-06 |
| reactive_balance_sensitivity | 72 | -7.283523e-09 | 2.477219e-09 | -1.331959e-09 |

The active-power balance multipliers are mapped to generator dispatch and wind injection variables at the same bus. Reactive-balance multipliers are currently diagnostic because the master does not yet include reactive decision variables.

## Remaining Work for Real Benders Cuts

The explicit constrained NLP now makes multipliers available. The remaining steps are:

1. Add fixed-master constraints directly into the AC subproblem for the exact master quantities used in the cut.
2. Decide the master variables that receive cut coefficients: commitment, scheduled dispatch, reserve, wind schedule, or eta.
3. Add the generated cut rows to the HiGHS master model and re-solve.
4. Repeat until the Fig. 5-style convergence log is produced by real lower/upper bound updates rather than the current proxy.
