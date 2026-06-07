# Stage 13: Full-Recourse AC NLP and Multi-Start Benders Loop

## Purpose

This stage moves the reproduction closer to Nasri et al.'s decomposed AC-UC algorithm by adding the missing second-stage adjustment variables, using an explicit constrained Ipopt AC NLP subproblem, and introducing a paper-aligned multi-start strategy for nonconvex AC subproblem solves.

## Implemented Changes

- Added full second-stage adjustment variables to the AC NLP subproblem:
  - generator upward reserve deployment
  - generator downward reserve deployment
  - involuntary active load shedding
  - wind power spillage
- Replaced the fixed active dispatch relation with recourse-coupled equalities:
  - `Pg_ac - Pg_master - reserve_up + reserve_down = 0`
  - `Wind_ac + wind_spill - Wind_master = 0`
- Corrected reserve adjustment bounds:
  - upward reserve is limited by headroom from `Pg_master` to `Pmax`
  - downward reserve is limited by room from `Pg_master` to `Pmin`
- Adjusted the second-stage cost proxy:
  - upward reserve deployment uses the generator marginal cost
  - downward reserve is no longer counted as a positive deployment cost
  - load shedding uses the paper parameter `load_shed_value`
  - wind spillage currently remains zero-cost, consistent with wind production cost being nil
- Added Ipopt multi-start for each AC NLP:
  - `dc_seed`
  - `flat_start`
  - `flat_start_high_v`
- Added attempt-level output for each NLP solve, including status, objective, runtime, and infeasibility measure.
- Added master objective decomposition:
  - startup cost
  - expected dispatch cost
  - expected load shedding cost
  - expected eta cost
- Updated the automatic loop reporting:
  - lower bound is the master objective
  - evaluated upper-bound proxy is reconstructed from first-stage proxy cost plus expected AC recourse from successful NLP subproblems

## Verification

The syntax check passed for:

- `src/ac_subproblem.py`
- `src/run_benders_auto_loop.py`
- `src/dc_uc_baseline.py`

A single AC NLP smoke test for scenario 1, hour 1 solved successfully with Ipopt using three starts. All starts converged; the selected solution was `dc_seed`.

The one-iteration Benders loop completed successfully:

- Output directory: `results/benders_auto_loop_full_recourse_multistart_1iter`
- Runtime: 256.663 seconds
- Iterations: 1
- New cuts generated: 3
- Successful NLP subproblems: 3 / 3
- Failed NLP subproblems: 0
- Lower bound: 639922.356028
- Evaluated upper-bound proxy: 639922.355943
- Reported gap: 0.00000000%

The three selected AC NLP subproblems were:

| Scenario | Hour | NLP objective | Max P residual MW | Max Q residual Mvar | Max AC line loading % | Load shed MW | Wind spill MW | Reserve up MW | Reserve down MW |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 26 | 17 | -0.0017006545 | 0.000102949 | 2.03e-08 | 99.954629 | 0.0 | 0.000194008 | 0.0 | 0.002275450 |
| 21 | 17 | -0.0017005307 | 0.000106231 | 9.55e-10 | 99.027099 | 0.0 | 0.000194151 | 0.0 | 0.002276628 |
| 26 | 24 | -0.0017006398 | 0.000106241 | 1.69e-08 | 99.999898 | 0.0 | 0.000188375 | 0.0 | 0.002326434 |

The generated optimality cuts are now based on coupling multipliers from explicit AC NLP constraints and retain scenario-probability weights for eta variables. The first iteration generated three cuts with 27 terms each.

## Interpretation

This is a meaningful step toward the paper's algorithm, but it should not yet be read as a complete reproduction of Fig. 5 or Table VI. The loop now has the correct workflow shape:

1. solve MILP master,
2. select AC-stressed scenario-hours,
3. solve continuous AC NLP subproblems with multi-start,
4. extract coupling multipliers,
5. generate optimality cuts,
6. feed the cuts back to the master.

The main remaining limitation is that the validation run only solved selected scenario-hours, not all 40 scenarios over 24 hours. Therefore the evaluated upper-bound proxy is still a partial diagnostic rather than the paper's full expected-cost upper curve.

## Remaining Gaps to Paper

- Full scenario-time coverage is not yet active. The paper solves one subproblem per scenario and time period.
- The current master remains a reconstructed synthetic-data DC-UC model, not the original GAMS/CPLEX model.
- The AC branch model is still simplified and lossless; the paper's Appendix A AC functions include fuller admittance-based expressions.
- Ramping and temporal decomposition details are still incomplete relative to the paper's Section III-B heuristic.
- The current cuts use local NLP multipliers. Because the AC subproblem is nonconvex, these cuts are local and not globally valid in the same way as convex Benders cuts.
- The one-iteration run stopped by the proxy tolerance because the partial expected recourse is tiny and negative. For paper-style convergence analysis, the loop should be run with full or broader scenario-hour coverage and a stricter upper-bound interpretation.

## Recommended Next Step

The next implementation stage should replace the current "top stressed scenario-hour selection" with configurable coverage modes:

- `selected`: current fast mode
- `scenario_hour_grid`: solve all requested scenario-hours
- `paper_case_b`: solve 40 scenarios x 24 hours with batching/checkpointing

That will make the convergence data closer to the paper's Case B Fig. 5 and expose whether the local optimality cuts remain numerically useful beyond the first diagnostic iteration.
