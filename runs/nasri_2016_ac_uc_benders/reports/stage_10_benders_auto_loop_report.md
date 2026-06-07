# Stage 10 - Automatic Benders Loop

## Purpose

This stage turns the manual Benders sequence into an automatic loop:

1. Solve the DC-UC master problem with the current cumulative AC cuts.
2. Screen all scenario-hour AC subproblems using the fixed master solution.
3. Select the highest-violation scenario-hours that have not already produced cuts.
4. Solve selected AC NLP subproblems with explicit Ipopt constraints and slack variables.
5. Generate Benders-form optimality proxy cuts only from successful NLP solves.
6. Repeat until the proxy gap is below the configured threshold or the iteration cap is reached.

## Implementation Changes

- Script updated: `src/run_benders_auto_loop.py`
- Failed Ipopt NLP rows are now retained in batch CSVs as diagnostics but excluded from:
  - dual coefficient generation,
  - Benders cut construction,
  - upper-bound/gap proxy calculation.
- Scenario-hour selection now prioritizes fresh subproblems:
  - previously successful cut subproblems are skipped first,
  - previously failed subproblems are also skipped when enough fresh candidates exist.
- Per-iteration logs now record:
  - successful and failed NLP counts,
  - maximum successful NLP objective,
  - maximum failed NLP objective,
  - skipped previously cut/failed candidates,
  - unique cut subproblem count.

## Stable Run

Command:

```bash
/Users/yunhe/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 src/run_benders_auto_loop.py \
  --max-iterations 3 \
  --min-iterations 2 \
  --cuts-per-iteration 3 \
  --out-dir ../results/benders_auto_loop_stable
```

Main outputs:

- `results/benders_auto_loop_stable/auto_loop_report.md`
- `results/benders_auto_loop_stable/auto_loop_iteration_log.csv`
- `results/benders_auto_loop_stable/auto_loop_result.json`
- `results/benders_auto_loop_stable/cumulative_cuts/benders_cut_constraints.csv`
- `results/benders_auto_loop_stable/cumulative_cuts/benders_cut_terms.csv`

## Result Summary

| Iteration | Lower Bound | Upper Bound Proxy | Gap % | Active Cuts | New Cuts | NLP Solves | Failed NLP |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 639922.356028 | 639922.356031 | 0.00000000 | 0 | 3 | 3/3 | 0 |
| 2 | 639922.356028 | 639922.356028 | 0.00000000 | 3 | 3 | 3/3 | 0 |

The stable loop stopped after iteration 2 with `tolerance_reached`.

## Interpretation

The workflow-level Benders loop is now closed and automatic. The master accepts cumulative AC cuts, the selected AC NLP subproblems expose Ipopt multipliers, and new cuts are fed back into the next master solve.

This is still a simplified reproduction rather than an exact numerical reproduction of Nasri et al.:

- wind scenarios are synthetic calibrated data,
- only a selected subset of scenario-hours is solved per iteration,
- the master is still the current DC-UC formulation,
- the AC subproblem uses a lossless AC approximation with explicit P/Q balance and line-flow constraints,
- the current cuts are optimality proxy cuts from balance multipliers, not yet the paper's full feasibility/optimality cut set.

## Next Work

1. Add exact fixed-master equality rows inside the AC NLP so cut multipliers are directly associated with master variables.
2. Separate paper-style feasibility cuts from optimality cuts.
3. Expand the batch from 3 selected scenario-hours to a configurable larger subset once runtime is acceptable.
4. Add convergence plots comparable to the paper's Benders iteration table/figure.
