# Stage 11 - Explicit Master-Coupling Duals

## Purpose

This stage replaces the earlier P-balance-multiplier proxy with a more standard Benders coupling structure. The AC NLP now contains explicit copies of selected master active-power decisions and fixed equality constraints:

```text
Pg_ac[g]   - Pg_master_bar[g]   = 0
Wind_ac[w] - Wind_master_bar[w] = 0
```

The Ipopt multipliers of these fixed-master equalities are exported and converted into Benders cut coefficients.

## Implementation

Updated files:

- `src/ac_subproblem.py`
- `src/run_benders_auto_loop.py`
- `src/build_dual_cut_coefficients.py`

The constrained AC NLP now includes:

- voltage angle variables,
- voltage magnitude variables,
- reactive generation variables,
- active generation coupling variables,
- wind generation coupling variables,
- P/Q balance slack variables,
- line MVA slack variables,
- explicit fixed-master equality constraints.

The exported multiplier table now includes rows such as:

```text
constraint_type = fixed_master_dispatch_eq
constraint_type = fixed_master_wind_eq
```

For a coupling equality of the form:

```text
p_ac - p_master_bar = 0
```

the Benders coefficient is currently exported as:

```text
benders_coefficient = - multiplier
```

This follows the derivative of the subproblem value with respect to the fixed right-hand-side master value.

## Degeneracy Handling

An initial coupled run fixed all generator dispatch variables, including units with zero dispatch. This created degeneracy because `Pg_ac = 0` and the nonnegative lower bound were active simultaneously. The final coupled-active version only creates active-power coupling rows for online or positive-dispatch units. Closed/offline unit effects should later be represented through commitment coupling and feasibility cuts rather than zero-dispatch equality multipliers.

## Coupled-Active Run

Command:

```bash
/Users/yunhe/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 src/run_benders_auto_loop.py \
  --max-iterations 2 \
  --min-iterations 2 \
  --cuts-per-iteration 3 \
  --out-dir ../results/benders_auto_loop_coupled_active
```

Main outputs:

- `results/benders_auto_loop_coupled_active/auto_loop_report.md`
- `results/benders_auto_loop_coupled_active/auto_loop_iteration_log.csv`
- `results/benders_auto_loop_coupled_active/cumulative_cuts/benders_cut_constraints.csv`
- `results/benders_auto_loop_coupled_active/cumulative_cuts/benders_cut_terms.csv`

## Result Summary

| Iteration | Lower Bound | Upper Bound Proxy | Gap % | Active Cuts | New Cuts | NLP Solves | Failed NLP |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 639922.356028 | 639922.356028 | 0.00000000 | 0 | 3 | 3/3 | 0 |
| 2 | 639922.356028 | 639922.356028 | 0.00000000 | 3 | 3 | 3/3 | 0 |

The coupled-active loop stopped after iteration 2 with `tolerance_reached`.

## Cut Source Check

The generated coefficient file now reports:

```text
source_constraint = fixed_master_dispatch_eq
source_constraint = fixed_master_wind_eq
```

This confirms that the cut terms are no longer inferred indirectly from nodal P-balance multipliers. They are extracted from explicit master-coupling equalities.

## Remaining Gap to Full Paper-Style Cuts

This is a stronger Benders implementation, but not yet the complete Nasri et al. cut system:

1. Commitment variables still need explicit coupling or a clean decomposition rule.
2. Feasibility cuts need a separate infeasibility/restoration subproblem and not just slack-penalty optimality cuts.
3. The AC NLP remains nonconvex, so Ipopt multipliers are local KKT multipliers rather than global dual certificates.
4. The current loop still solves only selected scenario-hours per iteration.
5. Synthetic wind scenarios remain a reproduction limitation.
