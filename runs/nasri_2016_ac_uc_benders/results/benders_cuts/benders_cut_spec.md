# Benders Cut Specification

## Current Candidate Cut Template

The current implementation records feasibility-cut candidates from AC subproblem diagnostics.
Rows are not added back to the MILP master yet because they do not contain dual-derived coefficients.

Generic feasibility cut form:

```text
0 >= phi(x_bar) + lambda_u * (u - u_bar) + lambda_p * (p - p_bar) + lambda_w * (w - w_bar)
```

where:

- `x_bar` is the current master solution.
- `phi(x_bar)` is the AC subproblem infeasibility measure.
- `lambda_u`, `lambda_p`, and `lambda_w` are dual multipliers/sensitivities for fixed master quantities.
- `u`, `p`, and `w` are master commitment, dispatch, and scheduled wind variables.

## Current Cut Pool Summary

- Candidate cuts: 960
- Added cuts: 0
- Largest violation metric: 516.797544
- Mean violation metric: 417.145214

## Required Next Step

Use the lower-level `cyipopt.Problem` interface with explicit equality and inequality constraints, then extract Ipopt multipliers for fixed-master constraints. Those multipliers replace the proxy fields in `case_b_cut_pool.csv` and make the cuts valid for master re-optimization.
