# Stage 7 Benders Cut Pool and Paper-Style Tests Report

## What Was Added

This stage adds explicit Benders cut artifacts and re-runs the paper-style experiment display.

Implemented outputs:

- Feasibility-cut candidate pool
- Cut summary
- Cut formula/specification note
- Table V-style summary including Case A, Case B, Case C, and Case B 0.1% tolerance sensitivity
- Table VI-style commitment matrix
- Fig. 4-style generation/wind schedule data
- Fig. 5-style convergence log

The current cuts are still proxy candidates. They are not added back to the master because the current AC NLP prototype does not yet expose dual-derived coefficients for fixed master constraints.

## Benders Cut Template

Current intended feasibility cut form:

```text
0 >= phi(x_bar)
   + lambda_u * (u - u_bar)
   + lambda_p * (p - p_bar)
   + lambda_w * (w - w_bar)
```

where:

- `x_bar` is the current master solution.
- `phi(x_bar)` is the AC subproblem infeasibility metric.
- `lambda_u`, `lambda_p`, and `lambda_w` are dual multipliers for fixed commitment, dispatch, and wind schedule constraints.
- The current cut pool stores `phi(x_bar)` and the relevant scenario/hour, but not yet the dual coefficients.

## Cut Pool Result

- Iteration: 1
- Case: Case B
- Candidate cuts: 960
- Added cuts: 0
- Largest violation metric: 516.797544
- Mean violation metric: 417.145214
- Blocking item: need AC NLP dual multipliers and explicit fixed-master constraints.

Largest cut candidates:

| Cut | Scenario | Hour | Violation Metric |
|---|---:|---:|---:|
| FC-001-0001 | 26 | 17 | 516.797544 |
| FC-001-0002 | 21 | 17 | 514.949521 |
| FC-001-0003 | 26 | 24 | 514.595381 |

## Paper-Style Tests

The same result-display structure as the paper has been generated:

| Case | Model | Algorithm | Objective | Iterations | Notes |
|---|---|---|---:|---:|---|
| A | DC | Direct extensive-form MILP | 639,922.3560 | 1 | Synthetic wind data |
| B | AC V=[0.9,1.1] | One-pass Benders-style screening | 639,922.3560 | 1 | Cut pool generated |
| B_tol_0.1 | AC V=[0.9,1.1] | Tolerance sensitivity row | 639,922.3560 | 1 | Current proxy gap below 0.1% |
| C | AC V=[0.5,1.5] | One-pass Benders-style screening | 639,922.3560 | 1 | Fixed-voltage screen, same as B |

Fig. 5-style convergence:

| Iteration | Lower Bound | Upper Bound Proxy | Gap | Cuts Added |
|---:|---:|---:|---:|---:|
| 1 | 639,922.3560 | 640,439.1536 | 0.080694% | 960 |

## Output Files

- `results/benders_cuts/case_b_cut_pool.csv`
- `results/benders_cuts/case_b_cut_summary.csv`
- `results/benders_cuts/benders_cut_spec.md`
- `results/paper_style_results/table_v_summary.csv`
- `results/paper_style_results/table_vi_commitment.csv`
- `results/paper_style_results/fig4_generation_schedule.csv`
- `results/paper_style_results/fig5_benders_convergence.csv`
- `results/paper_style_results/paper_style_report.md`

## Interpretation

The workflow now mirrors the paper's experiment display, but it is still a synthetic-data and proxy-cut reproduction. The missing piece for true paper-level Benders is not the outer experiment harness; it is the lower-level AC NLP formulation with explicit constraints and Ipopt multipliers.
