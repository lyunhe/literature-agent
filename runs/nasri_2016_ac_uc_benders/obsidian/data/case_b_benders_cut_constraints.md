# Benders Cut Constraints

- Cuts generated: 3
- Term rows: 102
- LP text: `/Users/yunhe/paper-reconstruct/runs/nasri_2016_ac_uc_benders/results/benders_cuts/case_b_benders_cuts.lp.txt`

Generated cut form:

```text
eta_ac_s_t >= phi(x_bar) + sum_i beta_i * (x_i - xbar_i)
```

Equivalent LP row:

```text
eta_ac_s_t - sum_i beta_i * x_i >= phi(x_bar) - sum_i beta_i * xbar_i
```

| Cut | Scenario | Hour | Phi | RHS Constant | Terms | Status |
|---|---:|---:|---:|---:|---:|---|
| BC-0001 | 26 | 17 | 6.439760e-10 | 6.385382e-08 | 34 | generated_not_added_to_master |
| BC-0002 | 21 | 17 | 6.440730e-10 | 1.700691e-06 | 34 | generated_not_added_to_master |
| BC-0003 | 26 | 24 | 2.992576e-06 | -1.477209e-02 | 34 | generated_not_added_to_master |

These rows are now in Benders algebraic form and use master variable names from the HiGHS model. They still need eta variables and cut insertion in the master solve loop.
