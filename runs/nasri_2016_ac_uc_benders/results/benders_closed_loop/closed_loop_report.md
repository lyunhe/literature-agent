# Benders Closed-Loop Smoke Test

- Cut headers: `/Users/yunhe/paper-reconstruct/runs/nasri_2016_ac_uc_benders/results/benders_cuts/case_b_benders_cut_constraints.csv`
- Cut terms: `/Users/yunhe/paper-reconstruct/runs/nasri_2016_ac_uc_benders/results/benders_cuts/case_b_benders_cut_terms.csv`
- Objective delta: 1.2188684195280075e-07

| Iteration | Master | Objective | Cuts Added | Eta Variables | Status |
|---:|---|---:|---:|---:|---|
| 1 | dc_uc_master_no_ac_cuts | 639922.356028 | 0 | 0 | solved |
| 2 | dc_uc_master_with_ac_benders_cuts | 639922.356028 | 3 | 3 | solved |

This is the first actual master re-solve with generated Benders-form rows. The current cut coefficients are small because the constrained AC NLP subproblems are nearly feasible after voltage/reactive optimization.
