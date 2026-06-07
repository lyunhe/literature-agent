# Paper-Style Result Tables

These outputs mirror the paper's result-display structure, but current Case B/C values are AC screening diagnostics rather than solved AC-Benders optima.

## Table V-Style Summary

| case | network_model | algorithm | objective | runtime_sec | iterations | ac_screened_violations | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | DC | direct extensive-form MILP | 639922.3560276568 | 27.957626916002482 | 1 |  | Synthetic wind data; current implemented model. |
| B | AC screening V=[0.9,1.1] | Benders-style one-pass screening | 639922.3560276568 |  | 1 | 960 | Cut pool generated; not directly comparable to original AC-Benders optimum. |
| B_tol_0.1 | AC screening V=[0.9,1.1] | Benders-style tolerance sensitivity | 639922.3560276568 |  | 1 | 960 | Paper-style 0.1% tolerance row; current one-pass proxy already below 0.1%. |
| C | AC screening V=[0.5,1.5] | Benders-style one-pass screening | 639922.3560276568 |  | 1 | 960 | Voltage bounds do not bind with fixed V=1.0 screening. |

## AC Screening Summary

- Case B screened violations: 960 / 960
- Case B max AC line loading: 99.952%
- Case B max reactive violation: 516.798 Mvar
- Case C screened violations: 960 / 960
- Case C max AC line loading: 99.952%
- Case C max reactive violation: 516.798 Mvar

## Fig. 5-Style Convergence Log

| iteration | lower_bound | upper_bound | relative_gap_percent | master_status | subproblem_status | cuts_added | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 639922.3560276568 | 640439.1535720726 | 0.08069424574268301 | solved | ac_screening_complete | 960 | AC screening pass only; feasibility cuts are counted but not added as dual-derived cuts. |

## Benders Cut Pool

| iteration | case_id | candidate_cuts | added_cuts | largest_violation_metric | mean_violation_metric | cut_status | blocking_item |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | case_b_ac_uc_benders | 960 | 0 | 516.7975444158053 | 417.1452136453419 | proxy_candidates_only | Need AC NLP dual multipliers and explicit fixed-master constraints. |
