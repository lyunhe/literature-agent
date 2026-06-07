# Stage 4 Benders and AC Subproblem Screening Report

## Scope

This stage extends the workflow from a solved DC-UC master to a Benders-style AC feasibility screening pass.

Implemented:

- Re-solve Case A DC-UC master with HiGHS.
- Export master bus angles for every scenario-hour.
- Evaluate all 40 x 24 scenario-hour AC subproblems for Case B and Case C.
- Compute lossless AC active/reactive branch flow diagnostics.
- Check apparent line loading and reactive generation capability.
- Export paper-style result tables matching the structure of Table V, Table VI, Fig. 4, and Fig. 5.

Not yet implemented:

- AC NLP optimization with voltage magnitude variables.
- Dual multipliers/sensitivities from AC subproblems.
- True Benders feasibility and optimality cuts.
- Iterative master re-solve after adding cuts.
- Original Fig. 3 wind scenarios.

## Benders-Style Run

- Command: `python run_reproduction.py --experiment benders --solve`
- Status: `benders_screening_complete`
- Total runtime: 51.954 s
- Master objective: 639,922.3560
- Master solver: HiGHS
- Master variables: 80,256
- Master constraints: 159,168
- Master solve runtime: 29.766 s

## Case B AC Screening

- Voltage setting: Case B nominal bounds [0.9, 1.1], screened at fixed 1.0 p.u.
- Subproblems evaluated: 960
- Screened feasible: 0
- Screened violations: 960
- Maximum AC apparent line loading: 99.952%
- AC overloaded line count: 0 in every screened subproblem
- Mean reactive violation proxy: 417.145 Mvar
- Maximum reactive violation proxy: 516.798 Mvar

The DC dispatch respects active line limits, but it does not satisfy the reactive-power capability screen under the fixed-voltage, lossless-AC approximation. In a full implementation, the AC NLP subproblem would adjust voltages, reactive generation, load shedding, and wind spillage, then return dual sensitivities for Benders cuts.

## Case C AC Screening

- Voltage setting: Case C relaxed bounds [0.5, 1.5], screened at fixed 1.0 p.u.
- Subproblems evaluated: 960
- Screened feasible: 0
- Screened violations: 960
- Maximum AC apparent line loading: 99.952%
- Maximum reactive violation proxy: 516.798 Mvar

Case B and Case C are identical in this screening pass because voltage magnitudes are fixed at 1.0 p.u. The relaxed voltage range will only matter after an AC NLP backend allows voltage magnitudes to move.

## Fig. 5-Style Convergence Data

| Iteration | Lower Bound | Upper Bound Proxy | Relative Gap % | Master Status | Subproblem Status | Cuts Needed |
|---:|---:|---:|---:|---|---|---:|
| 1 | 639,922.3560 | 640,439.1536 | 0.0807 | solved | ac_screening_complete | 960 |

This is a one-pass diagnostic log, not the original paper's iterative convergence curve. The paper reports Case B convergence in 25 iterations at 0.3% tolerance; reproducing that curve requires AC NLP duals and iterative cut addition.

## Paper-Style Output Files

- `results/paper_style_results/table_v_summary.csv`
- `results/paper_style_results/table_vi_commitment.csv`
- `results/paper_style_results/fig4_generation_schedule.csv`
- `results/paper_style_results/fig5_benders_convergence.csv`
- `results/paper_style_results/paper_style_report.md`

## AC Subproblem Output Files

- `results/ac_subproblem/case_b_ac_uc_benders_subproblem_summary.csv`
- `results/ac_subproblem/case_b_ac_uc_benders_line_eval.csv`
- `results/ac_subproblem/case_b_ac_uc_benders_reactive_eval.csv`
- `results/ac_subproblem/case_c_ac_uc_relaxed_voltage_subproblem_summary.csv`
- `results/ac_subproblem/case_c_ac_uc_relaxed_voltage_line_eval.csv`
- `results/ac_subproblem/case_c_ac_uc_relaxed_voltage_reactive_eval.csv`

## Next Technical Step

The next necessary implementation step is to attach an AC NLP solver interface. The subproblem needs decision variables for voltage magnitude, voltage angle, reactive generation, wind spillage, load shedding, and feasibility slacks. Once the NLP returns dual multipliers for fixed master quantities, the Benders driver can replace the current screening-count proxy with real feasibility/optimality cuts.
