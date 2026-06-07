# Reproduction Plan: Nasri et al. 2016 AC-UC Benders

## Reproducibility Verdict

**Overall potential: medium-high for approximate reproduction, medium for exact reproduction.**

This paper is more reproducible than the first automatic audit suggested because the case-study section provides several concrete anchors:

- Base system: IEEE one-area 24-node Reliability Test System (RTS), from the IEEE RTS-1996 reference.
- Horizon: 24 hourly periods.
- Scaling: 1 p.u. = 100 MW.
- Wind: two wind farms at nodes 3 and 14, capacities 2.85 p.u. and 2.96 p.u.
- Scenarios: 40 wind scenarios with probabilities in Table IV and profiles in Fig. 3.
- Load: RTS hourly active/reactive loads multiplied by hourly load factors in Table III.
- Voltage cases:
  - Case A: dc-UC, solved directly.
  - Case B: ac-UC with voltage magnitudes in [0.9, 1.1] p.u., solved by Benders.
  - Case C: ac-UC with relaxed voltage magnitudes in [0.5, 1.5] p.u., solved by Benders.
- Solvers: GAMS + CPLEX 12.1 for the MILP master; GAMS + CONOPT for nonlinear subproblems.
- Hardware: Sun Fire X4600M2, 8 Quad-Core CPUs at 2.9 GHz, 256 GB RAM.
- Convergence: tolerance is 0.3% of expected cost; Fig. 5 reports convergence in 25 iterations for Case B.

Exact reproduction remains blocked by table/figure extraction and implementation choices:

- Tables I-IV and VI must be transcribed or recovered from the PDF with visual/table extraction.
- Fig. 3 contains the 40 wind scenario time series; exact reconstruction from a plot is hard unless original data are found.
- Benders cuts rely on sensitivities from nonlinear AC subproblems. CONOPT/GAMS dual handling must be matched or replaced carefully.
- The AC problem is nonconvex. The authors explicitly state that global optimality is not guaranteed and use multi-start decomposition.

## Recommended Reproduction Scope

### Tier 1: Deterministic and Data Reconstruction

Goal: rebuild the case data and verify the deterministic dc-UC baseline.

Tasks:

1. Import or encode IEEE RTS-1996 one-area 24-node network.
2. Apply paper modifications:
   - remove synchronous condenser at node 14;
   - add wind farms at nodes 3 and 14;
   - set wind capacities to 2.85 and 2.96 p.u.;
   - set load-shed value to 10000 USD/p.u.
3. Transcribe:
   - Table I: modified network data;
   - Table II: generating unit data;
   - Table III: hourly load factors;
   - Table IV: scenario probabilities.
4. Reconstruct hourly active/reactive loads from RTS data and Table III.
5. Run Case A dc-UC.

Deliverables:

- `data/rts24_buses.csv`
- `data/rts24_lines_ac.csv`
- `data/generators_ac_uc.csv`
- `data/load_factors.csv`
- `data/wind_scenario_probabilities.csv`
- `reports/data_reconstruction_report.md`

### Tier 2: AC Feasibility Subproblem

Goal: solve the AC recourse problem for fixed first-stage schedules.

Tasks:

1. Implement AC power-flow expressions from Appendix A.
2. Build the scenario/time subproblem (5).
3. Include slack variables for reactive power and voltage magnitude infeasibility.
4. Validate Case B voltage bounds [0.9, 1.1] and Case C bounds [0.5, 1.5].
5. Log NLP status, objective, slack usage, voltages, flows, and dual/sensitivity values.

Deliverables:

- `src/ac_subproblem.py` or equivalent modeling module.
- `results/subproblem_validation/`
- `reports/ac_subproblem_validation.md`

### Tier 3: Benders Master and Iteration

Goal: reproduce the decomposed AC-UC algorithm.

Tasks:

1. Implement master problem (7).
2. Add Benders cuts (7b), using sensitivities from (6a)-(6c).
3. Add lower-bound constraint (7c).
4. Include strengthened bounds (7e)-(7f), updated from previous subproblem solutions.
5. Iterate over all 40 scenarios and 24 hours, or start with reduced scenarios for debugging.
6. Stop when upper/lower bound gap is below 0.3% of expected cost.
7. Re-run Case B with 0.1% tolerance for sensitivity comparison.

Deliverables:

- `results/benders_logs/iteration_log.csv`
- `results/benders_logs/cuts.csv`
- `reports/benders_convergence_report.md`

### Tier 4: Paper Result Alignment

Goal: align headline results.

Targets:

- Table V: mathematical characteristics of each UC problem.
- Table VI: expected cost and computational time for Cases A-C.
- Fig. 4: commitment status for Cases A-C.
- Fig. 5: Benders convergence for Case B, expected 25 iterations at 0.3% tolerance.
- Text claims:
  - no load curtailment or wind spillage in all cases;
  - units 22 and 23 decommitted in Case A but committed in Case B peak hours;
  - unit 9 committed in periods 1 and 2 of Case B;
  - unit 32 scheduled at peak hour 3.2 p.u. in Case A and 2.5 p.u. in Case B.

Deliverables:

- `reports/alignment_report.md`
- `results/alignment/case_a_dc_uc.csv`
- `results/alignment/case_b_ac_uc.csv`
- `results/alignment/case_c_ac_uc.csv`

## Implementation Recommendation

Use a staged solver strategy:

1. Prototype data and Case A in Python/Pyomo with a MILP solver.
2. Prototype AC subproblem using Pyomo + IPOPT, or JuMP + Ipopt, before attempting CONOPT equivalence.
3. Only after AC subproblem duals/sensitivities are stable, implement Benders cuts.
4. Keep a GAMS-compatible notes file because the original paper used GAMS/CPLEX/CONOPT.

## Risk Register

| Risk | Severity | Mitigation |
|---|---:|---|
| Wind scenarios are only plotted in Fig. 3 | High | Try image digitization, author data, or scenario reconstruction with documented approximation. |
| Nonconvex AC subproblem duals may be solver-dependent | High | Compare IPOPT/GAMS/CONOPT if possible; log multi-start sensitivity. |
| OCR loses equation details | Medium | Use PDF screenshots/manual transcription for equations (1)-(9). |
| RTS modifications not fully captured | Medium | Start from RTS-1996 and apply Tables I-II manually. |
| Exact runtime impossible on different hardware | Low | Compare iteration counts and relative time instead of raw CPU seconds. |

## Go/No-Go Criteria

Proceed to implementation if:

- Tables I-IV are transcribed or sourced.
- Fig. 3 wind scenarios are reconstructed to acceptable precision, or replaced by a documented surrogate scenario set.
- AC subproblem solves reliably for fixed schedules.

Do not claim exact reproduction unless:

- original wind scenario values and all case-study tables are recovered;
- Benders iteration count and Case B cost are close to Table VI/Fig. 5;
- solver stack differences are documented.

