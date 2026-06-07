# Stage 5 AC NLP Solver Integration Report

## Solver Choice

An open-source AC NLP prototype has been connected using SciPy SLSQP:

- Package: `scipy`
- Solver call: `scipy.optimize.minimize(method='SLSQP')`
- Role: AC feasibility subproblem prototype
- Current command: `python run_reproduction.py --experiment ac-subproblem --scenario-id 1 --hour 1 --solve`

Ipopt/cyipopt remains the preferred next backend for exact Benders cuts because it can expose the nonlinear-programming structure more naturally and is closer to the paper's CONOPT-style NLP subproblems. The SciPy backend was selected first because it installs cleanly in the current environment.

## Implemented NLP Prototype

For a fixed DC-UC master solution, the AC NLP optimizes:

- Bus voltage magnitudes
- Non-slack bus voltage angles
- Online generator reactive outputs

It fixes:

- Unit commitment
- Active generation dispatch
- Wind usage
- Load profile

Objective:

- Minimize active-power balance residuals
- Minimize reactive-power balance residuals
- Penalize apparent-flow overloads

Current simplifications:

- Lossless AC branch equations
- No branch resistance, charging, shunts, or transformer taps
- No dual multipliers for Benders cuts from SciPy SLSQP
- No active redispatch, wind spillage, or load shedding decisions inside the NLP yet

## Single-Subproblem Test

Scenario 1, hour 1, Case B:

- Status: `ac_nlp_solved`
- NLP success: true
- Iterations: 83
- Objective: 6.287638e-07
- Max active-power residual: 0.021074 MW
- Max reactive-power residual: 0.032984 Mvar
- Max AC line loading: 48.854253%

This confirms that the fixed DC master point can be repaired into a near-feasible AC operating point for at least one scenario-hour when voltage and reactive variables are optimized.

## Worst-Reactive Batch

Command:

```bash
python run_ac_nlp_batch.py --limit 3 --selection worst-reactive
```

Results:

| Scenario | Hour | Screening Q Violation | NLP P Residual | NLP Q Residual | NLP Line Loading |
|---:|---:|---:|---:|---:|---:|
| 26 | 17 | 516.798 | 0.029686 | 0.057317 | 99.966% |
| 21 | 17 | 514.950 | 0.015989 | 0.090848 | 99.984% |
| 26 | 24 | 514.595 | 0.098316 | 0.028322 | 99.978% |

All three selected worst-reactive screening cases solved successfully with residuals below 0.1 MW/Mvar. This shows that the previous screening violations were largely caused by fixed-voltage/reactive assumptions, not necessarily by true AC infeasibility.

## Output Files

- `results/ac_nlp_subproblem/case_b_ac_uc_benders_scenario_1_hour_1_nlp_summary.json`
- `results/ac_nlp_subproblem/case_b_ac_uc_benders_scenario_1_hour_1_nlp_solution.csv`
- `results/ac_nlp_subproblem/case_b_ac_uc_benders_nlp_batch_worst-reactive_3.csv`
- `results/ac_nlp_subproblem/case_b_ac_uc_benders_nlp_batch_worst-reactive_3.md`

## Next Benders Step

The next practical step is to add a backend abstraction:

- `scipy_slsqp`: current feasibility prototype, no duals.
- `cyipopt` or direct Ipopt: target backend for dual multipliers and Benders cuts.

After that, the Benders driver should call the AC NLP subproblem for violated scenario-hours, collect dual/sensitivity information where available, and generate real feasibility or optimality cuts instead of the current screening-count proxy.
