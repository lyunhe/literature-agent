# Stage 3 Case A Solver Experiment Report

## Experiment Scope

This stage solves a synthetic-data DC unit commitment MILP for Nasri 2016 Case A.

- Solver: HiGHS via `highspy`
- Model type: extensive-form DC-UC MILP
- Wind data: `SYNTHETIC_CALIBRATED_TO_NASRI_2016`
- Scenarios: 40
- Hours: 24
- Generators: 32
- Buses: 24
- Lines: 38
- Variables: 80,256
- Constraints: 159,168

Implemented constraints:

- First-stage unit commitment binaries
- Startup binaries and startup cost
- Scenario-dependent active power dispatch
- Generator minimum/maximum output tied to commitment
- Wind usage and curtailment variables
- Nodal DC power balance
- Branch thermal limits
- Load shedding with high penalty

Not yet implemented:

- Minimum up/down time
- Ramping chronology
- Reserve constraints
- AC feasibility subproblems
- Benders optimality/feasibility cuts

## Solver Result

- Status: `kOptimal`
- Objective: 639,922.3560
- Total runtime: 29.565 s
- Solver runtime: 27.600 s
- Committed unit-hours: 578
- Startup count: 3
- Expected thermal energy: 50,621.076 MWh
- Expected wind used: 4,127.424 MWh
- Expected wind curtailed: 0.000 MWh
- Expected load shedding: 0.000000 MWh
- Maximum absolute line loading: 100.000%

## Hourly Summary

| Hour | Committed Units | Startups | Expected Thermal MW | Expected Wind MW | Max Line Loading |
|---:|---:|---:|---:|---:|---:|
| 1 | 23 | 0 | 1983.679 | 153.821 | 57.964% |
| 2 | 23 | 0 | 1836.763 | 158.237 | 53.440% |
| 3 | 23 | 0 | 1693.749 | 158.751 | 51.673% |
| 4 | 23 | 0 | 1548.839 | 161.161 | 52.104% |
| 5 | 23 | 0 | 1600.996 | 166.004 | 51.567% |
| 6 | 23 | 0 | 1620.013 | 175.487 | 52.020% |
| 7 | 23 | 0 | 1664.173 | 188.327 | 52.095% |
| 8 | 23 | 0 | 1728.874 | 209.126 | 53.076% |
| 9 | 23 | 0 | 1775.697 | 219.303 | 54.249% |
| 10 | 23 | 0 | 1819.504 | 232.496 | 56.137% |
| 11 | 23 | 0 | 1897.258 | 240.242 | 59.062% |
| 12 | 23 | 0 | 1990.522 | 232.478 | 61.779% |
| 13 | 24 | 1 | 2050.922 | 229.078 | 63.722% |
| 14 | 24 | 0 | 2212.382 | 210.118 | 53.571% |
| 15 | 24 | 0 | 2238.941 | 183.559 | 53.571% |
| 16 | 25 | 1 | 2403.605 | 161.395 | 100.000% |
| 17 | 25 | 0 | 2479.937 | 142.063 | 100.000% |
| 18 | 26 | 1 | 2582.733 | 124.767 | 100.000% |
| 19 | 26 | 0 | 2674.323 | 118.677 | 100.000% |
| 20 | 26 | 0 | 2731.723 | 118.277 | 100.000% |
| 21 | 26 | 0 | 2642.256 | 122.244 | 100.000% |
| 22 | 26 | 0 | 2520.715 | 129.785 | 100.000% |
| 23 | 25 | 0 | 2450.528 | 142.972 | 100.000% |
| 24 | 25 | 0 | 2472.944 | 149.056 | 100.000% |

## Network Bottleneck

The binding network limit is line 11 from bus 7 to bus 8:

- Limit: 175 MW
- Maximum solved flow: 175 MW
- Loading: 100%
- The line binds repeatedly from peak-load hours onward.

Compared with the earlier no-solver screening pass, the MILP dispatch removes overloads by redispatching generation while respecting the line limits.

## Output Files

- `results/case_a_dc_uc/result.json`
- `results/case_a_dc_uc/experiment_summary.md`
- `results/case_a_dc_uc/commitment.csv`
- `results/case_a_dc_uc/dispatch.csv`
- `results/case_a_dc_uc/wind_usage.csv`
- `results/case_a_dc_uc/load_shedding.csv`
- `results/case_a_dc_uc/line_flows.csv`
- `results/case_a_dc_uc/scenario_summary.csv`
- `results/case_a_dc_uc/hour_summary.csv`

## Interpretation

This stage demonstrates that the reconstructed RTS-24 data, synthetic wind scenarios, and DC network model can support a full MILP solve. It is a meaningful step beyond dry-run scaffolding, but it should be reported as a synthetic-data DC-UC experiment rather than a complete reproduction of the paper's AC-Benders Case B/C results.
