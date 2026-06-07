# Stage 2 Data and Screening Report

## Missing Data Assessment

Nasri 2016 is a scenario-based stochastic AC unit commitment paper, not an interval-uncertainty robust UC paper. After adjusting the target-specific validation rules, the current required data tables are complete:

- Complete required files: 11
- Empty required files: 0
- Missing required files: 0
- Bad headers: 0
- Optional empty files: `reserves.csv`, `uncertainty_bounds.csv`

The optional files are not treated as blockers:

- `reserves.csv`: aggregate hourly reserve requirements are not separately tabulated in the paper. Unit-level reserve capability is already stored in `generators.csv` as `reserve_up_mw`, `reserve_down_mw`, and `initial_reserve_mw`.
- `uncertainty_bounds.csv`: the paper uses 40 wind production scenarios with probabilities from Table IV. It does not define a box uncertainty set that would naturally populate lower and upper uncertainty bounds.

The original paper-level wind scenario trajectory is not public in the current local evidence. Per the current reproduction decision, `wind_profile.csv` is filled with a documented synthetic substitute marked `SYNTHETIC_CALIBRATED_TO_NASRI_2016`. Results using it should be described as synthetic-data reproduction, not exact paper-result reproduction.

## Completed Data Repair

`buses.csv` now includes RTS-24 load allocation fractions reconstructed from the MATPOWER source:

- Source: `runs/nasri_2016_ac_uc_benders/sources/case24_ieee_rts.m`
- Added columns: `base_pd_mw`, `base_qd_mvar`, `qd_fraction`
- Filled column: `pd_fraction`
- Base total Pd: 2850 MW
- Base total Qd: 580 Mvar
- Sum of `pd_fraction`: 1.0

Script:

```bash
python fill_bus_load_fractions.py
```

## Screening Dispatch

A no-solver screening workflow was added to exercise the reconstructed data across all 40 scenarios and 24 hours:

```bash
python run_reproduction.py --experiment screening-dispatch --all-scenarios
```

Outputs:

- `results/screening_dispatch/result.json`
- `results/screening_dispatch/summary.csv`
- `results/screening_dispatch/dispatch.csv`
- `results/screening_dispatch/line_flows.csv`

Result summary:

- Status: `screening_complete`
- Scenario-hours evaluated: 960
- Max shortage: 0 MW
- Max wind curtailment: 0 MW
- Max absolute line loading: 250%
- Scenario-hours with overloads under merit-order dispatch: 426

## Synthetic Wind Scenario Dataset

Synthetic wind scenarios were generated with a fixed seed and calibrated to the paper's reported expected wind production:

- Script: `runs/nasri_2016_ac_uc_benders/src/generate_surrogate_wind_profiles.py`
- Output: `runs/nasri_2016_ac_uc_benders/data/wind_profile.csv`
- Scenario statistics: `runs/nasri_2016_ac_uc_benders/data/wind_scenario_statistics.csv`
- Source label: `SYNTHETIC_CALIBRATED_TO_NASRI_2016`
- Seed: 2016
- Target capacity factor: 0.296
- Achieved capacity factor: 0.296000
- Expected average wind production: 171.976 MW
- Expected average wind production on 100 MVA base: 1.719760 p.u.
- Expected total wind production over 24 hours: 41.274240 p.u.-h

This is not a UC solution. It deliberately omits commitment binaries, startup/shutdown logic, ramp chronology, reserve scheduling, AC constraints, and Benders cuts. Its purpose is to confirm that the reconstructed data can produce coherent scenario-hour dispatch and network-flow artifacts before attaching MILP/NLP solvers.

## Next Reproduction Tasks

1. Attach a MILP backend for Case A DC-UC.
2. Implement commitment, startup, ramping, reserve deployment, DC nodal balance, and line-limit constraints.
3. Add Case B/C AC subproblem NLP interface and Benders cut logging.
4. Compare Table V, Table VI, Fig. 4, and Fig. 5 as synthetic-data reproduction.
5. Optionally replace synthetic wind with digitized Fig. 3 or author data later for exact paper-result alignment.
