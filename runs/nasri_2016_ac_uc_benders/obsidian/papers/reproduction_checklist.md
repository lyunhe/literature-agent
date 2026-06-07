# Reproduction Checklist: Network-Constrained AC Unit Commitment Under Uncertainty: A Benders' Decomposition Approach

## Data

- [ ] Public base case selected and version recorded.
- [ ] Modified IEEE 118-bus changes traced.
- [ ] Generator UC parameters completed.
- [ ] 24-hour load profile completed.
- [ ] Wind farm buses and forecast profiles completed.
- [ ] Line limits and LSF/PTDF construction documented.
- [ ] Unit conventions and base MVA documented.

## Model

- [ ] Deterministic UC implemented.
- [ ] Deterministic UC unit tests pass on toy data.
- [ ] DC/LSF transmission constraints validated.
- [ ] Robust master problem implemented.
- [ ] Feasibility subproblem implemented.
- [ ] Optimality subproblem implemented.
- [ ] C&CG bound trace logged.
- [ ] LSF cutting-plane active-line selection implemented.

## Results

- [ ] Objective values exported.
- [ ] Commitment matrix exported.
- [ ] Worst-case scenarios exported.
- [ ] Runtime and iteration counts exported.
- [ ] Alignment report generated.
