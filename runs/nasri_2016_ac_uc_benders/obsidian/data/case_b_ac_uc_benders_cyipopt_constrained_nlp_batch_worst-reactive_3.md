# AC NLP Batch Report

- Selection: `worst-reactive`
- Solver: `cyipopt_constrained`
- Solved subproblems: 3
- Successful NLP solves: 3
- Max post-NLP P residual: 0.000392 MW
- Max post-NLP Q residual: 0.000000 Mvar
- Max post-NLP line loading: 99.999777%

| Scenario | Hour | Screening Q Violation | NLP P Residual | NLP Q Residual | NLP Line Loading |
|---:|---:|---:|---:|---:|---:|
| 26 | 17 | 516.798 | 0.000000 | 0.000000 | 96.455% |
| 21 | 17 | 514.950 | 0.000000 | 0.000000 | 98.549% |
| 26 | 24 | 514.595 | 0.000392 | 0.000000 | 100.000% |

The constrained Ipopt backend writes explicit constraint values and multipliers for future Benders cut construction.
