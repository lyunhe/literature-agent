# AC NLP Batch Report

- Selection: `worst-reactive`
- Solved subproblems: 3
- Successful NLP solves: 3
- Max post-NLP P residual: 0.098316 MW
- Max post-NLP Q residual: 0.090848 Mvar
- Max post-NLP line loading: 99.984496%

| Scenario | Hour | Screening Q Violation | NLP P Residual | NLP Q Residual | NLP Line Loading |
|---:|---:|---:|---:|---:|---:|
| 26 | 17 | 516.798 | 0.029686 | 0.057317 | 99.966% |
| 21 | 17 | 514.950 | 0.015989 | 0.090848 | 99.984% |
| 26 | 24 | 514.595 | 0.098316 | 0.028322 | 99.978% |

This batch uses SciPy SLSQP as an open-source NLP prototype. It does not provide the dual multipliers required for exact Benders cuts.
