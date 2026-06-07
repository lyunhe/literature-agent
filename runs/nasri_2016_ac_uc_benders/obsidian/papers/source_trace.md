# Source Trace: Network-Constrained AC Unit Commitment Under Uncertainty: A Benders' Decomposition Approach

## Data Dependencies

| Item | Current Status | Reproduction Action |
|---|---|---|
| Modified IEEE 118-bus network | Partially specified in paper | Start from MATPOWER/PGLib IEEE 118 and trace generator/line modifications. |
| 54-generator UC data | Paper reports count, not full machine-readable table | Trace cited UC data source or reconstruct from standard IEEE 118 UC datasets. |
| Three wind farms | Three identical wind stations are specified, but bus locations need tracing | Search cited references or define documented reproduction assumptions. |
| 24-hour load profile | Peak value is specified; full profile needs transcription/source tracing | Extract table/figure or reuse cited benchmark profile with clear note. |
| Uncertainty set | +/-20% wind and +/-3% load are specified | Directly implement after profile reconstruction. |
| Solver stack | AMPL/CPLEX and C++/Coin-OR Bcp are specified | Reproduce first in Pyomo/JuMP + Gurobi/CPLEX/HiGHS, record solver differences. |

## Source-Tracing Questions

- Which IEEE 118-bus data version is modified?
- Which buses host the three wind stations?
- What is the exact 24-hour load shape?
- Are generator startup, shutdown, ramp, reserve and piecewise cost parameters taken from a cited UC benchmark?
- Are line limits original, modified, or scaled?
