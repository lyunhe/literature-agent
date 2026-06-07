# Case A DC-UC Implementation Plan

1. Build unit commitment binary variables for each generator and hour.
2. Add active generation, scheduled wind, voltage angle, startup cost variables.
3. Add nodal active-power balance using DC line flow.
4. Add generator min/max output, ramping, line capacity, reference angle, and startup constraints.
5. Solve direct MILP and export objective, commitment matrix, dispatch, and line flows.
6. Align with Table VI and Fig. 4 in the paper.
