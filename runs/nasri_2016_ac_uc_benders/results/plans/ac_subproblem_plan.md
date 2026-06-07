# AC Subproblem Implementation Plan

1. Implement Appendix A AC active/reactive/apparent flow equations.
2. For a fixed commitment and scheduled dispatch, solve one NLP for each scenario-hour pair.
3. Include slack variables for reactive generation and voltage magnitude infeasibility as in subproblem (5).
4. Export objective, slack values, voltages, line flows, and dual multipliers for fixed first-stage constraints.
5. Use these dual multipliers to build Benders cuts for the MILP master.
6. Validate first on one scenario and one hour before enabling all 40 x 24 subproblems.
