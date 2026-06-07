# Stage 6 cyipopt/Ipopt Integration Report

## Environment Setup

Ipopt was installed locally without changing the system environment:

- Micromamba binary: `/Users/yunhe/.local/bin/micromamba`
- Ipopt environment: `/Users/yunhe/.cache/ipopt-env`
- Ipopt version: 3.14.19
- cyipopt version: 1.7.0

The successful cyipopt build used:

```bash
PATH=/Users/yunhe/.cache/ipopt-env/bin:$PATH \
PKG_CONFIG_PATH=/Users/yunhe/.cache/ipopt-env/lib/pkgconfig \
LDFLAGS='-Wl,-rpath,/Users/yunhe/.cache/ipopt-env/lib' \
/Users/yunhe/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m pip install --no-build-isolation cyipopt
```

Why the extra flags were needed:

- `cyipopt` needs native Ipopt headers/libraries.
- `pkg-config` was not available on the base system.
- Setting global `DYLD_LIBRARY_PATH` broke NumPy because conda-forge runtime libraries shadowed the Python runtime libraries.
- Using linker rpath avoids the NumPy import problem while letting `cyipopt` locate `libipopt`.

## Smoke Test

cyipopt successfully imported and solved a small bound-constrained NLP:

- `cyipopt`: 1.7.0
- Result: success
- Solution: `[1, -2]`
- Objective: approximately `3e-23`

This confirms that the Python package and Ipopt dynamic library are operational.

## AC NLP Backend Integration

`run_reproduction.py` now supports backend selection:

```bash
python run_reproduction.py --experiment ac-subproblem --scenario-id 1 --hour 1 --solve --ac-nlp-solver scipy_slsqp
python run_reproduction.py --experiment ac-subproblem --scenario-id 1 --hour 1 --solve --ac-nlp-solver cyipopt
```

The default remains `scipy_slsqp` because it is currently more numerically reliable for the black-box residual-minimization prototype.

## Backend Comparison

Scenario 1, hour 1, Case B:

| Backend | Status | Objective | Iterations | Max P Residual | Max Q Residual | Max Line Loading |
|---|---|---:|---:|---:|---:|---:|
| scipy_slsqp | ac_nlp_solved | 6.287638e-07 | 83 | 0.021074 MW | 0.032984 Mvar | 48.854253% |
| cyipopt | ac_nlp_failed | 5.145145e-02 | 400 | 8.130012 MW | 5.854399 Mvar | 49.477007% |

cyipopt is installed and callable, but the current black-box formulation does not yet give it the structured derivatives and constraints it needs to perform well. A simple SLSQP warm start and finite-difference gradient were tested, but Ipopt still hit the iteration cap on this prototype objective.

## Interpretation

The integration attempt succeeded at the system/package level and partially at the application level:

- cyipopt/Ipopt can be imported and run from the Codex Python.
- The reproduction runner can select cyipopt as an AC NLP backend.
- For the current residual-penalty formulation, SciPy SLSQP remains the better default.

The next real Ipopt step is not more installation work; it is model restructuring:

- Express AC power balance as equality constraints, not only residual penalties.
- Express branch MVA limits as inequality constraints.
- Provide analytic objective gradient and constraint Jacobian.
- Use cyipopt's lower-level `Problem` interface so Ipopt can return multipliers suitable for Benders cuts.

That restructuring is required before cyipopt can support the paper-style Benders feasibility/optimality cuts.
