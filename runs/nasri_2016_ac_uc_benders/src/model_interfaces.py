from __future__ import annotations

from pathlib import Path
from typing import Any

from ac_subproblem import solve_ac_subproblem_placeholder
from benders_driver import run_benders_placeholder
from case_data import CaseData, load_case_data
from dc_uc_baseline import solve_case_a_dc_uc
from uc_results import SolveResult


def solve_deterministic_uc(data: dict[str, Any] | CaseData, solver_config: dict[str, Any]) -> SolveResult:
    case = data if isinstance(data, CaseData) else load_case_data(Path(data["data_dir"]))
    return solve_case_a_dc_uc(case, solver_config, dry_run=True)


def solve_robust_uc_ccg(data: dict[str, Any] | CaseData, solver_config: dict[str, Any]) -> SolveResult:
    case = data if isinstance(data, CaseData) else load_case_data(Path(data["data_dir"]))
    return run_benders_placeholder(case, solver_config, dry_run=True)


def solve_single_ac_subproblem(
    data: dict[str, Any] | CaseData,
    solver_config: dict[str, Any],
    *,
    case_id: str = "case_b_ac_uc_benders",
    scenario_id: int = 1,
    hour: int = 1,
) -> SolveResult:
    case = data if isinstance(data, CaseData) else load_case_data(Path(data["data_dir"]))
    return solve_ac_subproblem_placeholder(case, case_id, scenario_id, hour, solver_config, dry_run=True)
