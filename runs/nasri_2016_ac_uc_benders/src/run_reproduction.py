from __future__ import annotations

import argparse
import json
from pathlib import Path

from ac_subproblem import solve_ac_subproblem_placeholder, write_ac_subproblem_plan
from benders_driver import run_benders_placeholder
from case_data import load_case_data, write_summary
from dc_uc_baseline import solve_case_a_dc_uc, write_case_a_plan
from screening_dispatch import DispatchScreeningConfig, run_dispatch_screening


def main() -> None:
    parser = argparse.ArgumentParser(description="Nasri 2016 AC-UC reproduction runner")
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--solver-config", default="../configs/solver_config.json")
    parser.add_argument(
        "--experiment",
        choices=["data-summary", "case-a", "screening-dispatch", "ac-subproblem", "benders", "plans"],
        default="data-summary",
    )
    parser.add_argument("--results-dir", default="../results")
    parser.add_argument("--scenario-id", type=int, default=1)
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--hour", type=int, default=1)
    parser.add_argument("--solve", action="store_true", help="Run solver-backed implementation where available")
    parser.add_argument("--ac-nlp-solver", choices=["scipy_slsqp", "cyipopt", "cyipopt_constrained"], help="Override AC NLP backend")
    args = parser.parse_args()

    data_dir = resolve_relative(Path(args.data_dir))
    solver_config_path = resolve_relative(Path(args.solver_config))
    results_dir = resolve_relative(Path(args.results_dir))
    results_dir.mkdir(parents=True, exist_ok=True)

    case = load_case_data(data_dir)
    solver_config = json.loads(solver_config_path.read_text(encoding="utf-8"))
    if args.ac_nlp_solver:
        solver_config["ac_nlp_solver"] = args.ac_nlp_solver

    if args.experiment == "data-summary":
        out = results_dir / "case_data_summary.json"
        write_summary(case, out)
        print(f"Wrote {out}")
        return

    if args.experiment == "plans":
        print(write_case_a_plan(results_dir / "plans"))
        print(write_ac_subproblem_plan(results_dir / "plans"))
        return

    if args.experiment == "case-a":
        result = solve_case_a_dc_uc(case, solver_config, dry_run=not args.solve, out_dir=results_dir / "case_a_dc_uc")
        out = results_dir / "case_a_dc_uc" / "result.json"
        result.write_json(out)
        print(f"Wrote {out}")
        print(result)
        return

    if args.experiment == "screening-dispatch":
        scenario_id = None if args.all_scenarios else args.scenario_id
        result = run_dispatch_screening(
            case,
            solver_config,
            results_dir / "screening_dispatch",
            config=DispatchScreeningConfig(scenario_id=scenario_id),
        )
        out = results_dir / "screening_dispatch" / "result.json"
        result.write_json(out)
        print(f"Wrote {out}")
        print(result)
        return

    if args.experiment == "ac-subproblem":
        result = solve_ac_subproblem_placeholder(
            case,
            "case_b_ac_uc_benders",
            args.scenario_id,
            args.hour,
            solver_config,
            dry_run=not args.solve,
        )
        out = results_dir / "ac_subproblem" / f"scenario_{args.scenario_id}_hour_{args.hour}.json"
        result.write_json(out)
        print(f"Wrote {out}")
        print(result)
        return

    if args.experiment == "benders":
        result = run_benders_placeholder(
            case,
            solver_config,
            dry_run=not args.solve,
            out_dir=results_dir / "benders_logs",
        )
        out = results_dir / "benders_logs" / "result.json"
        result.write_json(out)
        print(f"Wrote {out}")
        print(result)
        return


def resolve_relative(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (Path(__file__).resolve().parent / path).resolve()


if __name__ == "__main__":
    main()
