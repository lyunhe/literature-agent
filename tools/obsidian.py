from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any


def write_obsidian_bundle(
    *,
    target: dict[str, Any],
    audit_md: str,
    audit_json_path: str | Path,
    text_json_path: str | Path,
    run_dir: str | Path,
) -> Path:
    run_dir = Path(run_dir)
    vault_dir = run_dir / "obsidian"
    papers_dir = vault_dir / "papers"
    data_dir = vault_dir / "data"
    papers_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    home = vault_dir / "Home.md"
    audit_note = papers_dir / "reproducibility_audit.md"
    profile_note = papers_dir / "paper_profile.md"

    home.write_text(
        "\n".join(
            [
                f"# {target['title']}",
                "",
                "- [[papers/paper_profile|Paper profile]]",
                "- [[papers/reproducibility_audit|Reproducibility audit]]",
                "- [[data/reproducibility_audit.json|Audit JSON]]",
                "- [[data/paper_text.json|Extracted PDF text]]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    profile_note.write_text(render_profile(target), encoding="utf-8")
    audit_note.write_text(audit_md, encoding="utf-8")
    shutil.copy2(audit_json_path, data_dir / "reproducibility_audit.json")
    shutil.copy2(text_json_path, data_dir / "paper_text.json")
    artifacts_dir = run_dir / "artifacts"
    for name in [
        "model_spec.md",
        "algorithm_trace.md",
        "source_trace.md",
        "dataset_registry.csv",
        "figures_tables_manifest.json",
        "equations_manifest.json",
        "ac_uc_data_requirements.csv",
    ]:
        src = artifacts_dir / name
        if src.exists():
            shutil.copy2(src, papers_dir / name)
    reports_dir = run_dir / "reports"
    for name in [
        "reproduction_checklist.md",
        "reproduction_plan.md",
        "stage_1_framework_report.md",
        "stage_2_data_and_screening_report.md",
        "stage_3_case_a_solver_report.md",
        "stage_4_benders_ac_screening_report.md",
        "stage_5_ac_nlp_solver_report.md",
        "stage_6_cyipopt_ipopt_integration_report.md",
        "stage_7_benders_cut_and_paper_tests_report.md",
        "stage_8_explicit_ac_nlp_and_dual_cuts_report.md",
        "stage_9_benders_closed_loop_report.md",
        "stage_10_benders_auto_loop_report.md",
        "stage_11_explicit_master_coupling_duals_report.md",
        "stage_12_expected_objective_optimality_cuts_report.md",
        "stage_13_full_recourse_multistart_benders_report.md",
        "bus_load_fraction_report.md",
        "surrogate_wind_profile_report.md",
        "transcribed_tables_i_iv.md",
        "nasri_case_adjustments.md",
        "data_validation.md",
        "data_validation.json",
    ]:
        src = reports_dir / name
        if src.exists():
            shutil.copy2(src, papers_dir / name)
    source_data_dir = run_dir / "data"
    for name in [
        "wind_scenario_statistics.csv",
    ]:
        src = source_data_dir / name
        if src.exists():
            shutil.copy2(src, data_dir / name)
    results_dir = run_dir / "results" / "case_a_dc_uc"
    for name in [
        "experiment_summary.md",
        "hour_summary.csv",
        "scenario_summary.csv",
        "commitment.csv",
        "bus_angles.csv",
    ]:
        src = results_dir / name
        if src.exists():
            shutil.copy2(src, data_dir / f"case_a_{name}")
    paper_results_dir = run_dir / "results" / "paper_style_results"
    for name in [
        "table_v_summary.csv",
        "table_vi_commitment.csv",
        "fig4_generation_schedule.csv",
        "fig5_benders_convergence.csv",
        "paper_style_report.md",
    ]:
        src = paper_results_dir / name
        if src.exists():
            shutil.copy2(src, data_dir / name)
    cuts_dir = run_dir / "results" / "benders_cuts"
    for name in [
        "case_b_cut_pool.csv",
        "case_b_cut_summary.csv",
        "benders_cut_spec.md",
        "case_b_dual_cut_coefficients.csv",
        "case_b_dual_cut_coefficients.md",
        "case_b_benders_cut_constraints.csv",
        "case_b_benders_cut_terms.csv",
        "case_b_benders_cuts.lp.txt",
        "case_b_benders_cut_constraints.md",
    ]:
        src = cuts_dir / name
        if src.exists():
            shutil.copy2(src, data_dir / name)
    closed_loop_dir = run_dir / "results" / "benders_closed_loop"
    for name in [
        "closed_loop_result.json",
        "closed_loop_iteration_log.csv",
        "closed_loop_report.md",
    ]:
        src = closed_loop_dir / name
        if src.exists():
            shutil.copy2(src, data_dir / name)
    auto_loop_dir = run_dir / "results" / "benders_auto_loop_stable"
    for name in [
        "auto_loop_result.json",
        "auto_loop_iteration_log.csv",
        "auto_loop_report.md",
    ]:
        src = auto_loop_dir / name
        if src.exists():
            shutil.copy2(src, data_dir / f"stable_{name}")
    stable_cuts_dir = auto_loop_dir / "cumulative_cuts"
    for name in [
        "benders_cut_constraints.csv",
        "benders_cut_terms.csv",
    ]:
        src = stable_cuts_dir / name
        if src.exists():
            shutil.copy2(src, data_dir / f"stable_{name}")
    coupled_active_dir = run_dir / "results" / "benders_auto_loop_coupled_active"
    for name in [
        "auto_loop_result.json",
        "auto_loop_iteration_log.csv",
        "auto_loop_report.md",
    ]:
        src = coupled_active_dir / name
        if src.exists():
            shutil.copy2(src, data_dir / f"coupled_active_{name}")
    coupled_active_cuts_dir = coupled_active_dir / "cumulative_cuts"
    for name in [
        "benders_cut_constraints.csv",
        "benders_cut_terms.csv",
    ]:
        src = coupled_active_cuts_dir / name
        if src.exists():
            shutil.copy2(src, data_dir / f"coupled_active_{name}")
    expected_optimality_dir = run_dir / "results" / "benders_auto_loop_expected_optimality_v2"
    for name in [
        "auto_loop_result.json",
        "auto_loop_iteration_log.csv",
        "auto_loop_report.md",
    ]:
        src = expected_optimality_dir / name
        if src.exists():
            shutil.copy2(src, data_dir / f"expected_optimality_{name}")
    expected_optimality_cuts_dir = expected_optimality_dir / "cumulative_cuts"
    for name in [
        "benders_cut_constraints.csv",
        "benders_cut_terms.csv",
    ]:
        src = expected_optimality_cuts_dir / name
        if src.exists():
            shutil.copy2(src, data_dir / f"expected_optimality_{name}")
    full_recourse_multistart_dir = run_dir / "results" / "benders_auto_loop_full_recourse_multistart_1iter"
    for name in [
        "auto_loop_result.json",
        "auto_loop_iteration_log.csv",
        "auto_loop_report.md",
    ]:
        src = full_recourse_multistart_dir / name
        if src.exists():
            shutil.copy2(src, data_dir / f"full_recourse_multistart_{name}")
    full_recourse_multistart_cuts_dir = full_recourse_multistart_dir / "cumulative_cuts"
    for name in [
        "benders_cut_constraints.csv",
        "benders_cut_terms.csv",
    ]:
        src = full_recourse_multistart_cuts_dir / name
        if src.exists():
            shutil.copy2(src, data_dir / f"full_recourse_multistart_{name}")
    full_recourse_multistart_iter_dir = full_recourse_multistart_dir / "iteration_01"
    for name in [
        "ac_nlp_batch.csv",
        "dual_cut_coefficients.csv",
        "iteration_report.md",
    ]:
        src = full_recourse_multistart_iter_dir / name
        if src.exists():
            shutil.copy2(src, data_dir / f"full_recourse_multistart_iteration_01_{name}")
    ac_nlp_dir = run_dir / "results" / "ac_nlp_subproblem"
    for name in [
        "case_b_ac_uc_benders_nlp_batch_worst-reactive_3.csv",
        "case_b_ac_uc_benders_nlp_batch_worst-reactive_3.md",
        "case_b_ac_uc_benders_scenario_1_hour_1_nlp_summary.json",
        "case_b_ac_uc_benders_scenario_1_hour_1_nlp_solution.csv",
        "case_b_ac_uc_benders_scipy_slsqp_scenario_1_hour_1_nlp_summary.json",
        "case_b_ac_uc_benders_scipy_slsqp_scenario_1_hour_1_nlp_solution.csv",
        "case_b_ac_uc_benders_cyipopt_scenario_1_hour_1_nlp_summary.json",
        "case_b_ac_uc_benders_cyipopt_scenario_1_hour_1_nlp_solution.csv",
        "case_b_ac_uc_benders_cyipopt_constrained_scenario_1_hour_1_nlp_summary.json",
        "case_b_ac_uc_benders_cyipopt_constrained_scenario_1_hour_1_nlp_solution.csv",
        "case_b_ac_uc_benders_cyipopt_constrained_scenario_1_hour_1_nlp_constraints.csv",
        "case_b_ac_uc_benders_cyipopt_constrained_scenario_1_hour_1_nlp_multipliers.csv",
        "case_b_ac_uc_benders_cyipopt_constrained_nlp_batch_worst-reactive_3.csv",
        "case_b_ac_uc_benders_cyipopt_constrained_nlp_batch_worst-reactive_3.md",
    ]:
        src = ac_nlp_dir / name
        if src.exists():
            shutil.copy2(src, data_dir / name)
    return vault_dir


def render_profile(target: dict[str, Any]) -> str:
    authors = target.get("authors", [])
    if isinstance(authors, list):
        authors_text = ", ".join(str(a) for a in authors)
    else:
        authors_text = str(authors)
    notes = target.get("notes", [])
    lines = [
        "---",
        'type: "literature-note"',
        f'title: "{target.get("title", "")}"',
        f'year: "{target.get("year", "")}"',
        f'venue: "{target.get("venue", "")}"',
        f'doi: "{target.get("doi", "")}"',
        "---",
        "",
        f"# {target.get('title', '')}",
        "",
        "## Metadata",
        "",
        f"- Authors: {authors_text}",
        f"- Year: {target.get('year', '')}",
        f"- Venue: {target.get('venue', '')}",
        f"- DOI: [{target.get('doi', '')}](https://doi.org/{target.get('doi', '')})",
        f"- Role: `{target.get('role', '')}`",
        "",
        "## Notes",
        "",
    ]
    lines += [f"- {note}" for note in notes]
    lines.append("")
    return "\n".join(lines)
