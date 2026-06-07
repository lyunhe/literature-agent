from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .audit import audit_to_markdown, run_audit
from .config_io import ensure_dirs, load_simple_yaml
from .evidence import load_text_json, select_evidence
from .model_spec import model_spec_to_markdown, run_model_spec
from .matpower_import import import_matpower_case
from .obsidian import write_obsidian_bundle
from .pdf_extract import extract_pdf
from .repro_scaffold import (
    extract_reproduction_manifest,
    scaffold_reproduction_package,
    validate_data_templates,
)
from .traces import write_algorithm_trace, write_source_trace


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(prog="paper-repro", description="Paper reproduction tooling")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in [
        "init-target",
        "extract-pdf",
        "audit",
        "model-spec",
        "traces",
        "scaffold-repro",
        "validate-data",
        "extract-manifest",
        "prepare-repro",
        "import-matpower",
        "write-obsidian",
        "run-all",
    ]:
        cmd = sub.add_parser(name)
        cmd.add_argument("--target", required=True, help="Path to target YAML config")
        if name == "import-matpower":
            cmd.add_argument("--case", required=True, help="Path to MATPOWER .m case file")
        if name in {"audit", "model-spec", "run-all"}:
            cmd.add_argument("--offline", action="store_true", help="Do not call LLM API")

    args = parser.parse_args()
    target = load_target(args.target)

    if args.command == "init-target":
        init_target(target)
    elif args.command == "extract-pdf":
        init_target(target)
        cmd_extract_pdf(target)
    elif args.command == "audit":
        init_target(target)
        cmd_audit(target, offline=args.offline)
    elif args.command == "model-spec":
        init_target(target)
        cmd_model_spec(target, offline=args.offline)
    elif args.command == "traces":
        init_target(target)
        cmd_traces(target)
    elif args.command == "scaffold-repro":
        init_target(target)
        cmd_scaffold_repro(target)
    elif args.command == "validate-data":
        init_target(target)
        cmd_validate_data(target)
    elif args.command == "extract-manifest":
        init_target(target)
        cmd_extract_manifest(target)
    elif args.command == "prepare-repro":
        init_target(target)
        ensure_text_exists(target)
        cmd_traces(target)
        cmd_scaffold_repro(target)
        cmd_extract_manifest(target)
        cmd_validate_data(target)
    elif args.command == "import-matpower":
        init_target(target)
        cmd_import_matpower(target, args.case)
    elif args.command == "write-obsidian":
        init_target(target)
        cmd_write_obsidian(target)
    elif args.command == "run-all":
        init_target(target)
        cmd_extract_pdf(target)
        cmd_audit(target, offline=args.offline)
        cmd_model_spec(target, offline=args.offline)
        cmd_traces(target)
        cmd_scaffold_repro(target)
        cmd_extract_manifest(target)
        cmd_validate_data(target)
        cmd_write_obsidian(target)


def load_target(path: str | Path) -> dict:
    target = load_simple_yaml(path)
    target["_config_path"] = str(Path(path).resolve())
    target["run_dir"] = str(resolve_path(target["run_dir"]))
    return target


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def init_target(target: dict) -> None:
    run_dir = Path(target["run_dir"])
    ensure_dirs(
        run_dir,
        run_dir / "pdfs",
        run_dir / "extracted_text",
        run_dir / "audits",
        run_dir / "artifacts",
        run_dir / "logs",
    )
    target_copy = run_dir / "target.yaml"
    if not target_copy.exists():
        shutil.copy2(target["_config_path"], target_copy)
    source_pdf = Path(str(target.get("source_pdf", "")))
    if source_pdf.exists():
        dest = run_dir / "pdfs" / source_pdf.name
        if not dest.exists():
            shutil.copy2(source_pdf, dest)
    print(f"Initialized {run_dir}")


def cmd_extract_pdf(target: dict) -> Path:
    source_pdf = find_run_pdf(target)
    out = Path(target["run_dir"]) / "extracted_text" / "paper_text.json"
    doc = extract_pdf(source_pdf, out)
    evidence = select_evidence(doc)
    evidence_out = Path(target["run_dir"]) / "extracted_text" / "evidence_snippets.json"
    evidence_out.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Extracted {doc['page_count']} pages to {out}")
    print(f"Wrote {len(evidence)} evidence snippets to {evidence_out}")
    return out


def cmd_audit(target: dict, offline: bool) -> Path:
    text_path = ensure_text_exists(target)
    text_json = load_text_json(text_path)
    audit = run_audit(
        target=target,
        text_json=text_json,
        schema_path=ROOT / "config" / "schemas" / "reproducibility_audit.schema.json",
        prompt_path=ROOT / "config" / "prompts" / "reproducibility_audit.md",
        offline=offline,
    )
    out_json = Path(target["run_dir"]) / "audits" / "reproducibility_audit.json"
    out_md = Path(target["run_dir"]) / "audits" / "reproducibility_audit.md"
    out_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(audit_to_markdown(audit), encoding="utf-8")
    print(f"Wrote audit to {out_md}")
    return out_md


def cmd_model_spec(target: dict, offline: bool) -> Path:
    text_path = ensure_text_exists(target)
    text_json = load_text_json(text_path)
    spec = run_model_spec(
        target=target,
        text_json=text_json,
        schema_path=ROOT / "config" / "schemas" / "model_spec.schema.json",
        prompt_path=ROOT / "config" / "prompts" / "model_spec.md",
        offline=offline,
    )
    out_json = Path(target["run_dir"]) / "artifacts" / "model_spec.json"
    out_md = Path(target["run_dir"]) / "artifacts" / "model_spec.md"
    out_json.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(model_spec_to_markdown(spec), encoding="utf-8")
    print(f"Wrote model spec to {out_md}")
    return out_md


def cmd_write_obsidian(target: dict) -> Path:
    run_dir = Path(target["run_dir"])
    audit_json = run_dir / "audits" / "reproducibility_audit.json"
    audit_md = run_dir / "audits" / "reproducibility_audit.md"
    text_json = run_dir / "extracted_text" / "paper_text.json"
    if not audit_json.exists() or not audit_md.exists():
        raise FileNotFoundError("Run `audit` before `write-obsidian`.")
    if not text_json.exists():
        raise FileNotFoundError("Run `extract-pdf` before `write-obsidian`.")
    vault = write_obsidian_bundle(
        target=target,
        audit_md=audit_md.read_text(encoding="utf-8"),
        audit_json_path=audit_json,
        text_json_path=text_json,
        run_dir=run_dir,
    )
    print(f"Wrote Obsidian bundle to {vault}")
    return vault


def cmd_traces(target: dict) -> None:
    artifacts = Path(target["run_dir"]) / "artifacts"
    algorithm_path = write_algorithm_trace(target, artifacts / "algorithm_trace.md")
    source_path, registry_path = write_source_trace(target, artifacts)
    print(f"Wrote algorithm trace to {algorithm_path}")
    print(f"Wrote source trace to {source_path}")
    print(f"Wrote dataset registry to {registry_path}")


def cmd_scaffold_repro(target: dict) -> None:
    created = scaffold_reproduction_package(target)
    print(f"Scaffolded reproduction package with {len(created)} new files")
    for path in created:
        print(f"  {path}")


def cmd_validate_data(target: dict) -> None:
    summary = validate_data_templates(target)
    report = Path(target["run_dir"]) / "reports" / "data_validation.md"
    print(
        "Data validation: "
        f"{summary['complete_files']} complete, "
        f"{summary['empty_files']} empty, "
        f"{summary['missing_files']} missing, "
        f"{summary['bad_header_files']} bad headers"
    )
    print(f"Wrote {report}")


def cmd_extract_manifest(target: dict) -> None:
    text_path = ensure_text_exists(target)
    text_json = load_text_json(text_path)
    out_dir = Path(target["run_dir"]) / "artifacts"
    fig_path, eq_path = extract_reproduction_manifest(text_json, out_dir)
    print(f"Wrote figures/tables manifest to {fig_path}")
    print(f"Wrote equations manifest to {eq_path}")


def cmd_import_matpower(target: dict, case_path: str) -> None:
    manifest = import_matpower_case(case_path, target)
    print(
        f"Imported MATPOWER case with matrices {manifest['matrices']} "
        f"into {Path(target['run_dir']) / 'data'}"
    )


def ensure_text_exists(target: dict) -> Path:
    text_path = Path(target["run_dir"]) / "extracted_text" / "paper_text.json"
    if not text_path.exists():
        cmd_extract_pdf(target)
    return text_path


def find_run_pdf(target: dict) -> Path:
    run_pdf_dir = Path(target["run_dir"]) / "pdfs"
    pdfs = sorted(run_pdf_dir.glob("*.pdf"))
    if pdfs:
        return pdfs[0]
    source_pdf = Path(str(target.get("source_pdf", "")))
    if source_pdf.exists():
        return source_pdf
    raise FileNotFoundError(f"No PDF found for target {target.get('id')}")


if __name__ == "__main__":
    main()
