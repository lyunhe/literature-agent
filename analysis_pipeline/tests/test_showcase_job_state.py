from __future__ import annotations

import json

import literature_showcase.app as showcase_app


def test_frontend_progress_resume_is_scoped_to_active_run() -> None:
    script_path = showcase_app.APP_DIR / "static" / "showcase.js"
    script = script_path.read_text(encoding="utf-8")
    active_guard_index = script.index("if (activeRunId)")
    running_fallback_index = script.index("const runningMatches")

    assert active_guard_index < running_fallback_index
    assert "activeJob.progress?.run_id !== activeRunId" in script
    assert "fetchTrackedJob({id: `run:${activeRunId}`, run_id: activeRunId})" in script


def test_failed_zero_pdf_job_marks_discovery_phase_failed(tmp_path) -> None:
    run_dir = tmp_path / "20260529_1009_demo"
    discovery_dir = run_dir / "01_discovery"
    discovery_dir.mkdir(parents=True)
    (discovery_dir / "raw_candidates.json").write_text("[]", encoding="utf-8")
    (discovery_dir / "downloadable_candidates.json").write_text("[]", encoding="utf-8")
    (discovery_dir / "selected_pdfs.json").write_text("[]", encoding="utf-8")
    (run_dir / "unified_run_report.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "max_papers": 100,
                "steps": [
                    {
                        "index": 1,
                        "name": "0. 发现阶段：检索/方向预筛/PDF 准备",
                        "status": "completed",
                        "elapsed_seconds": 51.337,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    payload = showcase_app.public_job_payload(
        {"id": "web_123456789_demo", "status": "failed", "output_dir": str(run_dir), "max_papers": 100}
    )

    assert payload["progress"]["status"] == "failed"
    assert payload["progress"]["steps"][0]["status"] == "failed"
    assert "在线检索没有返回题录候选" in payload["progress"]["notes"][0]


def test_persisted_job_uses_terminal_report_status(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(showcase_app, "OUTPUT_ROOT", tmp_path)
    run_dir = tmp_path / "20260528_0000_demo"
    run_dir.mkdir()
    (run_dir / "unified_run_report.json").write_text(
        json.dumps({"status": "completed", "steps": []}, ensure_ascii=False),
        encoding="utf-8",
    )
    job = {
        "id": "web_123456789_demo",
        "status": "running",
        "output_dir": str(run_dir),
        "max_papers": 1,
    }
    showcase_app.persist_job(job)

    loaded = showcase_app.load_persisted_job("web_123456789_demo")
    payload = showcase_app.public_job_payload(loaded or {})

    assert loaded is not None
    assert payload["status"] == "completed"
    assert payload["progress"]["status"] == "completed"
    assert payload["progress"]["run_id"] == run_dir.name
