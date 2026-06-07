# Stage 15: Showcase Materials and Toolchain Summary

## 1. Showcase Materials in This Run

### A. Paper evidence and source material

- `artifacts/pdf_previews/page_08.pdf.png`, `page_09.pdf.png`, `page_10.pdf.png`
  - Best for showing the original paper's case-study setup, Tables I-VI, and the three cases.
- `extracted_text/paper_text.json`
  - Machine-readable extraction of the paper.
- `extracted_text/evidence_snippets.json`
  - Short evidence fragments used to anchor audit and reproduction decisions.
- `artifacts/figures_tables_manifest.json`
  - Captured references to Fig. 1-5 and Tables I-VI from the PDF text.
- `artifacts/equations_manifest.json`
  - Equation references and context snippets.

### B. Reproducibility assessment material

- `audits/reproducibility_audit.md`
- `audits/reproducibility_audit.json`
- `artifacts/model_spec.md`
- `artifacts/model_spec.json`
- `artifacts/source_trace.md`
- `artifacts/algorithm_trace.md`
- `reports/reproduction_plan.md`
- `reports/reproduction_checklist.md`
- `reports/data_validation.md`

These are good for explaining:

- whether the paper has enough data clarity,
- which parts are explicit vs inferred,
- why full exact reproduction is still incomplete,
- what the implementation path is.

### C. Case data and benchmark reconstruction

- `data/buses.csv`
- `data/lines.csv`
- `data/generators.csv`
- `data/generator_cost_segments.csv`
- `data/load_profile.csv`
- `data/load_factors.csv`
- `data/wind_farms.csv`
- `data/wind_profile.csv`
- `data/scenario_probabilities.csv`
- `data/uncertainty_bounds.csv`
- `data/wind_scenario_statistics.csv`
- `data/paper_parameters.csv`

These are useful for showing how the paper's benchmark inputs were reconstructed or approximated.

### D. Algorithm and solver artifacts

- `results/case_a_dc_uc/*`
- `results/ac_subproblem/*`
- `results/ac_nlp_subproblem/*`
- `results/benders_cuts/*`
- `results/benders_auto_loop*/*`
- `results/benders_closed_loop/*`

Good for showing:

- deterministic DC baseline,
- AC subproblem behavior,
- dual multipliers and constraint traces,
- Benders cut pools,
- iteration logs and convergence behavior.

### E. Presentation-ready figures

- `results/paper_style_figures/fig3_wind_scenarios.png`
- `results/paper_style_figures/fig4_generation_schedule.png`
- `results/paper_style_figures/fig5_benders_convergence.png`
- `results/paper_style_figures/paper_vs_reproduction_comparison.png`

These are the main slides-ready visual assets.

## 2. Direct Correspondence to the Original Paper

| Original paper item | Current material |
| --- | --- |
| Fig. 3 wind scenarios | `fig3_wind_scenarios.png` |
| Fig. 4 commitment status | `fig4_generation_schedule.png` plus `table_vi_commitment.csv` |
| Fig. 5 Benders convergence | `fig5_benders_convergence.png` |
| Table V mathematical characteristics | `table_v_summary.csv` |
| Table VI numerical results | `table_vi_commitment.csv` and `paper_vs_reproduction_comparison.csv` |
| Case study setup, assumptions, and parameters | `paper_profile.md`, `model_spec.md`, `reproduction_plan.md` |
| Algorithm description | `algorithm_trace.md`, `stage_9` to `stage_13` reports |

## 3. Best Materials for a Demo Deck

Recommended sequence:

1. Original paper evidence page 8 and 9 screenshots.
2. Reproducibility audit summary.
3. Reconstruction plan and data validation.
4. Fig. 3, Fig. 4, Fig. 5.
5. Paper-vs-reproduction comparison table.
6. A solver trace screenshot or CSV snippet from AC subproblem / Benders loop.

## 4. How the Toolchain Works Without an Autonomous Agent

The workflow is script-first, not agent-first.

### Step 1. Local parsing and extraction

The pipeline begins with deterministic scripts:

- `tools/pdf_extract.py` extracts the PDF text into JSON.
- `tools/evidence.py` selects evidence snippets.
- `tools/traces.py` derives source trace and algorithm trace notes.
- `tools/repro_scaffold.py` creates the reproduction workspace and data templates.

### Step 2. LLM only for structured judgment and drafting

`tools/llm_client.py` sends a single prompt plus a JSON schema to the OpenAI Responses API.
It:

- reads `OPENAI_API_KEY` and `OPENAI_BASE_URL`,
- falls back to Codex app config in `~/.codex/config.toml` and `~/.codex/auth.json`,
- requests strict JSON output,
- validates the returned JSON by parsing it locally.

This means the LLM is used as a controlled extraction and drafting service, not as an autonomous code-running agent.

### Step 3. Local code generation and artifact writing

`tools/repro_cli.py` orchestrates the whole run:

- initialize target workspace,
- extract paper text,
- run audit and model spec generation,
- write traces,
- scaffold reproducibility package,
- validate data,
- write Obsidian notes.

All files are written deterministically by Python scripts.

### Step 4. Reproduction implementation in reusable modules

The actual reproduction logic is kept in modular scripts under `runs/nasri_2016_ac_uc_benders/src/`:

- `case_data.py`
- `dc_uc_baseline.py`
- `ac_power_flow.py`
- `ac_subproblem.py`
- `benders_driver.py`
- `run_benders_auto_loop.py`
- `run_ac_nlp_batch.py`
- `generate_benders_cut_constraints.py`
- `build_dual_cut_coefficients.py`
- `render_paper_style_figures.py`

This makes the workflow reusable for other power-system papers with similar structure.

## 5. Reusable Toolchain Components

### Core reusable pieces

- `tools/repro_cli.py`: command entry point for the whole pipeline.
- `tools/llm_client.py`: structured LLM call wrapper with Codex config fallback.
- `tools/pdf_extract.py`: PDF text extraction.
- `tools/repro_scaffold.py`: folder/data/template generator.
- `tools/obsidian.py`: Obsidian bundle writer.
- `tools/traces.py`: trace and manifest generation.
- `tools/audit.py` and `tools/model_spec.py`: structured paper screening and model extraction.

### What can be reused for another paper

- paper PDF extraction,
- evidence snippet selection,
- reproducibility audit schema,
- model-spec schema,
- data-template scaffold,
- Obsidian note export,
- LLM JSON drafting via the same client,
- solver loop structure and result logging.

## 6. Presentation Message

The strongest demo angle is:

> The LLM is not “doing everything.” It is used to structure reasoning, audit the paper, and draft reproducibility artifacts, while the actual engineering remains in local, inspectable scripts and data files.

That is the reusable pattern.
