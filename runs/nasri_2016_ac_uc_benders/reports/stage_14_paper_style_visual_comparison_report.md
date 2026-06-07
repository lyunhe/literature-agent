# Stage 14: Paper-Style Visual Comparison

## Purpose

This stage updates the presentation-oriented outputs for the Nasri 2016 AC-UC Benders reproduction. The goal is to make the reproduction artifacts visually comparable with the original paper while clearly separating current computed results from paper-aligned proxy visualization.

## Generated Figures

| Figure | Output | Presentation role |
| --- | --- | --- |
| Fig. 3 style | `results/paper_style_figures/fig3_wind_scenarios.png` | Shows 40 reconstructed 24-hour wind scenarios for two wind farms, with legend. |
| Fig. 4 style | `results/paper_style_figures/fig4_generation_schedule.png` | Shows expected thermal generation, wind usage/curtailment, and committed unit counts, with legend. |
| Fig. 5 style | `results/paper_style_figures/fig5_benders_convergence.png` | Shows a 25-iteration Case-B-like convergence curve: objective bounds plus relative gap. |
| Paper comparison | `results/paper_style_figures/paper_vs_reproduction_comparison.png` | Summarizes original paper values against current reproduction outputs and gaps. |

## Paper Anchors Used

| Item | Original paper value used for comparison |
| --- | --- |
| Case A expected cost | `$638,537.8` |
| Case B expected cost | `$651,909.9` |
| Case C expected cost | `$650,368.9` |
| Case B convergence | 25 Benders iterations at 0.3% tolerance |
| Wind scenario setting | 40 scenarios over 24 hours |

## Current Reproduction Status

Fig. 3 and Fig. 4 are generated from the current reconstruction pipeline: synthetic/reconstructed wind scenarios, paper-style expected schedule data, and the current simplified AC-UC/Benders reproduction artifacts.

Fig. 5 is intentionally marked as `paper_aligned_proxy_for_visualization`. It is not yet the result of a full 40-scenario by 24-hour AC NLP Benders solve at every iteration. The curve is anchored to the paper's Case B expected cost and reported 25-iteration convergence, so it is suitable for explaining the intended algorithmic behavior in a presentation, not for claiming exact numerical reproduction.

The exact full-paper experiment still requires:

- Solving all 40 scenarios and 24 hours in each Benders iteration.
- Running the AC NLP recourse model with full scenario-hour coupling and stable dual extraction.
- Building complete feasibility and optimality cut handling for the nonconvex AC subproblems.
- Reproducing Case B and Case C objective values from computed runs rather than proxy alignment.

## Synchronized Locations

The updated PNG/SVG figures and comparison CSVs were copied to:

- `runs/nasri_2016_ac_uc_benders/obsidian/figures`
- `runs/nasri_2016_ac_uc_benders/obsidian/data`
- `/Users/yunhe/ai for research分享/obsidian-vaults/two-stage-benders-uc-reproduction/figures`
- `/Users/yunhe/ai for research分享/obsidian-vaults/two-stage-benders-uc-reproduction/data`

## Recommended Slide Framing

Use the figure sequence as:

1. Fig. 3: reconstructed uncertainty input, showing that the paper's 40-scenario setting has been rebuilt.
2. Fig. 4: operational schedule result, showing the current pipeline can produce paper-style dispatch/commitment summaries.
3. Fig. 5: algorithmic convergence target, explicitly labeled as a paper-aligned proxy until the full multi-scenario AC NLP loop is completed.
4. Paper comparison table: transparent status check, showing what is already close, what is proxy, and what remains missing.
