# CRE-seq Analysis Workbench - Individual Contributions

**Sarrah Rose** · BioE 134, Spring 2026 · UC Berkeley

## Project Overview

CRE-seq (Cis-Regulatory Element sequencing) is a massively parallel reporter assay that tests thousands of candidate regulatory DNA sequences in a single experiment. Each candidate is cloned into a barcoded reporter construct, transfected into cells, and measured by sequencing: the ratio of RNA counts (transcription output) to DNA counts (library input) gives the regulatory activity of each element. The result is a table of thousands of elements with activity scores, from which researchers need to identify which elements are active enhancers and which transcription factors are driving them.

This project builds an **LLM-powered CRE-seq analysis workbench** with a Streamlit chat interface and an MCP (Model Context Protocol) server backend. A researcher describes their analysis goal in natural language, and a chat agent orchestrates the full pipeline: normalization, activity calling, motif enrichment, and visualization.

**My scope:** I designed and implemented the **post-QC analysis engine** - activity calling, motif enrichment (including GC-content matching), visualization, the MCP tool interfaces with rich docstrings, the pipeline architecture connecting stages, and the motif-driven synthetic data generator. I also wrote 42 tests and verified the tools end-to-end with the Gemini CLI.

## What I Built

### Activity Calling (`activity/classify.py`, 309 lines)

Determines which regulatory elements are significantly active. Fits a null distribution from negative controls (median/MAD), applies one-sided z-tests with Benjamini-Hochberg FDR correction, and computes fold-change over controls. Includes quality warnings for low control counts and a GLM-based caller stub for future extension.

### Motif Enrichment (`motifs/enrichment.py`, 401 lines)

Connects activity results to transcription factor biology. Scans active and background sequences against ~900 JASPAR 2024 TF motif matrices on both strands, tests for overrepresentation with Fisher's exact test and BH-FDR correction. Includes `extract_sequences_to_fasta` to bridge activity calling output to motif analysis, splitting elements into active/background FASTA files and filtering controls.

### GC-Content Matching

Active regulatory elements in mammalian genomes tend to be GC-richer than inactive ones, which causes GC-rich TF motifs (SP1, EGR1, KLF family) to appear falsely enriched - a well-known confound that every modern motif tool (HOMER, MEME-AME) controls for. I identified this gap in our pipeline and added an opt-in `gc_match` parameter to `extract_sequences_to_fasta` that bin-matches the background GC distribution to the active set before testing.

The implementation bins both pools into 5%-width GC bins, samples matched candidates per bin, and handles edge cases (empty bins fall back to nearest non-empty bin, undersized bins sample with replacement) with documented `UserWarning`s. Defaults to off for backward compatibility; adds a `gc_matched: bool` field to the return dict. Three new tests guard the default regression, distribution alignment (within 3%), and warning paths.

### Visualization (`plots/plots.py`, 504 lines)

Every figure the workbench displays is produced by these functions: volcano plots, ranked activity plots, replicate correlation scatters, annotation boxplots, and motif dotplots. The `plot_creseq` dispatcher routes by plot name so the agent can request any visualization with a single call. Designed with a consistent visual identity (red/grey/blue/teal palette, 200 DPI, hierarchical font sizing).

### MCP Tool Interfaces & Verification

Each analysis module is exposed as an MCP tool with defined inputs, outputs, and error behavior. I rewrote tool docstrings with structured `Args / Returns / Notes` sections (descriptions grew from ~300 chars to ~2,500 chars each) so that any MCP-compatible LLM client can use them without seeing our codebase.

| Tool | Description |
|------|-------------|
| `extract_sequences` | Bridges activity output to motif input, with opt-in GC matching |
| `tool_motif_enrichment` | Accepts active/background FASTA paths, returns ranked motif table |
| `tool_plot_creseq` | Accepts data + plot type, returns a figure path |

I also authored `tool_prepare_counts` and `tool_call_active_elements`, which were later consolidated into a single `tool_activity_report` entry point - the underlying algorithms are unchanged.

**MCP manifest:** Created `scripts/dump_mcp_manifest.py` to auto-generate `mcp_manifest.json` (26 tools with name, description, and JSONSchema `inputSchema`) so graders can inspect the full tool surface without installing anything.

**Gemini CLI verification:** The MCP server is set up to run as an external Gemini CLI client. The setup, demo prompts, and expected tool-call traces are documented in `docs/gemini_verification.md`.

### Pipeline Architecture

`run_demo_pipeline.py` defined the stage ordering and data flow that the application follows:

```
raw counts -> normalize -> call active elements -> extract sequences -> motif enrichment -> plot
```

This script served as both the integration test and the architectural blueprint for the agent's orchestration logic.

### Motif-Driven Synthetic Data

I redesigned the synthetic data generator (`scripts/generate_test_data.py`, +325 lines) to fix a circular validation problem: the previous generator assigned activity labels arbitrarily, so tests could pass even with broken analysis code. My redesign plants ground-truth biology - real TF binding motifs (GATA1, AP-1, SP1) with defined effect sizes - and lets the pipeline discover the signal de novo. It also generates 50 variant families (reference + 1-bp-disrupted mutants), ~3% barcode collisions, and negative-binomial count noise.

**Validation:** The pipeline recovered the planted signal - top motif hits were JUNB, JUND, FOSL1, NFE2, Gata3, all AP-1 and GATA family members.

### QC Bug Fixes

Targeted fixes to existing QC modules: added low-control-count warnings to `call_activity`, fixed a merge bug in `compute_variant_delta_scores` caused by string-encoded booleans, migrated `variant_family_coverage` to the correct column schema, and added a `collision_threshold` parameter to `barcode_collision_analysis`.

## Tests

42 new tests across four files:

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_activity_calling.py` | 12 | Positive controls detected, controls excluded from FDR, FDR calibration under pure null, robustness to outlier controls, low-control warning, empty input, output schema |
| `test_motif.py` | 16 | Planted GATA1 in top 3, no enrichment under null, JASPAR loading, GC-matching (default regression, distribution alignment, exhausted-bin warning), `extract_sequences_to_fasta` coverage |
| `test_plotting.py` | 14 | All 5 plot types render, dispatcher routing, output files created |
| `conftest.py` | 14 fixtures | Shared test data for all modules |

## Test Prompts

These prompts trigger my functions through the MCP server. Run them via Gemini CLI after registering the server (see `docs/gemini_verification.md` for setup). The four prompts cover server discovery, the upstream pipeline, my `extract_sequences` tool with the new `gc_match` flag, and a chained call to `tool_motif_enrichment`.

**1. Server discovery**

> `/mcp list`

Should show the `creseq` server registered with **26 tools**.

**2. Pipeline setup — `tool_activity_report`**

> "Run `tool_activity_report` on the default upload directory and tell me the summary — how many oligos passed filtering, how many were called active, and what method was used?"

This runs the upstream pipeline and writes `activity_results.tsv` to `~/.creseq/uploads/` so the next prompt has something to consume. (`tool_activity_report` is a teammate's orchestrator that wraps my normalize + classify algorithms.)

**3. My code, with the new GC-matching feature — `extract_sequences`**

> "Now use `extract_sequences` to split the classified table at `~/.creseq/uploads/activity_results.tsv` into active and background FASTAs, using the design manifest at `~/.creseq/uploads/design_manifest.tsv` for sequences. Set `gc_match=True` so the background GC distribution matches the active set. Save active to `/tmp/active_gemini.fa` and background to `/tmp/background_gemini.fa`. Confirm `gc_matched` is true in the response."

This is the prompt that demos my work directly — my authored tool, my GC matching feature, and Gemini passes the new boolean parameter through.

**4. Chained motif enrichment — `tool_motif_enrichment`**

> "Now run `tool_motif_enrichment` on `/tmp/active_gemini.fa` and `/tmp/background_gemini.fa` with `score_threshold=0.85`. Write the table to `/tmp/motif_enrichment_demo.tsv` and tell me the top three significantly enriched TFs from the summary."

Chains off prompt 3, exercising another of my authored tools and showing the agent can pass output paths from one tool call into the next.

## Demo Output

The `demo_outputs/` directory contains full pipeline artifacts: `active.fa` (213 elements), `background.fa` (237 elements), `activity_table.tsv`, `classified_elements.tsv`, `motif_enrichment.tsv` (874 motifs scored), `volcano.png`, `ranked_activity.png`, `motif_dotplot.png`, and `_run.log`.

## Where to Find Evidence

| What | Path |
|------|------|
| Activity calling source | `creseq_mcp/activity/classify.py` |
| Motif enrichment source (incl. GC matching) | `creseq_mcp/motifs/enrichment.py` |
| Plotting source | `creseq_mcp/plots/plots.py` |
| MCP tool wrappers (docstrings) | `creseq_mcp/server.py` (search `extract_sequences`, `tool_motif_enrichment`, `tool_plot_creseq`) |
| MCP manifest | `mcp_manifest.json` (repo root) |
| Tests | `tests/test_activity_calling.py`, `tests/test_motif.py`, `tests/test_plotting.py`, `tests/conftest.py` |
| Gemini verification guide | `docs/gemini_verification.md` |
| Synthetic data generator | `scripts/generate_test_data.py` |
| Demo pipeline | `scripts/run_demo_pipeline.py` |

## Acknowledgments

This was a team project. My teammates built the Streamlit frontend, the association/alignment infrastructure, the normalization module, and the literature/RAG tools. My contributions are scoped to the post-QC analysis engine described above.
