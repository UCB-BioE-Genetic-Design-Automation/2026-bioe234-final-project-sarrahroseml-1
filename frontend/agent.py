"""Claude agent with real CRE-seq library-QC tool calling via the Anthropic SDK."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic

from creseq_mcp.server import UPLOAD_DIR, _summary
from creseq_mcp.qc.activity import compute_variant_delta_scores
from creseq_mcp.qc.motifs import annotate_top_motifs
from creseq_mcp.qc.library import (
    barcode_collision_analysis,
    barcode_complexity,
    barcode_uniformity,
    gc_content_bias,
    library_summary_report,
    oligo_length_qc,
    oligo_recovery,
    plasmid_depth_summary,
    synthesis_error_profile,
    variant_family_coverage,
)
from creseq_mcp.stats.library import (
    normalize_activity,
    call_active_elements,
    rank_cre_candidates,
    motif_enrichment_summary,
    prepare_rag_context,
    count_barcodes_from_fastq,
    aggregate_fastq_counts_to_elements,
    search_pubmed,
    search_jaspar_motif,
    search_encode_tf,
    literature_search_for_motifs,
    interpret_literature_evidence,
)

_SYSTEM_PROMPT = (
    "You are a CRE-seq library QC assistant backed by real analysis tools. "
    "File path arguments are optional — omit them and the tools will automatically use "
    "whatever data the user has uploaded via the UI. "
    "Summarise results clearly: state PASS/FAIL and highlight any warnings. "
    "These tools cover library-side QC only (before RNA analysis). "
    "For downstream steps (normalisation, activity calling, motif enrichment) tell the user "
    "those tools are not yet implemented."
)

def _p(args: dict, key: str, filename: str) -> str:
    return args.get(key) or str(UPLOAD_DIR / filename)


def _activity_report(args: dict) -> tuple:
    from creseq_mcp.qc.activity import activity_report
    _, summary = activity_report(
        _p(args, "dna_counts_path", "plasmid_counts.tsv"),
        _p(args, "rna_counts_path", "rna_counts.tsv"),
        _p(args, "design_manifest_path", "design_manifest.tsv") if args.get("design_manifest_path") else None,
        upload_dir=UPLOAD_DIR,
    )
    return ({}, summary)

_TOOLS: list[dict] = [
    {
        "name": "tool_barcode_complexity",
        "description": (
            "Per-oligo barcode count statistics. Returns distinct barcodes per oligo, "
            "fraction error-free, and median read depth. PASS when median barcodes/oligo >= 10."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "mapping_table_path": {"type": "string", "description": "Path to the mapping table TSV file"},
                "min_reads_per_barcode": {"type": "integer", "description": "Minimum reads per barcode (default 1)"},
            },
            "required": [],
        },
    },
    {
        "name": "tool_oligo_recovery",
        "description": (
            "Recovery rate of designed oligos by category. "
            "PASS when test_element recovery@10 >= 80% AND positive_control recovery@10 >= 95%."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "mapping_table_path": {"type": "string", "description": "Path to the mapping table TSV file"},
                "design_manifest_path": {"type": "string", "description": "Path to the design manifest TSV file"},
            },
            "required": [],
        },
    },
    {
        "name": "tool_synthesis_error_profile",
        "description": (
            "Per-oligo synthesis error characterisation from CIGAR/MD tags. "
            "Reports mismatches, indels, soft-clip rates. PASS when median perfect_fraction >= 0.50."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "mapping_table_path": {"type": "string", "description": "Path to the mapping table TSV file"},
                "design_manifest_path": {"type": "string", "description": "Path to the design manifest TSV file (optional)"},
            },
            "required": [],
        },
    },
    {
        "name": "tool_barcode_collision_analysis",
        "description": "Barcodes that map to more than one oligo. PASS when collision rate < 3%.",
        "input_schema": {
            "type": "object",
            "properties": {
                "mapping_table_path": {"type": "string", "description": "Path to the mapping table TSV file"},
                "min_read_support": {"type": "integer", "description": "Minimum read support to count a mapping (default 2)"},
            },
            "required": [],
        },
    },
    {
        "name": "tool_barcode_uniformity",
        "description": (
            "Per-oligo barcode abundance evenness in the plasmid pool (Gini coefficient). "
            "PASS when median Gini < 0.30."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "plasmid_count_path": {"type": "string", "description": "Path to the plasmid count TSV file"},
                "min_barcodes_per_oligo": {"type": "integer", "description": "Minimum barcodes per oligo (default 5)"},
            },
            "required": [],
        },
    },
    {
        "name": "tool_gc_content_bias",
        "description": (
            "Synthesis recovery stratified by oligo GC content. "
            "PASS when no GC bins show dropout below 50% of the median bin."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "mapping_table_path": {"type": "string", "description": "Path to the mapping table TSV file"},
                "design_manifest_path": {"type": "string", "description": "Path to the design manifest TSV file"},
                "gc_bins": {"type": "integer", "description": "Number of GC bins (default 10)"},
            },
            "required": [],
        },
    },
    {
        "name": "tool_oligo_length_qc",
        "description": (
            "Synthesis-truncation check comparing observed alignment length to designed length. "
            "PASS when median fraction_full_length >= 0.80."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "mapping_table_path": {"type": "string", "description": "Path to the mapping table TSV file"},
                "design_manifest_path": {"type": "string", "description": "Path to the design manifest TSV file"},
            },
            "required": [],
        },
    },
    {
        "name": "tool_plasmid_depth_summary",
        "description": (
            "Barcode-level read-count statistics in the plasmid DNA library. "
            "PASS when median dna_count >= 10 AND fewer than 10% of barcodes have zero counts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "plasmid_count_path": {"type": "string", "description": "Path to the plasmid count TSV file"},
            },
            "required": [],
        },
    },
    {
        "name": "tool_variant_family_coverage",
        "description": (
            "Coverage of CRE-seq variant families (reference + knockouts/mutants). "
            "PASS when >= 80% of families fully recovered AND zero families missing their reference."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "mapping_table_path": {"type": "string", "description": "Path to the mapping table TSV file"},
                "design_manifest_path": {"type": "string", "description": "Path to the design manifest TSV file"},
            },
            "required": [],
        },
    },
    {
        "name": "tool_library_summary_report",
        "description": (
            "Comprehensive one-shot CRE-seq library QC report. "
            "Runs all applicable tools. Returns overall_pass, failed_checks, warnings, and per-tool results."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "mapping_table_path": {"type": "string", "description": "Path to the mapping table TSV file"},
                "plasmid_count_path": {"type": "string", "description": "Path to the plasmid count TSV file"},
                "design_manifest_path": {"type": "string", "description": "Path to the design manifest TSV file (optional)"},
            },
            "required": [],
        },
    },
    {
        "name": "tool_activity_report",
        "description": (
            "Normalize DNA/RNA counts → compute log2(RNA/DNA) per oligo → call active CREs. "
            "Saves activity_results.tsv. Uses z-test vs. negative controls when available; "
            "falls back to log2 > 1 threshold. Requires plasmid_counts.tsv and rna_counts.tsv."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dna_counts_path": {"type": "string", "description": "Path to plasmid_counts.tsv (optional)"},
                "rna_counts_path": {"type": "string", "description": "Path to rna_counts.tsv (optional)"},
                "design_manifest_path": {"type": "string", "description": "Path to design_manifest.tsv (optional)"},
            },
            "required": [],
        },
    },
    {
        "name": "tool_annotate_motifs",
        "description": (
            "Annotate each oligo with its top JASPAR TF motif using PWM scanning. "
            "Fetches PWMs from JASPAR REST API for ~20 liver/HepG2-relevant TFs (or a custom list), "
            "scans oligo sequences, and adds a top_motif column to activity_results.tsv. "
            "Run this before tool_motif_enrichment_summary or tool_literature_search_for_motifs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "activity_results_path": {"type": "string", "description": "Path to activity_results.tsv (optional)"},
                "design_manifest_path": {"type": "string", "description": "Path to design_manifest.tsv (optional)"},
                "tf_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of TF names to scan (optional, defaults to HepG2-relevant TFs)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "tool_variant_delta_scores",
        "description": (
            "Compute per-mutant delta = mutant_log2_ratio − reference_log2_ratio for each variant family. "
            "Tests significance with BH FDR. Requires activity_results.tsv and design_manifest.tsv. "
            "Saves variant_delta_scores.tsv."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "activity_results_path": {"type": "string", "description": "Path to activity_results.tsv (optional)"},
                "design_manifest_path": {"type": "string", "description": "Path to design_manifest.tsv (optional)"},
            },
            "required": [],
        },
    },
    {
        "name": "tool_normalize_activity",
        "description": (
            "Compute log2 RNA/DNA activity scores from an element-level count table. "
            "Expects columns: element_id (or oligo_id), dna_counts, rna_counts. "
            "Use tool_activity_report for our standard plasmid/RNA count files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "count_table_path": {"type": "string", "description": "Path to count table with dna_counts and rna_counts columns"},
                "pseudocount": {"type": "number", "description": "Pseudocount added before log2 (default 1.0)"},
                "dna_col": {"type": "string", "description": "DNA count column name (default 'dna_counts')"},
                "rna_col": {"type": "string", "description": "RNA count column name (default 'rna_counts')"},
                "element_col": {"type": "string", "description": "Element ID column name (default 'element_id', use 'oligo_id' for our data)"},
            },
            "required": [],
        },
    },
    {
        "name": "tool_call_active_elements",
        "description": (
            "Classify active CREs from activity_results.tsv using an empirical negative-control background. "
            "Computes z-scores, p-values, and BH FDR. Works on our activity_results.tsv (log2_ratio column)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "activity_table_path": {"type": "string", "description": "Path to activity table (default: activity_results.tsv)"},
                "activity_col": {"type": "string", "description": "Activity column name (default 'log2_ratio' for our data)"},
                "category_col": {"type": "string", "description": "Category column for negative controls (default 'designed_category')"},
                "activity_threshold": {"type": "number", "description": "log2 activity threshold for calling active (default 1.0)"},
                "fdr_threshold": {"type": "number", "description": "BH FDR threshold (default 0.05)"},
            },
            "required": [],
        },
    },
    {
        "name": "tool_rank_cre_candidates",
        "description": (
            "Rank CRE candidates by activity strength and statistical confidence. "
            "Returns top N hits from activity_results.tsv with rank scores."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "activity_table_path": {"type": "string", "description": "Path to activity table (default: activity_results.tsv)"},
                "top_n": {"type": "integer", "description": "Number of top candidates to return (default 20)"},
                "activity_col": {"type": "string", "description": "Activity column name (default 'log2_ratio' for our data)"},
                "q_col": {"type": "string", "description": "FDR column name (default 'fdr' for our data)"},
            },
            "required": [],
        },
    },
    {
        "name": "tool_motif_enrichment_summary",
        "description": (
            "Summarize motif enrichment among active vs. inactive CREs. "
            "Requires a 'top_motif' column in the activity table. "
            "Returns active-vs-inactive enrichment ratios per motif."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "activity_table_path": {"type": "string", "description": "Path to activity table with top_motif and active columns"},
                "motif_col": {"type": "string", "description": "Motif column name (default 'top_motif')"},
                "active_col": {"type": "string", "description": "Active flag column name (default 'active')"},
            },
            "required": [],
        },
    },
    {
        "name": "tool_prepare_rag_context",
        "description": (
            "Prepare suggested literature search queries for the top-ranked CRE candidates. "
            "Input is the ranked table from tool_rank_cre_candidates. "
            "Returns suggested PubMed/JASPAR/ENCODE queries for each motif."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ranked_table_path": {"type": "string", "description": "Path to ranked CRE table"},
                "top_n": {"type": "integer", "description": "Number of top elements to include (default 10)"},
                "target_cell_type": {"type": "string", "description": "Target cell type for context-specific queries (e.g. 'HepG2')"},
                "off_target_cell_type": {"type": "string", "description": "Off-target cell type to exclude from queries"},
            },
            "required": [],
        },
    },
    {
        "name": "tool_count_barcodes_from_fastq",
        "description": (
            "Count fixed-position barcodes from a raw FASTQ file. "
            "Returns a barcode count table. Useful for quick barcode tallies before full pipeline."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "fastq_path": {"type": "string", "description": "Path to FASTQ or FASTQ.gz file"},
                "barcode_start": {"type": "integer", "description": "Start position of barcode in read (default 0)"},
                "barcode_length": {"type": "integer", "description": "Length of barcode (default 10)"},
                "max_reads": {"type": "integer", "description": "Maximum reads to process (optional, for quick checks)"},
            },
            "required": ["fastq_path"],
        },
    },
    {
        "name": "tool_aggregate_fastq_counts_to_elements",
        "description": (
            "Convert raw barcode count tables into element-level DNA/RNA counts. "
            "Joins DNA barcode counts + RNA barcode counts with barcode-to-oligo map. "
            "Use after tool_count_barcodes_from_fastq."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dna_barcode_counts_path": {"type": "string", "description": "Path to DNA barcode count table (barcode, count)"},
                "rna_barcode_counts_path": {"type": "string", "description": "Path to RNA barcode count table (barcode, count)"},
                "barcode_map_path": {"type": "string", "description": "Path to barcode→oligo mapping table"},
            },
            "required": ["dna_barcode_counts_path", "rna_barcode_counts_path", "barcode_map_path"],
        },
    },
    {
        "name": "tool_search_pubmed",
        "description": (
            "Search PubMed literature via NCBI E-utilities. "
            "Returns titles, journals, dates, and URLs for matching papers. "
            "Use for literature context on active CREs or transcription factors."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "PubMed search query (supports [Title/Abstract] tags)"},
                "max_results": {"type": "integer", "description": "Maximum results to return (default 5)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "tool_search_jaspar_motif",
        "description": (
            "Search JASPAR for transcription factor motif profiles. "
            "Returns matrix IDs, TF family, class, and URLs. "
            "Use to look up motifs for TFs active in your CRE-seq data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tf_name": {"type": "string", "description": "Transcription factor name (e.g. 'GATA1', 'HNF4A')"},
                "species": {"type": "integer", "description": "NCBI taxonomy ID (default 9606 for human)"},
                "max_results": {"type": "integer", "description": "Maximum results to return (default 5)"},
            },
            "required": ["tf_name"],
        },
    },
    {
        "name": "tool_search_encode_tf",
        "description": (
            "Search ENCODE for TF ChIP-seq or functional genomics experiments. "
            "Returns accessions, assay types, cell lines, and URLs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tf_name": {"type": "string", "description": "Transcription factor name (e.g. 'CTCF', 'SPI1')"},
                "cell_type": {"type": "string", "description": "Cell type to restrict search (e.g. 'HepG2')"},
                "max_results": {"type": "integer", "description": "Maximum results to return (default 5)"},
            },
            "required": ["tf_name"],
        },
    },
    {
        "name": "tool_literature_search_for_motifs",
        "description": (
            "Comprehensive API-backed literature/database search for enriched TF motifs. "
            "Queries PubMed, JASPAR, and ENCODE for each top motif. "
            "Input is the output from tool_motif_enrichment_summary."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "motif_table_path": {"type": "string", "description": "Path to motif enrichment table (motif column required)"},
                "target_cell_type": {"type": "string", "description": "Target cell type for query context (e.g. 'HepG2')"},
                "off_target_cell_type": {"type": "string", "description": "Off-target cell type to exclude"},
                "top_n_motifs": {"type": "integer", "description": "Number of top motifs to search (default 5)"},
            },
            "required": ["motif_table_path"],
        },
    },
    {
        "name": "tool_interpret_literature_evidence",
        "description": (
            "Summarize retrieved literature and database evidence into an interpretation. "
            "Input is the combined evidence table from tool_literature_search_for_motifs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "evidence_table_path": {"type": "string", "description": "Path to combined evidence table"},
            },
            "required": ["evidence_table_path"],
        },
    },
]

_DISPATCH: dict[str, Any] = {
    "tool_barcode_complexity": lambda a: barcode_complexity(
        _p(a, "mapping_table_path", "mapping_table.tsv"), a.get("min_reads_per_barcode", 1)
    ),
    "tool_oligo_recovery": lambda a: oligo_recovery(
        _p(a, "mapping_table_path", "mapping_table.tsv"),
        _p(a, "design_manifest_path", "design_manifest.tsv"),
        a.get("thresholds"),
    ),
    "tool_synthesis_error_profile": lambda a: synthesis_error_profile(
        _p(a, "mapping_table_path", "mapping_table.tsv"),
        _p(a, "design_manifest_path", "design_manifest.tsv") if a.get("design_manifest_path") else None,
    ),
    "tool_barcode_collision_analysis": lambda a: barcode_collision_analysis(
        _p(a, "mapping_table_path", "mapping_table.tsv"), a.get("min_read_support", 2)
    ),
    "tool_barcode_uniformity": lambda a: barcode_uniformity(
        _p(a, "plasmid_count_path", "plasmid_counts.tsv"), a.get("min_barcodes_per_oligo", 5)
    ),
    "tool_gc_content_bias": lambda a: gc_content_bias(
        _p(a, "mapping_table_path", "mapping_table.tsv"),
        _p(a, "design_manifest_path", "design_manifest.tsv"),
        a.get("gc_bins", 10),
    ),
    "tool_oligo_length_qc": lambda a: oligo_length_qc(
        _p(a, "mapping_table_path", "mapping_table.tsv"),
        _p(a, "design_manifest_path", "design_manifest.tsv"),
    ),
    "tool_plasmid_depth_summary": lambda a: plasmid_depth_summary(
        _p(a, "plasmid_count_path", "plasmid_counts.tsv")
    ),
    "tool_variant_family_coverage": lambda a: variant_family_coverage(
        _p(a, "mapping_table_path", "mapping_table.tsv"),
        _p(a, "design_manifest_path", "design_manifest.tsv"),
    ),
    "tool_library_summary_report": lambda a: library_summary_report(
        _p(a, "mapping_table_path", "mapping_table.tsv"),
        _p(a, "plasmid_count_path", "plasmid_counts.tsv"),
        _p(a, "design_manifest_path", "design_manifest.tsv"),
    ),
    "tool_activity_report": lambda a: _activity_report(a),
    "tool_annotate_motifs": lambda a: annotate_top_motifs(
        _p(a, "activity_results_path", "activity_results.tsv"),
        _p(a, "design_manifest_path", "design_manifest.tsv"),
        tf_names=a.get("tf_names"),
        upload_dir=UPLOAD_DIR,
    ),
    "tool_variant_delta_scores": lambda a: compute_variant_delta_scores(
        _p(a, "activity_results_path", "activity_results.tsv"),
        _p(a, "design_manifest_path", "design_manifest.tsv"),
        upload_dir=UPLOAD_DIR,
    ),
    "tool_normalize_activity": lambda a: normalize_activity(
        a["count_table_path"],
        pseudocount=a.get("pseudocount", 1.0),
        dna_col=a.get("dna_col", "dna_counts"),
        rna_col=a.get("rna_col", "rna_counts"),
        element_col=a.get("element_col", "element_id"),
    ),
    "tool_call_active_elements": lambda a: call_active_elements(
        _p(a, "activity_table_path", "activity_results.tsv"),
        activity_col=a.get("activity_col", "log2_ratio"),
        category_col=a.get("category_col", "designed_category"),
        activity_threshold=a.get("activity_threshold", 1.0),
        fdr_threshold=a.get("fdr_threshold", 0.05),
    ),
    "tool_rank_cre_candidates": lambda a: rank_cre_candidates(
        _p(a, "activity_table_path", "activity_results.tsv"),
        top_n=a.get("top_n", 20),
        activity_col=a.get("activity_col", "log2_ratio"),
        q_col=a.get("q_col", "fdr"),
    ),
    "tool_motif_enrichment_summary": lambda a: motif_enrichment_summary(
        _p(a, "activity_table_path", "activity_results.tsv"),
        motif_col=a.get("motif_col", "top_motif"),
        active_col=a.get("active_col", "active"),
    ),
    "tool_prepare_rag_context": lambda a: prepare_rag_context(
        a["ranked_table_path"],
        top_n=a.get("top_n", 10),
        target_cell_type=a.get("target_cell_type"),
        off_target_cell_type=a.get("off_target_cell_type"),
    ),
    "tool_count_barcodes_from_fastq": lambda a: count_barcodes_from_fastq(
        a["fastq_path"],
        barcode_start=a.get("barcode_start", 0),
        barcode_length=a.get("barcode_length", 10),
        max_reads=a.get("max_reads"),
    ),
    "tool_aggregate_fastq_counts_to_elements": lambda a: aggregate_fastq_counts_to_elements(
        a["dna_barcode_counts_path"],
        a["rna_barcode_counts_path"],
        a["barcode_map_path"],
    ),
    "tool_search_pubmed": lambda a: search_pubmed(
        a["query"],
        max_results=a.get("max_results", 5),
    ),
    "tool_search_jaspar_motif": lambda a: search_jaspar_motif(
        a["tf_name"],
        species=a.get("species", 9606),
        max_results=a.get("max_results", 5),
    ),
    "tool_search_encode_tf": lambda a: search_encode_tf(
        a["tf_name"],
        cell_type=a.get("cell_type"),
        max_results=a.get("max_results", 5),
    ),
    "tool_literature_search_for_motifs": lambda a: literature_search_for_motifs(
        a["motif_table_path"],
        target_cell_type=a.get("target_cell_type"),
        off_target_cell_type=a.get("off_target_cell_type"),
        top_n_motifs=a.get("top_n_motifs", 5),
    ),
    "tool_interpret_literature_evidence": lambda a: interpret_literature_evidence(
        a["evidence_table_path"],
    ),
}


@dataclass
class AgentResponse:
    text: str
    tools_called: list[str] = field(default_factory=list)


class ClaudeQCAgent:
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._messages: list[dict] = []

    def send_message(self, prompt: str) -> AgentResponse:
        self._messages.append({"role": "user", "content": prompt})
        tools_called: list[str] = []

        while True:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=4096,
                system=_SYSTEM_PROMPT,
                tools=_TOOLS,
                messages=self._messages,
            )
            self._messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                break

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                tools_called.append(block.name)
                try:
                    content = json.dumps(_summary(_DISPATCH[block.name](block.input)))
                except Exception as exc:
                    content = json.dumps({"error": str(exc)})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content,
                })

            self._messages.append({"role": "user", "content": tool_results})

        text = next(
            (b.text for b in response.content if hasattr(b, "text")), ""
        )
        return AgentResponse(text=text, tools_called=tools_called)

    def reset(self) -> None:
        self._messages = []


def is_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))
