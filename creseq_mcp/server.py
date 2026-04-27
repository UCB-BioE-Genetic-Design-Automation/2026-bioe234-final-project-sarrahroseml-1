"""
creseq_mcp/server.py
====================
MCP server entry point for the CRE-seq analysis toolkit.

Run with::

    python -m creseq_mcp.server
    # or
    mcp run creseq_mcp/server.py
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

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

from creseq_mcp.literature.search import (
    interpret_literature_evidence,
    literature_search_for_motifs,
    motif_enrichment_summary,
    prepare_rag_context,
    rank_cre_candidates,
    search_encode_tf,
    search_jaspar_motif,
    search_pubmed,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(os.environ.get("CRESEQ_UPLOAD_DIR", Path.home() / ".creseq" / "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

mcp = FastMCP(
    "creseq-mcp",
    instructions=(
        "CRE-seq library QC toolkit. "
        "File path arguments are optional — omit them to use data uploaded via the UI. "
        "Do NOT use for lentiMPRA or STARR-seq without adjusting thresholds."
    ),
)

_PAPERS_DIR = Path(__file__).parent / "data" / "papers"


@mcp.resource("paper://agarwal2025-lentimpra")
def paper_agarwal2025() -> str:
    """
    Agarwal et al. 2025, Nature — 'Massively parallel characterization of
    transcriptional regulatory elements'. DOI: 10.1038/s41586-024-08430-9

    Large-scale lentiMPRA of >680,000 cCREs across HepG2, K562 and WTC11 cells.
    This is the primary reference for the ENCODE HepG2 lentiMPRA dataset
    (ENCSR463IRX) used by this pipeline.
    """
    return (_PAPERS_DIR / "agarwal2025_lentimpra.txt").read_text()


def _path(arg: str | None, filename: str) -> str:
    return arg or str(UPLOAD_DIR / filename)


def _summary(result: tuple | dict) -> dict:
    """Extract the summary dict, dropping the DataFrame and coercing numpy scalars."""
    s = result[1] if isinstance(result, tuple) else {
        k: v[1] if isinstance(v, tuple) else v for k, v in result.items()
    }
    return json.loads(json.dumps(s, default=lambda o: o.item() if hasattr(o, "item") else str(o)))

def _serialise(result: tuple | dict) -> dict:
    """Convert a (DataFrame, summary) result to JSON-safe rows + summary."""
    import pandas as pd

    if isinstance(result, tuple):
        df, summary = result
        out = {
            "rows": df.to_dict(orient="records") if isinstance(df, pd.DataFrame) else [],
            "summary": summary,
        }
    else:
        out = result

    return json.loads(
        json.dumps(
            out,
            default=lambda o: o.item() if hasattr(o, "item") else str(o),
        )
    )


# ---------------------------------------------------------------------------
# Tool registrations
# ---------------------------------------------------------------------------


@mcp.tool()
def tool_barcode_complexity(
    mapping_table_path: str | None = None,
    min_reads_per_barcode: int = 1,
) -> dict:
    """
    Per-oligo barcode count statistics.

    Returns how many distinct barcodes support each designed oligo, what
    fraction are error-free (perfect CIGAR/MD), and the median read depth
    per barcode.  PASS when median barcodes/oligo >= 10.
    """
    return _summary(barcode_complexity(
        _path(mapping_table_path, "mapping_table.tsv"), min_reads_per_barcode
    ))


@mcp.tool()
def tool_oligo_recovery(
    mapping_table_path: str | None = None,
    design_manifest_path: str | None = None,
    thresholds: list[int] | None = None,
) -> dict:
    """
    Recovery rate of designed oligos, broken out by designed_category.

    PASS when test_element recovery@10 >= 80% AND positive_control recovery@10 >= 95%.
    """
    return _summary(oligo_recovery(
        _path(mapping_table_path, "mapping_table.tsv"),
        _path(design_manifest_path, "design_manifest.tsv"),
        thresholds,
    ))


@mcp.tool()
def tool_synthesis_error_profile(
    mapping_table_path: str | None = None,
    design_manifest_path: str | None = None,
) -> dict:
    """
    Per-oligo synthesis error characterisation from CIGAR/MD tags.

    Reports mismatches, indels, soft-clip rates, and Spearman correlation
    between GC content and synthesis fidelity.  PASS when median perfect_fraction >= 0.50.
    """
    return _summary(synthesis_error_profile(
        _path(mapping_table_path, "mapping_table.tsv"),
        _path(design_manifest_path, "design_manifest.tsv") if design_manifest_path else None,
    ))


@mcp.tool()
def tool_barcode_collision_analysis(
    mapping_table_path: str | None = None,
    min_read_support: int = 2,
) -> dict:
    """
    Barcodes that map to more than one designed oligo.

    PASS when collision rate < 3%.
    """
    return _summary(barcode_collision_analysis(
        _path(mapping_table_path, "mapping_table.tsv"), min_read_support
    ))


@mcp.tool()
def tool_barcode_uniformity(
    plasmid_count_path: str | None = None,
    min_barcodes_per_oligo: int = 5,
) -> dict:
    """
    Per-oligo barcode abundance evenness in the plasmid pool (Gini coefficient).

    PASS when median Gini < 0.30.
    """
    return _summary(barcode_uniformity(
        _path(plasmid_count_path, "plasmid_counts.tsv"), min_barcodes_per_oligo
    ))


@mcp.tool()
def tool_gc_content_bias(
    mapping_table_path: str | None = None,
    design_manifest_path: str | None = None,
    gc_bins: int = 10,
) -> dict:
    """
    Synthesis recovery stratified by oligo GC content.

    Flags GC bins with recovery < 50% of the median bin.  PASS when no dropout bins found.
    """
    return _summary(gc_content_bias(
        _path(mapping_table_path, "mapping_table.tsv"),
        _path(design_manifest_path, "design_manifest.tsv"),
        gc_bins,
    ))


@mcp.tool()
def tool_oligo_length_qc(
    mapping_table_path: str | None = None,
    design_manifest_path: str | None = None,
) -> dict:
    """
    Synthesis-truncation check comparing observed alignment length to designed length.

    PASS when median fraction_full_length >= 0.80.
    """
    return _summary(oligo_length_qc(
        _path(mapping_table_path, "mapping_table.tsv"),
        _path(design_manifest_path, "design_manifest.tsv"),
    ))


@mcp.tool()
def tool_plasmid_depth_summary(plasmid_count_path: str | None = None) -> dict:
    """
    Barcode-level read-count statistics in the plasmid DNA library.

    PASS when median dna_count >= 10 AND fewer than 10% of barcodes have zero counts.
    """
    return _summary(plasmid_depth_summary(_path(plasmid_count_path, "plasmid_counts.tsv")))


@mcp.tool()
def tool_variant_family_coverage(
    mapping_table_path: str | None = None,
    design_manifest_path: str | None = None,
) -> dict:
    """
    Coverage of CRE-seq variant families (reference + motif knockouts / point mutants).

    PASS when >= 80% of families fully recovered AND zero families missing their reference.
    """
    return _summary(variant_family_coverage(
        _path(mapping_table_path, "mapping_table.tsv"),
        _path(design_manifest_path, "design_manifest.tsv"),
    ))


@mcp.tool()
def tool_library_summary_report(
    mapping_table_path: str | None = None,
    plasmid_count_path: str | None = None,
    design_manifest_path: str | None = None,
) -> dict:
    """
    Comprehensive one-shot CRE-seq library QC report.

    Runs all applicable tools and returns overall_pass, failed_checks, warnings,
    and per-tool summaries.
    """
    return _summary(library_summary_report(
        _path(mapping_table_path, "mapping_table.tsv"),
        _path(plasmid_count_path, "plasmid_counts.tsv"),
        _path(design_manifest_path, "design_manifest.tsv"),
    ))


@mcp.tool()
def tool_run_association(
    fastq_r1: str,
    design_fasta: str,
    fastq_r2: str | None = None,
    fastq_bc: str | None = None,
    labels_path: str | None = None,
    min_cov: int = 3,
    min_frac: float = 0.5,
    mapq_threshold: int = 20,
) -> dict:
    """
    Run the full association step: mappy alignment + STARCODE clustering.

    Maps R1 reads to the design FASTA, links barcodes, clusters with STARCODE,
    and writes mapping_table.tsv, plasmid_counts.tsv, design_manifest.tsv to
    CRESEQ_ASSOC_DIR (default ~/creseq_outputs).

    fastq_r1: R1 oligo reads FASTQ (required).
    design_fasta: FASTA of designed oligo sequences (required).
    fastq_r2: optional R2 paired-end reads.
    fastq_bc: optional barcode index FASTQ (ENCODE i5 format).
    labels_path: optional TSV with oligo_id + designed_category columns.
    """
    from creseq_mcp.association.association import run_association
    import shutil

    assoc_dir = Path(os.environ.get("CRESEQ_ASSOC_DIR", Path.home() / "creseq_outputs"))
    assoc_dir.mkdir(parents=True, exist_ok=True)

    result = run_association(
        fastq_r1=fastq_r1,
        design_fasta=design_fasta,
        outdir=assoc_dir,
        fastq_r2=fastq_r2,
        fastq_bc=fastq_bc,
        labels_path=labels_path,
        min_cov=min_cov,
        min_frac=min_frac,
        mapq_threshold=mapq_threshold,
    )

    # Copy outputs into UPLOAD_DIR so downstream tools can find them
    for fname in ("mapping_table.tsv", "plasmid_counts.tsv", "design_manifest.tsv"):
        src = assoc_dir / fname
        if src.exists():
            shutil.copy(src, UPLOAD_DIR / fname)

    return result

# ---------------------------------------------------------------------------
# Stats tool registrations
# ---------------------------------------------------------------------------

@mcp.tool()
def tool_rank_cre_candidates(
    activity_table_path: str | None = None,
    top_n: int = 20,
    activity_col: str = "log2_ratio",
    q_col: str = "fdr",
) -> dict:
    """
    Rank CRE candidates by activity strength and statistical confidence.
    """
    return _serialise(
        rank_cre_candidates(
            activity_table_path=_path(activity_table_path, "activity_results.tsv"),
            top_n=top_n,
            activity_col=activity_col,
            q_col=q_col,
        )
    )


@mcp.tool()
def tool_motif_enrichment_summary(
    activity_table_path: str,
    motif_col: str = "top_motif",
    active_col: str = "active",
) -> dict:
    """
    Summarize TF motifs enriched among active CREs.
    """
    return _serialise(
        motif_enrichment_summary(
            activity_table_path=activity_table_path,
            motif_col=motif_col,
            active_col=active_col,
        )
    )


@mcp.tool()
def tool_prepare_rag_context(
    ranked_table_path: str,
    top_n: int = 10,
    motif_col: str = "top_motif",
    target_cell_type: str | None = None,
    off_target_cell_type: str | None = None,
) -> dict:
    """
    Prepare top CREs and TF motif search terms for literature/API interpretation.
    """
    return _serialise(
        prepare_rag_context(
            ranked_table_path=ranked_table_path,
            top_n=top_n,
            motif_col=motif_col,
            target_cell_type=target_cell_type,
            off_target_cell_type=off_target_cell_type,
        )
    )


@mcp.tool()
def tool_search_pubmed(
    query: str,
    max_results: int = 5,
    email: str | None = None,
    api_key: str | None = None,
) -> dict:
    """
    Search PubMed for literature evidence using NCBI E-utilities.
    """
    return _serialise(
        search_pubmed(
            query=query,
            max_results=max_results,
            email=email,
            api_key=api_key,
        )
    )


@mcp.tool()
def tool_search_jaspar_motif(
    tf_name: str,
    species: int = 9606,
    collection: str = "CORE",
    max_results: int = 5,
) -> dict:
    """
    Search JASPAR for TF motif matrix profiles.
    """
    return _serialise(
        search_jaspar_motif(
            tf_name=tf_name,
            species=species,
            collection=collection,
            max_results=max_results,
        )
    )


@mcp.tool()
def tool_search_encode_tf(
    tf_name: str,
    cell_type: str | None = None,
    max_results: int = 5,
) -> dict:
    """
    Search ENCODE for TF/cell-type functional genomics records.
    """
    return _serialise(
        search_encode_tf(
            tf_name=tf_name,
            cell_type=cell_type,
            max_results=max_results,
        )
    )


@mcp.tool()
def tool_literature_search_for_motifs(
    motif_table_path: str,
    motif_col: str = "motif",
    target_cell_type: str | None = None,
    off_target_cell_type: str | None = None,
    top_n_motifs: int = 5,
    max_pubmed_results_per_motif: int = 3,
    max_database_results_per_motif: int = 3,
    email: str | None = None,
    ncbi_api_key: str | None = None,
) -> dict:
    """
    Run PubMed, JASPAR, and ENCODE API searches for top enriched motifs.
    """
    return _serialise(
        literature_search_for_motifs(
            motif_table_path=motif_table_path,
            motif_col=motif_col,
            target_cell_type=target_cell_type,
            off_target_cell_type=off_target_cell_type,
            top_n_motifs=top_n_motifs,
            max_pubmed_results_per_motif=max_pubmed_results_per_motif,
            max_database_results_per_motif=max_database_results_per_motif,
            email=email,
            ncbi_api_key=ncbi_api_key,
        )
    )


@mcp.tool()
def tool_interpret_literature_evidence(
    evidence_table_path: str,
) -> dict:
    """
    Summarize API-retrieved literature/database evidence for display.
    """
    return _serialise(
        interpret_literature_evidence(
            evidence_table_path=evidence_table_path,
        )
    )


@mcp.tool()
def tool_process_dna_counting(
    fastq_path: str,
    barcode_len: int = 20,
    barcode_end: str = "3prime",
    max_mismatch: int = 0,
) -> dict:
    """
    Count DNA barcodes from a plasmid-pool FASTQ → overwrites plasmid_counts.tsv.

    Requires mapping_table.tsv from the association step.
    barcode_end: "3prime" (default) or "5prime".
    """
    from creseq_mcp.activity.counting import process_dna_counting

    return process_dna_counting(
        fastq_path,
        str(UPLOAD_DIR / "mapping_table.tsv"),
        UPLOAD_DIR,
        barcode_len=barcode_len,
        barcode_end=barcode_end,
        max_mismatch=max_mismatch,
    )


@mcp.tool()
def tool_process_rna_counting(
    fastq_paths: list[str],
    rep_names: list[str] | None = None,
    barcode_len: int = 20,
    barcode_end: str = "3prime",
    max_mismatch: int = 0,
) -> dict:
    """
    Count RNA barcodes across one or more replicate FASTQs → writes rna_counts.tsv.

    Requires mapping_table.tsv from the association step.
    fastq_paths: list of FASTQ paths, one per replicate.
    rep_names: optional list of replicate labels (default: rep1, rep2, …).
    """
    from creseq_mcp.activity.counting import process_rna_counting

    return process_rna_counting(
        fastq_paths,
        str(UPLOAD_DIR / "mapping_table.tsv"),
        UPLOAD_DIR,
        rep_names=rep_names,
        barcode_len=barcode_len,
        barcode_end=barcode_end,
        max_mismatch=max_mismatch,
    )


@mcp.tool()
def tool_activity_report(
    dna_counts_path: str | None = None,
    rna_counts_path: str | None = None,
    design_manifest_path: str | None = None,
) -> dict:
    """
    Normalize DNA/RNA counts → compute log2(RNA/DNA) per oligo → call active CREs.

    Saves activity_results.tsv to the upload directory.
    Uses z-test vs. negative controls when available; falls back to log2 > 1 threshold.
    """
    from creseq_mcp.activity.normalize import activity_report

    _, summary = activity_report(
        _path(dna_counts_path, "plasmid_counts.tsv"),
        _path(rna_counts_path, "rna_counts.tsv"),
        _path(design_manifest_path, "design_manifest.tsv") if design_manifest_path else None,
        upload_dir=UPLOAD_DIR,
    )
    return summary


@mcp.tool(name="extract_sequences")
def tool_extract_sequences(
    classified_table: str,
    sequence_source: str,
    active_output: str = "active.fa",
    background_output: str = "background.fa",
) -> dict:
    """
    Bridge ``call_active_elements`` → ``motif_enrichment``.

    Reads a classified-elements TSV (with ``element_id``, ``active``,
    ``pvalue`` columns) and a sequence-source TSV (with ``element_id`` +
    ``sequence``) and writes two FASTAs: actives, and inactive test elements
    as background.  Negative controls (NaN pvalue) are excluded from both.
    Returns paths plus per-set counts.
    """
    from creseq_mcp.motifs.enrichment import extract_sequences_to_fasta

    return extract_sequences_to_fasta(
        classified_table=classified_table,
        sequence_source=sequence_source,
        active_output=active_output,
        background_output=background_output,
    )


@mcp.tool()
def tool_motif_enrichment(
    active_fasta: str,
    background_fasta: str,
    motif_database: str = "JASPAR2024",
    collection: str = "CORE",
    tax_group: str = "Vertebrates",
    score_threshold: float = 0.8,
    output_path: str | None = None,
) -> dict:
    """
    Test for TF binding motif enrichment in active CRE-seq elements.

    Scans active and background FASTA sequences against JASPAR motif PWMs on
    both strands and tests each motif for enrichment with one-sided Fisher's
    exact + BH-FDR.  Returns the enrichment table path and a summary of the
    top significant motifs.
    """
    from creseq_mcp.motifs.enrichment import motif_enrichment

    return motif_enrichment(
        active_fasta=active_fasta,
        background_fasta=background_fasta,
        motif_database=motif_database,
        collection=collection,
        tax_group=tax_group,
        score_threshold=score_threshold,
        output_path=output_path,
    )


@mcp.tool()
def tool_plot_creseq(
    data_file: str,
    plot_type: str,
    output_path: str = "plot.png",
    highlight_ids: list[str] | None = None,
    neg_control_ids: list[str] | None = None,
    annotation_file: str | None = None,
) -> dict:
    """
    Generate a publication-quality CRE-seq plot.

    plot_type ∈ {volcano, ranked_activity, replicate_correlation,
    annotation_boxplot, motif_dotplot}.  Returns the path to the saved
    figure plus a natural-language description of what it shows.
    """
    from creseq_mcp.plots.plots import plot_creseq

    return plot_creseq(
        data_file=data_file,
        plot_type=plot_type,
        output_path=output_path,
        highlight_ids=highlight_ids,
        neg_control_ids=neg_control_ids,
        annotation_file=annotation_file,
    )


@mcp.tool()
def tool_annotate_motifs(
    activity_results_path: str | None = None,
    design_manifest_path: str | None = None,
    tf_names: list[str] | None = None,
) -> dict:
    """
    Annotate each CRE with its best-matching TF motif (lightweight JASPAR REST scan).

    Updates activity_results.tsv in-place with a top_motif column.
    Enables the Motif Analysis tab in the frontend and motif_enrichment_summary.

    tf_names: optional list of TF names to scan (defaults to liver/HepG2 panel).
    """
    from creseq_mcp.motifs.annotate import annotate_top_motifs

    return annotate_top_motifs(
        activity_results_path=_path(activity_results_path, "activity_results.tsv"),
        design_manifest_path=_path(design_manifest_path, "design_manifest.tsv"),
        tf_names=tf_names,
        upload_dir=UPLOAD_DIR,
    )


@mcp.tool()
def tool_variant_delta_scores(
    activity_results_path: str | None = None,
    design_manifest_path: str | None = None,
) -> dict:
    """
    Compute variant effect scores: delta = mutant_log2_ratio - reference_log2_ratio.

    Requires oligo IDs with R:/A:/C: prefix convention (lentiMPRA standard),
    or a design manifest with variant_family and is_reference columns.
    Saves variant_delta_scores.tsv to the upload directory.
    """
    from creseq_mcp.variants.delta_scores import compute_variant_delta_scores

    _, summary = compute_variant_delta_scores(
        activity_results_path=_path(activity_results_path, "activity_results.tsv"),
        design_manifest_path=_path(design_manifest_path, "design_manifest.tsv"),
        upload_dir=UPLOAD_DIR,
    )
    return summary


if __name__ == "__main__":
    mcp.run()
