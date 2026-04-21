"""
creseq_mcp/server.py
====================
MCP server entry point for the CRE-seq analysis toolkit.

Registration pattern: decorator-based using ``mcp.server.fastmcp.FastMCP``.
Each public tool in qc/library.py is wrapped here to convert the
``(pd.DataFrame, dict)`` return value to a JSON-serialisable dict before
sending it over the MCP wire.

Run with::

    python -m creseq_mcp.server
    # or
    mcp run creseq_mcp/server.py
"""

from __future__ import annotations

import json
import logging
from typing import Any

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP(
    "creseq-mcp",
    instructions=(
        "CRE-seq library and expression QC toolkit. "
        "Tools operate on barcode→oligo mapping tables and plasmid count tables. "
        "All thresholds default to CRE-seq conventions (84–200 bp oligos, "
        "9–11 bp barcodes, episomal delivery). "
        "Do NOT use for lentiMPRA or STARR-seq without adjusting thresholds."
    ),
)


def _serialise(df_summary_pair: tuple | dict) -> dict[str, Any]:
    """Convert a (DataFrame, dict) tool result to a JSON-safe dict."""
    import pandas as pd  # local import keeps module-level deps minimal

    if isinstance(df_summary_pair, dict):
        # library_summary_report returns a nested dict
        out: dict[str, Any] = {}
        for k, v in df_summary_pair.items():
            if isinstance(v, tuple) and len(v) == 2:
                df, summ = v
                out[k] = {
                    "rows": df.to_dict(orient="records") if isinstance(df, pd.DataFrame) else [],
                    "summary": summ,
                }
            else:
                out[k] = v
        return out

    df, summary = df_summary_pair
    return {
        "rows": df.to_dict(orient="records") if isinstance(df, pd.DataFrame) else [],
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Tool registrations
# ---------------------------------------------------------------------------


@mcp.tool()
def tool_barcode_complexity(
    mapping_table_path: str,
    min_reads_per_barcode: int = 1,
) -> dict:
    """
    Per-oligo barcode count statistics.

    Returns how many distinct barcodes support each designed oligo, what
    fraction are error-free (perfect CIGAR/MD), and the median read depth
    per barcode.  PASS when median barcodes/oligo >= 10.
    """
    return _serialise(barcode_complexity(mapping_table_path, min_reads_per_barcode))


@mcp.tool()
def tool_oligo_recovery(
    mapping_table_path: str,
    design_manifest_path: str,
    thresholds: list[int] | None = None,
) -> dict:
    """
    Recovery rate of designed oligos, broken out by designed_category.

    PASS when test_element recovery@10 >= 80% AND positive_control recovery@10 >= 95%.
    """
    return _serialise(oligo_recovery(mapping_table_path, design_manifest_path, thresholds))


@mcp.tool()
def tool_synthesis_error_profile(
    mapping_table_path: str,
    design_manifest_path: str | None = None,
) -> dict:
    """
    Per-oligo synthesis error characterisation from CIGAR/MD tags.

    Reports mismatches, indels, soft-clip rates, and (if GC content is available)
    Spearman correlation between GC content and synthesis fidelity.
    PASS when median perfect_fraction >= 0.50.
    """
    return _serialise(synthesis_error_profile(mapping_table_path, design_manifest_path))


@mcp.tool()
def tool_barcode_collision_analysis(
    mapping_table_path: str,
    min_read_support: int = 2,
) -> dict:
    """
    Barcodes that map to more than one designed oligo.

    PASS when collision rate < 3% (stricter than generic MPRA because CRE-seq
    uses short 9–11 bp barcodes with limited sequence space).
    """
    return _serialise(barcode_collision_analysis(mapping_table_path, min_read_support))


@mcp.tool()
def tool_barcode_uniformity(
    plasmid_count_path: str,
    min_barcodes_per_oligo: int = 5,
) -> dict:
    """
    Per-oligo barcode abundance evenness in the plasmid pool (Gini coefficient).

    PASS when median Gini < 0.30.
    """
    return _serialise(barcode_uniformity(plasmid_count_path, min_barcodes_per_oligo))


@mcp.tool()
def tool_gc_content_bias(
    mapping_table_path: str,
    design_manifest_path: str,
    gc_bins: int = 10,
) -> dict:
    """
    Synthesis recovery and complexity stratified by oligo GC content.

    Flags GC bins with recovery < 50% of the median bin.  PASS when no such
    dropout bin is found.  Especially important for CRE-seq enhancer libraries
    which are often GC-rich (40–70%).
    """
    return _serialise(gc_content_bias(mapping_table_path, design_manifest_path, gc_bins))


@mcp.tool()
def tool_oligo_length_qc(
    mapping_table_path: str,
    design_manifest_path: str,
) -> dict:
    """
    Synthesis-truncation check for fixed-length CRE-seq oligos.

    Compares observed alignment length (from CIGAR) against designed length.
    PASS when median fraction_full_length >= 0.80.  Warns if the design manifest
    contains oligos of multiple lengths (atypical for CRE-seq).
    """
    return _serialise(oligo_length_qc(mapping_table_path, design_manifest_path))


@mcp.tool()
def tool_plasmid_depth_summary(plasmid_count_path: str) -> dict:
    """
    Barcode-level read-count statistics in the plasmid DNA library.

    PASS when median dna_count >= 10 AND fewer than 10% of barcodes have zero counts.
    """
    return _serialise(plasmid_depth_summary(plasmid_count_path))


@mcp.tool()
def tool_variant_family_coverage(
    mapping_table_path: str,
    design_manifest_path: str,
) -> dict:
    """
    Coverage of CRE-seq variant families (reference + motif knockouts / point mutants).

    PASS when >= 80% of families are fully recovered AND zero families are missing
    their reference sequence.  A missing reference makes delta-score calculation
    impossible for that family.
    """
    return _serialise(variant_family_coverage(mapping_table_path, design_manifest_path))


@mcp.tool()
def tool_library_summary_report(
    mapping_table_path: str,
    plasmid_count_path: str,
    design_manifest_path: str | None = None,
) -> dict:
    """
    Comprehensive one-shot CRE-seq library QC report.

    Runs all applicable tools.  Tools requiring a design manifest are skipped if
    design_manifest_path is not provided.  Returns overall_pass, failed_checks,
    warnings, and per-tool results.
    """
    return _serialise(
        library_summary_report(mapping_table_path, plasmid_count_path, design_manifest_path)
    )


if __name__ == "__main__":
    mcp.run()
