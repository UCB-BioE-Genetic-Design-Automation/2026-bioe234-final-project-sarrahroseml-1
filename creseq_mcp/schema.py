"""
creseq_mcp/schema.py
====================
Canonical data schemas and tool input models for the CRE-seq MCP toolkit.

Row-level schemas (MappingTableRow, PlasmidCountRow, DesignManifestRow) describe
the expected columns of each TSV input.  Tool input models (one per public tool in
qc/library.py) carry configurable thresholds with CRE-seq-appropriate defaults.

CRE-seq conventions encoded here:
- Oligo length: 84–200 bp (Agilent/IDT oPools for enhancer screens)
- Barcode length: 9–11 bp nominal, 8–12 bp acceptable window
- Typical barcodes per oligo: 10–100 (median ~25 is healthy)
- Episomal delivery — no lentiviral integration-site columns needed
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Canonical designed_category values for CRE-seq design manifests
# ---------------------------------------------------------------------------
DesignedCategory = Literal[
    "test_element",
    "scrambled_control",
    "motif_knockout",
    "single_bp_mutant",
    "positive_control",
    "negative_control",
]

# ---------------------------------------------------------------------------
# Row-level schemas  (used for column-presence validation on load)
# ---------------------------------------------------------------------------

MAPPING_TABLE_REQUIRED_COLS = {"barcode", "oligo_id", "n_reads", "cigar", "md_tag", "mapq"}
PLASMID_COUNT_REQUIRED_COLS = {"barcode", "oligo_id", "dna_count"}
DESIGN_MANIFEST_REQUIRED_COLS = {"oligo_id"}


class MappingTableRow(BaseModel):
    """One row of the barcode→oligo mapping table produced by MPRAmatch-style pipelines."""

    barcode: str
    oligo_id: str
    n_reads: int = Field(ge=0)
    cigar: str
    md_tag: str
    mapq: int = Field(ge=0, le=255)


class PlasmidCountRow(BaseModel):
    """One row of the plasmid DNA count table (per-barcode depth in the plasmid prep)."""

    barcode: str
    oligo_id: str
    dna_count: int = Field(ge=0)
    replicate: str | None = None


class DesignManifestRow(BaseModel):
    """One row of the oligo design manifest (FASTA or TSV)."""

    oligo_id: str
    sequence: str | None = None
    length: int | None = Field(default=None, ge=1)
    gc_content: float | None = Field(default=None, ge=0.0, le=1.0)
    designed_category: DesignedCategory | None = None
    parent_element_id: str | None = None


# ---------------------------------------------------------------------------
# Tool input models — one per public function in qc/library.py
# ---------------------------------------------------------------------------

class BarcodComplexityInput(BaseModel):
    mapping_table_path: str
    min_reads_per_barcode: int = Field(default=1, ge=0)


class OligoRecoveryInput(BaseModel):
    mapping_table_path: str
    design_manifest_path: str
    thresholds: list[int] = Field(default=[5, 10, 25])


class SynthesisErrorProfileInput(BaseModel):
    mapping_table_path: str
    design_manifest_path: str | None = None


class BarcodeCollisionInput(BaseModel):
    mapping_table_path: str
    min_read_support: int = Field(default=2, ge=1)


class BarcodeUniformityInput(BaseModel):
    plasmid_count_path: str
    min_barcodes_per_oligo: int = Field(default=5, ge=1)


class GcContentBiasInput(BaseModel):
    mapping_table_path: str
    design_manifest_path: str
    gc_bins: int = Field(default=10, ge=2)


class OligoLengthQcInput(BaseModel):
    mapping_table_path: str
    design_manifest_path: str


class PlasmidDepthSummaryInput(BaseModel):
    plasmid_count_path: str


class VariantFamilyCoverageInput(BaseModel):
    mapping_table_path: str
    design_manifest_path: str


class LibrarySummaryReportInput(BaseModel):
    mapping_table_path: str
    plasmid_count_path: str
    design_manifest_path: str | None = None
    thresholds_config: dict | None = None
