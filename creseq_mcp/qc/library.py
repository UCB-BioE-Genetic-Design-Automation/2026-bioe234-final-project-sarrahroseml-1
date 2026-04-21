"""
creseq_mcp/qc/library.py
========================
Library-QC module for CRE-seq (cis-regulatory element sequencing) analysis.

**Assay context** — CRE-seq (Kwasnieski & Cohen 2012; Melnikov et al. 2012) is a
massively parallel reporter assay in which short candidate enhancers (84–200 bp) are
cloned episomally upstream of a minimal promoter and a short barcode (9–11 bp).
Activity is read out as RNA barcode counts / DNA barcode counts.

**Scope** — This module covers library-side QC only: oligo synthesis fidelity,
barcode diversity, plasmid representation, and variant-family completeness.  It
operates BEFORE any RNA analysis.

**NOT for** — lentiMPRA (integration-site bias, UMI deduplication), STARR-seq
(fragment-length distributions), or SuRE (long-insert coverage).  Those belong in
separate modules if they are ever added.

**Tidy-schema contract** — Every public tool:
  1. Accepts a path (or paths) to canonical TSV inputs defined in creseq_mcp/schema.py
  2. Returns ``tuple[pd.DataFrame, dict]`` — a per-element tidy DataFrame plus a
     summary dict that always contains a top-level ``"pass"`` bool and a ``"warnings"``
     list.
  3. Never writes files, never plots, never prints — use logging.
"""

from __future__ import annotations

import logging
import re
import warnings
from pathlib import Path
from statistics import mode as _stat_mode
from typing import Any

import numpy as np
import pandas as pd

from creseq_mcp.schema import (
    DESIGN_MANIFEST_REQUIRED_COLS,
    MAPPING_TABLE_REQUIRED_COLS,
    PLASMID_COUNT_REQUIRED_COLS,
    BarcodComplexityInput,
    BarcodeCollisionInput,
    BarcodeUniformityInput,
    GcContentBiasInput,
    LibrarySummaryReportInput,
    OligoLengthQcInput,
    OligoRecoveryInput,
    PlasmidDepthSummaryInput,
    SynthesisErrorProfileInput,
    VariantFamilyCoverageInput,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CRE-seq conventions (used as default thresholds and validation bounds)
# ---------------------------------------------------------------------------
_OLIGO_LEN_MIN: int = 84
_OLIGO_LEN_MAX: int = 200
_BC_LEN_MIN: int = 8
_BC_LEN_MAX: int = 12
_BC_LEN_NOMINAL_MIN: int = 9
_BC_LEN_NOMINAL_MAX: int = 11

try:
    from scipy.stats import spearmanr as _spearmanr  # type: ignore

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Custom warning class
# ---------------------------------------------------------------------------


class CreSeqAssumptionWarning(UserWarning):
    """
    Raised (via warnings.warn) when the library's observed characteristics fall
    outside CRE-seq conventions — e.g., barcodes longer than 12 bp (lentiMPRA
    range) or oligos longer than 200 bp (STARR-seq range).

    These are warnings, not errors, because the tools can still run; they just
    may produce misleading QC calls if the wrong assay type was loaded.
    """


# ---------------------------------------------------------------------------
# Private I/O helpers
# ---------------------------------------------------------------------------

_TSV_SUFFIXES = {".tsv", ".txt"}
_TSV_GZ_SUFFIXES = {".gz"}


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.exists():
        raise ValueError(f"Path does not exist: {p}")
    return p


def _read_tsv(path: Path) -> pd.DataFrame:
    """Read a plain or gzip-compressed TSV / txt file."""
    suffixes = {s.lower() for s in path.suffixes}
    if ".gz" in suffixes:
        return pd.read_csv(path, sep="\t", compression="gzip")
    return pd.read_csv(path, sep="\t")


def _check_columns(df: pd.DataFrame, required: set[str], source: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{source}: missing required columns {missing}. "
            f"Found: {list(df.columns)}"
        )


def _load_mapping_table(path: str | Path) -> pd.DataFrame:
    """Load and validate the barcode→oligo mapping table (MPRAmatch-style TSV)."""
    p = _resolve_path(path)
    df = _read_tsv(p)
    if df.empty:
        raise ValueError(f"Mapping table is empty: {p}")
    _check_columns(df, MAPPING_TABLE_REQUIRED_COLS, str(p))
    df["n_reads"] = df["n_reads"].astype(int)
    df["mapq"] = df["mapq"].astype(int)
    df["md_tag"] = df["md_tag"].astype(str)  # numeric-only MDs (e.g. "84") are read as int
    logger.debug("Loaded mapping table: %d rows from %s", len(df), p)
    return df


def _load_plasmid_counts(path: str | Path) -> pd.DataFrame:
    """Load and validate the plasmid DNA count table."""
    p = _resolve_path(path)
    df = _read_tsv(p)
    if df.empty:
        raise ValueError(f"Plasmid count table is empty: {p}")
    _check_columns(df, PLASMID_COUNT_REQUIRED_COLS, str(p))
    df["dna_count"] = df["dna_count"].astype(int)
    logger.debug("Loaded plasmid count table: %d rows from %s", len(df), p)
    return df


def _load_design_manifest(path: str | Path) -> pd.DataFrame:
    """
    Load the oligo design manifest.

    Accepts:
    - TSV with columns: oligo_id, sequence (opt), length (opt), gc_content (opt),
      designed_category (opt), parent_element_id (opt)
    - FASTA: oligo_id from header, sequence from body; length and gc_content computed;
      designed_category and parent_element_id will be null.
    """
    p = _resolve_path(path)
    lower = p.name.lower()
    if lower.endswith(".fa") or lower.endswith(".fasta") or lower.endswith(".fa.gz"):
        df = _parse_fasta(p)
    else:
        df = _read_tsv(p)

    if df.empty:
        raise ValueError(f"Design manifest is empty: {p}")
    _check_columns(df, DESIGN_MANIFEST_REQUIRED_COLS, str(p))

    # Compute missing length / gc_content from sequence if available
    if "sequence" in df.columns and df["sequence"].notna().any():
        seq_col = df["sequence"].fillna("")
        if "length" not in df.columns or df["length"].isna().any():
            df["length"] = seq_col.str.len().where(seq_col != "", other=np.nan)
        if "gc_content" not in df.columns or df["gc_content"].isna().any():
            df["gc_content"] = seq_col.apply(
                lambda s: (s.count("G") + s.count("C")) / len(s) if s else np.nan
            )

    for col in ("designed_category", "parent_element_id", "length", "gc_content"):
        if col not in df.columns:
            df[col] = np.nan

    logger.debug("Loaded design manifest: %d rows from %s", len(df), p)
    return df


def _parse_fasta(path: Path) -> pd.DataFrame:
    """Minimal FASTA parser; produces oligo_id, sequence, length, gc_content columns."""
    records: list[dict] = []
    current_id: str | None = None
    seq_parts: list[str] = []

    opener = __import__("gzip").open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    seq = "".join(seq_parts).upper()
                    records.append(_fasta_record(current_id, seq))
                current_id = line[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(line)

    if current_id is not None:
        seq = "".join(seq_parts).upper()
        records.append(_fasta_record(current_id, seq))

    return pd.DataFrame(records)


def _fasta_record(oligo_id: str, seq: str) -> dict:
    gc = (seq.count("G") + seq.count("C")) / len(seq) if seq else np.nan
    return {
        "oligo_id": oligo_id,
        "sequence": seq,
        "length": len(seq),
        "gc_content": gc,
        "designed_category": None,
        "parent_element_id": None,
    }


# ---------------------------------------------------------------------------
# CIGAR / MD parsing
# ---------------------------------------------------------------------------

_CIGAR_RE = re.compile(r"(\d+)([MIDNSHP=X])")
_MD_DELETION_RE = re.compile(r"\^[A-Z]+")
_MD_MISMATCH_RE = re.compile(r"[A-Z]")


def _parse_cigar_errors(cigar: str, md: str) -> dict[str, Any]:
    """
    Parse a CIGAR string and MD tag from an oligo-read alignment and return
    per-read synthesis error statistics.

    The query is an oligo read; the reference is the designed oligo sequence.
    Soft-clipped bases indicate truncation or adaptor contamination.

    Parameters
    ----------
    cigar : str  CIGAR string (e.g. ``84M``, ``5S79M``, ``10M1D73M``)
    md    : str  MD tag string (e.g. ``84``, ``41A42``, ``^ATG81``)

    Returns
    -------
    dict with keys:
        mismatches, insertions, deletions, soft_clipped, is_perfect, observed_length
    """
    ops = _CIGAR_RE.findall(cigar)
    if not ops:
        raise ValueError(f"Cannot parse CIGAR string: {cigar!r}")

    insertions = 0
    deletions = 0
    soft_clipped = 0
    query_length = 0  # all query-consuming ops (M + I + S)

    for count_str, op in ops:
        count = int(count_str)
        if op in ("M", "=", "X"):
            query_length += count
        elif op == "I":
            insertions += count
            query_length += count
        elif op == "D":
            deletions += count
        elif op == "S":
            soft_clipped += count
            query_length += count
        # H (hard clip) and N/P do not consume query

    # Mismatches from MD: remove deletion markers (^ATG) then count remaining letters
    md_no_del = _MD_DELETION_RE.sub("", md)
    mismatches = len(_MD_MISMATCH_RE.findall(md_no_del))

    is_perfect = mismatches == 0 and insertions == 0 and deletions == 0 and soft_clipped == 0

    return {
        "mismatches": mismatches,
        "insertions": insertions,
        "deletions": deletions,
        "soft_clipped": soft_clipped,
        "is_perfect": bool(is_perfect),
        "observed_length": query_length,
    }


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def _gini(values: np.ndarray) -> float:
    """
    Gini coefficient of a non-negative array (0 = perfectly even, 1 = maximally uneven).
    Uses the sorting-based formula: G = (2·Σ(i·x_i) − (n+1)·Σx_i) / (n·Σx_i).
    """
    v = np.sort(np.abs(values.astype(float)))
    n = len(v)
    total = v.sum()
    if n == 0 or total == 0.0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * v).sum() - (n + 1) * total) / (n * total))


def _effective_count(counts: np.ndarray) -> float:
    """Effective number of elements = e^(Shannon entropy), also called Hill number N1."""
    c = counts.astype(float)
    total = c.sum()
    if total == 0.0:
        return 0.0
    p = c[c > 0] / total
    return float(np.exp(-np.sum(p * np.log(p))))


def _apply_thresholds(summary: dict, thresholds: dict[str, float]) -> dict[str, bool]:
    """
    Evaluate ``summary[key] >= value`` for each ``{key: value}`` in *thresholds*.
    Returns a dict of ``{key: bool}``.
    """
    return {k: float(summary.get(k, 0)) >= v for k, v in thresholds.items()}


def _validate_creseq_assumptions(design_manifest: pd.DataFrame) -> list[str]:
    """
    Inspect a design manifest for characteristics inconsistent with CRE-seq conventions.

    Returns a list of warning strings (empty = all good).
    """
    msgs: list[str] = []

    if "length" in design_manifest.columns:
        lengths = design_manifest["length"].dropna()
        if len(lengths):
            too_short = (lengths < _OLIGO_LEN_MIN).sum()
            too_long = (lengths > _OLIGO_LEN_MAX).sum()
            if too_short:
                msgs.append(
                    f"{too_short} designed oligos shorter than {_OLIGO_LEN_MIN} bp "
                    f"(CRE-seq convention is {_OLIGO_LEN_MIN}–{_OLIGO_LEN_MAX} bp)."
                )
            if too_long:
                msgs.append(
                    f"{too_long} designed oligos longer than {_OLIGO_LEN_MAX} bp; "
                    f"consider verifying assay type (STARR-seq uses longer inserts)."
                )

    return msgs


def _parse_errors_for_df(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorise _parse_cigar_errors across a mapping-table DataFrame."""
    parsed = df.apply(
        lambda r: _parse_cigar_errors(r["cigar"], r["md_tag"]),
        axis=1,
        result_type="expand",
    )
    return pd.concat([df.reset_index(drop=True), parsed], axis=1)


# ---------------------------------------------------------------------------
# Tool 1: barcode_complexity
# ---------------------------------------------------------------------------


def barcode_complexity(
    mapping_table_path: str | Path,
    min_reads_per_barcode: int = 1,
) -> tuple[pd.DataFrame, dict]:
    """
    Question answered:
        How many distinct barcodes support each designed oligo, and how evenly
        are reads distributed across those barcodes?

    Inputs:
        mapping_table_path    : path to barcode→oligo mapping TSV
        min_reads_per_barcode : discard barcodes with fewer reads than this
                                (removes likely sequencing noise)

    Outputs:
        DataFrame — one row per oligo_id:
            oligo_id, n_barcodes, n_perfect_barcodes, n_error_barcodes,
            median_reads_per_barcode
        summary dict — median_barcodes_per_oligo, fraction_oligos_gte_{5,10,25,50,100},
            median_barcode_length_bp, warnings, pass

    Pass/fail criteria:
        PASS  iff  median barcodes/oligo >= 10

    CRE-seq-specific notes:
        - Barcode length outside 8–12 bp triggers a CreSeqAssumptionWarning; healthy
          CRE-seq libraries use 9–11 bp barcodes.
        - Coverage thresholds (5, 10, 25, 50, 100) reflect CRE-seq's typical
          10–100 barcodes/oligo range, not lentiMPRA's smaller dynamic range.
    """
    params = BarcodComplexityInput(
        mapping_table_path=str(mapping_table_path),
        min_reads_per_barcode=min_reads_per_barcode,
    )
    df = _load_mapping_table(params.mapping_table_path)
    df = df[df["n_reads"] >= params.min_reads_per_barcode].copy()

    if df.empty:
        raise ValueError(
            f"No barcodes pass min_reads_per_barcode={params.min_reads_per_barcode}."
        )

    # Validate barcode lengths
    bc_lengths = df["barcode"].str.len()
    median_bc_len = float(bc_lengths.median())
    warn_msgs: list[str] = []

    if not (_BC_LEN_MIN <= median_bc_len <= _BC_LEN_MAX):
        msg = (
            f"Median barcode length {median_bc_len:.0f} bp is outside the CRE-seq "
            f"acceptable window ({_BC_LEN_MIN}–{_BC_LEN_MAX} bp).  "
            f"Nominal CRE-seq range is {_BC_LEN_NOMINAL_MIN}–{_BC_LEN_NOMINAL_MAX} bp. "
            f"Verify that the correct assay type is loaded."
        )
        warnings.warn(msg, CreSeqAssumptionWarning, stacklevel=2)
        warn_msgs.append(msg)
    elif not (_BC_LEN_NOMINAL_MIN <= median_bc_len <= _BC_LEN_NOMINAL_MAX):
        msg = (
            f"Median barcode length {median_bc_len:.0f} bp is in the acceptable "
            f"window but outside the nominal CRE-seq range "
            f"({_BC_LEN_NOMINAL_MIN}–{_BC_LEN_NOMINAL_MAX} bp)."
        )
        warn_msgs.append(msg)
        logger.warning(msg)

    # Per-barcode error parsing
    df = _parse_errors_for_df(df)

    # Per-oligo aggregation
    grp = df.groupby("oligo_id")
    result = pd.DataFrame(
        {
            "n_barcodes": grp["barcode"].count(),
            "n_perfect_barcodes": grp["is_perfect"].sum().astype(int),
            "median_reads_per_barcode": grp["n_reads"].median(),
        }
    ).reset_index()
    result["n_error_barcodes"] = result["n_barcodes"] - result["n_perfect_barcodes"]

    # Summary
    thresholds = [5, 10, 25, 50, 100]
    summary: dict[str, Any] = {
        "median_barcodes_per_oligo": float(result["n_barcodes"].median()),
        "median_barcode_length_bp": median_bc_len,
        "n_oligos": int(len(result)),
        "warnings": warn_msgs,
    }
    for t in thresholds:
        summary[f"fraction_oligos_gte_{t}_barcodes"] = float(
            (result["n_barcodes"] >= t).mean()
        )
    summary["pass"] = summary["median_barcodes_per_oligo"] >= 10

    return result, summary


# ---------------------------------------------------------------------------
# Tool 2: oligo_recovery
# ---------------------------------------------------------------------------


def oligo_recovery(
    mapping_table_path: str | Path,
    design_manifest_path: str | Path,
    thresholds: list[int] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Question answered:
        What fraction of designed oligos are detected in the sequenced library,
        broken out by design category?

    Inputs:
        mapping_table_path   : path to barcode→oligo mapping TSV
        design_manifest_path : path to design manifest TSV/FASTA
        thresholds           : list of barcode-count thresholds to evaluate
                               (default [5, 10, 25])

    Outputs:
        DataFrame — one row per designed oligo:
            oligo_id, designed_category, designed (True), recovered (bool),
            n_barcodes, passes_threshold_N (bool, one column per threshold)
        summary dict — recovery_by_category, warnings, pass

    Pass/fail criteria:
        PASS  iff  test_element recovery@10 >= 0.80
               AND positive_control recovery@10 >= 0.95

    CRE-seq-specific notes:
        - CRE-seq libraries use STRICTER recovery thresholds than lentiMPRA because
          episomal transfection variance cannot be rescued by deeper sequencing.
        - Positive controls at < 95% recovery@10 indicate a synthesis or cloning
          failure that invalidates downstream fold-change normalisation.
        - designed_category breakdown is critical: a missing positive control is a
          red flag; a missing test_element is merely a data loss.
    """
    if thresholds is None:
        thresholds = [5, 10, 25]
    params = OligoRecoveryInput(
        mapping_table_path=str(mapping_table_path),
        design_manifest_path=str(design_manifest_path),
        thresholds=thresholds,
    )

    mapping = _load_mapping_table(params.mapping_table_path)
    manifest = _load_design_manifest(params.design_manifest_path)

    # Barcode counts per oligo
    bc_counts = (
        mapping.groupby("oligo_id")["barcode"]
        .count()
        .reset_index()
        .rename(columns={"barcode": "n_barcodes"})
    )

    result = manifest[["oligo_id", "designed_category"]].copy()
    result["designed"] = True
    result = result.merge(bc_counts, on="oligo_id", how="left")
    result["n_barcodes"] = result["n_barcodes"].fillna(0).astype(int)
    result["recovered"] = result["n_barcodes"] > 0

    for t in params.thresholds:
        result[f"passes_threshold_{t}"] = result["n_barcodes"] >= t

    # Per-category summary
    cat_summary: dict[str, dict] = {}
    for cat, grp in result.groupby("designed_category", dropna=False):
        cat_key = str(cat) if cat is not None else "unknown"
        entry: dict[str, Any] = {
            "n_designed": int(len(grp)),
            "n_recovered": int(grp["recovered"].sum()),
        }
        entry["recovery_rate"] = (
            entry["n_recovered"] / entry["n_designed"] if entry["n_designed"] else 0.0
        )
        for t in params.thresholds:
            col = f"passes_threshold_{t}"
            entry[f"recovery_at_{t}"] = (
                float(grp[col].mean()) if col in grp.columns else None
            )
        cat_summary[cat_key] = entry

    warn_msgs: list[str] = []
    te = cat_summary.get("test_element", {})
    pc = cat_summary.get("positive_control", {})

    te_rate = te.get("recovery_at_10", 1.0) or 1.0
    pc_rate = pc.get("recovery_at_10", 1.0) or 1.0

    te_pass = te_rate >= 0.80 if te else True
    pc_pass = pc_rate >= 0.95 if pc else True

    if not te_pass:
        warn_msgs.append(
            f"test_element recovery@10 = {te_rate:.1%} < 80% — library quality is suspect."
        )
    if not pc_pass:
        warn_msgs.append(
            f"positive_control recovery@10 = {pc_rate:.1%} < 95% — "
            f"normalisation may be invalid."
        )

    summary: dict[str, Any] = {
        "recovery_by_category": cat_summary,
        "warnings": warn_msgs,
        "pass": te_pass and pc_pass,
    }
    return result, summary


# ---------------------------------------------------------------------------
# Tool 3: synthesis_error_profile
# ---------------------------------------------------------------------------


def synthesis_error_profile(
    mapping_table_path: str | Path,
    design_manifest_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Question answered:
        What is the per-oligo rate of synthesis errors (mismatches, indels,
        truncations) as inferred from CIGAR/MD tags, and is there a GC-content
        bias in synthesis fidelity?

    Inputs:
        mapping_table_path   : path to barcode→oligo mapping TSV
        design_manifest_path : optional; if provided and contains gc_content,
                               Spearman(GC, perfect_fraction) is reported

    Outputs:
        DataFrame — one row per oligo_id:
            oligo_id, n_barcodes, n_perfect_barcodes, perfect_fraction,
            mean_mismatches, mean_insertions, mean_deletions, mean_soft_clipped
        summary dict — median_perfect_fraction, gc_spearman_r (if manifest given),
            warnings, pass

    Pass/fail criteria:
        PASS  iff  median perfect_fraction >= 0.50

    CRE-seq-specific notes:
        - Default oligo-length assumption is 84–200 bp (Agilent/IDT oPools range).
          A warning is raised if designed oligos fall outside this window.
        - Agilent 244K and IDT oPools pools show modest GC bias in synthesis yield;
          reporting Spearman(GC, perfect_fraction) helps identify affected oligos.
    """
    params = SynthesisErrorProfileInput(
        mapping_table_path=str(mapping_table_path),
        design_manifest_path=str(design_manifest_path) if design_manifest_path else None,
    )

    df = _load_mapping_table(params.mapping_table_path)
    df = _parse_errors_for_df(df)

    grp = df.groupby("oligo_id")
    result = pd.DataFrame(
        {
            "n_barcodes": grp["barcode"].count(),
            "n_perfect_barcodes": grp["is_perfect"].sum().astype(int),
            "mean_mismatches": grp["mismatches"].mean(),
            "mean_insertions": grp["insertions"].mean(),
            "mean_deletions": grp["deletions"].mean(),
            "mean_soft_clipped": grp["soft_clipped"].mean(),
        }
    ).reset_index()
    result["perfect_fraction"] = result["n_perfect_barcodes"] / result["n_barcodes"].clip(lower=1)

    warn_msgs: list[str] = []
    gc_r: float | None = None
    gc_p: float | None = None

    if params.design_manifest_path:
        manifest = _load_design_manifest(params.design_manifest_path)
        asm_warns = _validate_creseq_assumptions(manifest)
        for w in asm_warns:
            warnings.warn(w, CreSeqAssumptionWarning, stacklevel=2)
        warn_msgs.extend(asm_warns)

        if "gc_content" in manifest.columns and _HAS_SCIPY:
            merged = result.merge(
                manifest[["oligo_id", "gc_content"]].dropna(subset=["gc_content"]),
                on="oligo_id",
                how="inner",
            )
            if len(merged) >= 5:
                corr_result = _spearmanr(merged["gc_content"], merged["perfect_fraction"])
                gc_r = float(corr_result.statistic)
                gc_p = float(corr_result.pvalue)
                if abs(gc_r) > 0.3:
                    warn_msgs.append(
                        f"GC–synthesis fidelity Spearman r={gc_r:.3f} (p={gc_p:.3g}): "
                        f"notable GC bias in this pool."
                    )
        elif "gc_content" in manifest.columns and not _HAS_SCIPY:
            warn_msgs.append("scipy not installed; GC–fidelity Spearman correlation skipped.")

    median_pf = float(result["perfect_fraction"].median())
    summary: dict[str, Any] = {
        "median_perfect_fraction": median_pf,
        "n_oligos": int(len(result)),
        "warnings": warn_msgs,
        "pass": median_pf >= 0.50,
    }
    if gc_r is not None:
        summary["gc_spearman_r"] = gc_r
        summary["gc_spearman_p"] = gc_p

    return result, summary


# ---------------------------------------------------------------------------
# Tool 4: barcode_collision_analysis
# ---------------------------------------------------------------------------


def barcode_collision_analysis(
    mapping_table_path: str | Path,
    min_read_support: int = 2,
) -> tuple[pd.DataFrame, dict]:
    """
    Question answered:
        What fraction of barcodes map to more than one designed oligo (collisions),
        and which oligos are most affected?

    Inputs:
        mapping_table_path : path to barcode→oligo mapping TSV
        min_read_support   : minimum n_reads for a barcode to be considered a
                             real mapping (default 2 to remove singleton noise)

    Outputs:
        DataFrame — one row per barcode that collides:
            barcode, n_oligos_mapped, oligo_ids (comma-joined), max_n_reads
        summary dict — n_barcodes_total, n_collisions, collision_rate, warnings, pass

    Pass/fail criteria:
        PASS  iff  collision_rate < 0.03

    CRE-seq-specific notes:
        - 9–11 bp barcodes have only 4M–4B possible sequences.  For libraries of
          10⁵ elements × 25 barcodes/oligo = 2.5M total barcodes, birthday-paradox
          collisions are non-trivial.  The 3% ceiling is stricter than the 5% used
          for longer-barcode MPRA protocols.
        - Collisions can arise from both true birthday-paradox coincidences and
          from chimeric PCR products during library preparation.
    """
    params = BarcodeCollisionInput(
        mapping_table_path=str(mapping_table_path),
        min_read_support=min_read_support,
    )

    df = _load_mapping_table(params.mapping_table_path)
    df = df[df["n_reads"] >= params.min_read_support]

    if df.empty:
        raise ValueError(
            f"No barcodes pass min_read_support={params.min_read_support}."
        )

    n_total = df["barcode"].nunique()

    bc_oligos = df.groupby("barcode").agg(
        n_oligos_mapped=("oligo_id", "nunique"),
        oligo_ids=("oligo_id", lambda x: ",".join(sorted(x.unique()))),
        max_n_reads=("n_reads", "max"),
    ).reset_index()

    collisions = bc_oligos[bc_oligos["n_oligos_mapped"] > 1].copy()
    n_collisions = len(collisions)
    collision_rate = n_collisions / n_total if n_total else 0.0

    warn_msgs: list[str] = []
    if collision_rate >= 0.03:
        warn_msgs.append(
            f"Collision rate {collision_rate:.1%} >= 3% CRE-seq threshold. "
            f"Consider lengthening barcodes or reducing library complexity."
        )

    summary: dict[str, Any] = {
        "n_barcodes_total": int(n_total),
        "n_collisions": int(n_collisions),
        "collision_rate": float(collision_rate),
        "warnings": warn_msgs,
        "pass": collision_rate < 0.03,
    }
    return collisions, summary


# ---------------------------------------------------------------------------
# Tool 5: barcode_uniformity
# ---------------------------------------------------------------------------


def barcode_uniformity(
    plasmid_count_path: str | Path,
    min_barcodes_per_oligo: int = 5,
) -> tuple[pd.DataFrame, dict]:
    """
    Question answered:
        How evenly are barcodes represented in the plasmid library (DNA input),
        as measured by the Gini coefficient of per-barcode counts within each oligo?

    Inputs:
        plasmid_count_path    : path to plasmid DNA count TSV
        min_barcodes_per_oligo: oligos with fewer barcodes are excluded from
                                Gini calculation (too few to be meaningful)

    Outputs:
        DataFrame — one row per oligo_id:
            oligo_id, n_barcodes, total_dna_count, gini_coefficient,
            effective_barcodes, cv_dna_count
        summary dict — median_gini, median_effective_barcodes, n_oligos_excluded,
            warnings, pass

    Pass/fail criteria:
        PASS  iff  median Gini < 0.30

    CRE-seq-specific notes:
        - CRE-seq libraries run thinner than lentiMPRA (fewer barcodes/oligo),
          so the default min_barcodes_per_oligo is 5, not 10.
        - High Gini (> 0.4) with low barcode count suggests barcode jackpotting
          during PCR amplification of the plasmid library.
    """
    params = BarcodeUniformityInput(
        plasmid_count_path=str(plasmid_count_path),
        min_barcodes_per_oligo=min_barcodes_per_oligo,
    )

    df = _load_plasmid_counts(params.plasmid_count_path)

    def _oligo_stats(grp: pd.DataFrame) -> pd.Series:
        counts = grp["dna_count"].values.astype(float)
        return pd.Series(
            {
                "n_barcodes": len(counts),
                "total_dna_count": counts.sum(),
                "gini_coefficient": _gini(counts),
                "effective_barcodes": _effective_count(counts),
                "cv_dna_count": (
                    counts.std() / counts.mean() if counts.mean() > 0 else np.nan
                ),
            }
        )

    all_stats = df.groupby("oligo_id").apply(_oligo_stats, include_groups=False).reset_index()
    excluded = all_stats[all_stats["n_barcodes"] < params.min_barcodes_per_oligo]
    included = all_stats[all_stats["n_barcodes"] >= params.min_barcodes_per_oligo]

    warn_msgs: list[str] = []
    if len(included) == 0:
        raise ValueError(
            f"No oligos have >= {params.min_barcodes_per_oligo} barcodes. "
            f"Lower min_barcodes_per_oligo or check the plasmid count table."
        )

    median_gini = float(included["gini_coefficient"].median())
    if median_gini >= 0.40:
        warn_msgs.append(
            f"Median Gini {median_gini:.3f} >= 0.40 suggests PCR jackpotting or "
            f"extreme barcode skew in the plasmid prep."
        )

    summary: dict[str, Any] = {
        "median_gini": median_gini,
        "median_effective_barcodes": float(included["effective_barcodes"].median()),
        "n_oligos_evaluated": int(len(included)),
        "n_oligos_excluded": int(len(excluded)),
        "warnings": warn_msgs,
        "pass": median_gini < 0.30,
    }
    return all_stats, summary


# ---------------------------------------------------------------------------
# Tool 6: gc_content_bias
# ---------------------------------------------------------------------------


def gc_content_bias(
    mapping_table_path: str | Path,
    design_manifest_path: str | Path,
    gc_bins: int = 10,
) -> tuple[pd.DataFrame, dict]:
    """
    Question answered:
        Does synthesis yield (recovery rate and barcode complexity) vary systematically
        with the GC content of designed oligos?

    Inputs:
        mapping_table_path   : path to barcode→oligo mapping TSV
        design_manifest_path : path to design manifest (must contain gc_content)
        gc_bins              : number of GC bins for stratified analysis

    Outputs:
        DataFrame — one row per GC bin:
            gc_bin_center, gc_bin_lo, gc_bin_hi, n_designed, n_recovered,
            recovery_rate, median_barcodes
        summary dict — overall_recovery, gc_bias_detected, worst_bin, warnings, pass

    Pass/fail criteria:
        PASS  iff  no GC bin has recovery_rate < 0.5 × median bin recovery_rate

    CRE-seq-specific notes:
        - Transcriptional enhancers are often GC-rich (40–70%), making GC-dropout
          especially damaging for CRE-seq screens.
        - Agilent oPools and IDT eBlocks differ in GC synthesis tolerance; this
          tool flags libraries where a specific GC window is depleted.
    """
    params = GcContentBiasInput(
        mapping_table_path=str(mapping_table_path),
        design_manifest_path=str(design_manifest_path),
        gc_bins=gc_bins,
    )

    mapping = _load_mapping_table(params.mapping_table_path)
    manifest = _load_design_manifest(params.design_manifest_path)

    if "gc_content" not in manifest.columns or manifest["gc_content"].isna().all():
        raise ValueError(
            "Design manifest does not contain gc_content. "
            "Provide a TSV with a gc_content column or a FASTA so it can be computed."
        )

    asm_warns = _validate_creseq_assumptions(manifest)
    for w in asm_warns:
        warnings.warn(w, CreSeqAssumptionWarning, stacklevel=2)

    # Per-oligo barcode count
    bc_counts = (
        mapping.groupby("oligo_id")["barcode"]
        .count()
        .reset_index()
        .rename(columns={"barcode": "n_barcodes"})
    )
    merged = manifest[["oligo_id", "gc_content"]].dropna(subset=["gc_content"]).merge(
        bc_counts, on="oligo_id", how="left"
    )
    merged["n_barcodes"] = merged["n_barcodes"].fillna(0).astype(int)
    merged["recovered"] = merged["n_barcodes"] > 0

    # Bin by GC
    merged["gc_bin"] = pd.cut(merged["gc_content"], bins=params.gc_bins, labels=False)
    bin_edges = pd.cut(merged["gc_content"], bins=params.gc_bins).cat.categories

    def _bin_stats(grp: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "n_designed": len(grp),
                "n_recovered": int(grp["recovered"].sum()),
                "recovery_rate": float(grp["recovered"].mean()),
                "median_barcodes": float(grp["n_barcodes"].median()),
            }
        )

    bin_result = merged.groupby("gc_bin", observed=True).apply(
        _bin_stats, include_groups=False
    ).reset_index()

    # Add bin centers/edges
    bin_result["gc_bin_lo"] = [float(bin_edges[int(i)].left) for i in bin_result["gc_bin"]]
    bin_result["gc_bin_hi"] = [float(bin_edges[int(i)].right) for i in bin_result["gc_bin"]]
    bin_result["gc_bin_center"] = (bin_result["gc_bin_lo"] + bin_result["gc_bin_hi"]) / 2
    bin_result = bin_result.drop(columns=["gc_bin"])

    warn_msgs = list(asm_warns)
    median_bin_recovery = float(bin_result["recovery_rate"].median())
    dropout_threshold = 0.5 * median_bin_recovery
    bad_bins = bin_result[bin_result["recovery_rate"] < dropout_threshold]

    if not bad_bins.empty:
        for _, row in bad_bins.iterrows():
            warn_msgs.append(
                f"GC bin {row['gc_bin_lo']:.2f}–{row['gc_bin_hi']:.2f}: "
                f"recovery {row['recovery_rate']:.1%} < 50% of median — "
                f"possible synthesis dropout."
            )

    worst_bin = (
        bin_result.loc[bin_result["recovery_rate"].idxmin(), "gc_bin_center"]
        if len(bin_result)
        else None
    )

    summary: dict[str, Any] = {
        "overall_recovery": float(merged["recovered"].mean()),
        "median_bin_recovery": median_bin_recovery,
        "gc_bias_detected": not bad_bins.empty,
        "worst_bin_gc_center": float(worst_bin) if worst_bin is not None else None,
        "warnings": warn_msgs,
        "pass": bad_bins.empty,
    }
    return bin_result, summary


# ---------------------------------------------------------------------------
# Tool 7: oligo_length_qc
# ---------------------------------------------------------------------------


def oligo_length_qc(
    mapping_table_path: str | Path,
    design_manifest_path: str | Path,
) -> tuple[pd.DataFrame, dict]:
    """
    Question answered:
        Are synthesised oligos the correct length, or is there systematic truncation
        detectable from the observed-vs-designed alignment length delta?

    Inputs:
        mapping_table_path   : path to barcode→oligo mapping TSV
        design_manifest_path : path to design manifest (must contain length column)

    Outputs:
        DataFrame — one row per oligo_id:
            oligo_id, designed_length, observed_length_mode,
            fraction_full_length, fraction_truncated_gt10bp, n_barcodes
        summary dict — fraction_oligos_pass, median_fraction_full_length, warnings, pass

    Pass/fail criteria:
        PASS  iff  median fraction_full_length >= 0.80

    CRE-seq-specific notes:
        - CRE-seq uses fixed-length oligo designs (all the same length per pool).
          If designed oligos are NOT all the same length, a warning is raised
          because this is atypical and may indicate a mixed-pool input or a
          lentiMPRA-style variable-length library was loaded by mistake.
        - "Truncation" is defined as observed_length < designed_length − 10 bp.
    """
    params = OligoLengthQcInput(
        mapping_table_path=str(mapping_table_path),
        design_manifest_path=str(design_manifest_path),
    )

    mapping = _load_mapping_table(params.mapping_table_path)
    manifest = _load_design_manifest(params.design_manifest_path)

    if "length" not in manifest.columns or manifest["length"].isna().all():
        raise ValueError(
            "Design manifest must contain a 'length' column for oligo_length_qc."
        )

    asm_warns = _validate_creseq_assumptions(manifest)
    for w in asm_warns:
        warnings.warn(w, CreSeqAssumptionWarning, stacklevel=2)
    warn_msgs = list(asm_warns)

    # Flag non-fixed-length designs
    unique_lengths = manifest["length"].dropna().unique()
    if len(unique_lengths) > 1:
        msg = (
            f"Designed oligos have {len(unique_lengths)} distinct lengths "
            f"({sorted(unique_lengths)[:5]}…). CRE-seq pools are typically "
            f"fixed-length. Confirm assay type — this may be a lentiMPRA or "
            f"mixed-design library."
        )
        warnings.warn(msg, CreSeqAssumptionWarning, stacklevel=2)
        warn_msgs.append(msg)

    # Parse CIGAR for observed oligo lengths
    mapping = _parse_errors_for_df(mapping)

    def _length_stats(grp: pd.DataFrame, designed_length: float) -> pd.Series:
        obs = grp["observed_length"].values
        try:
            mode_val = float(_stat_mode(obs))
        except Exception:
            mode_val = float(np.median(obs))

        n = len(obs)
        full = int((obs == designed_length).sum())
        truncated = int((obs < designed_length - 10).sum())
        return pd.Series(
            {
                "n_barcodes": n,
                "observed_length_mode": mode_val,
                "fraction_full_length": full / n if n else np.nan,
                "fraction_truncated_gt10bp": truncated / n if n else np.nan,
            }
        )

    rows = []
    len_lookup = manifest.set_index("oligo_id")["length"].to_dict()

    for oligo_id, grp in mapping.groupby("oligo_id"):
        dl = len_lookup.get(oligo_id, np.nan)
        stats = _length_stats(grp, dl)
        stats["oligo_id"] = oligo_id
        stats["designed_length"] = dl
        rows.append(stats)

    result = pd.DataFrame(rows)[
        [
            "oligo_id",
            "designed_length",
            "observed_length_mode",
            "fraction_full_length",
            "fraction_truncated_gt10bp",
            "n_barcodes",
        ]
    ]

    median_ffl = float(result["fraction_full_length"].median())
    frac_pass = float((result["fraction_full_length"] >= 0.8).mean())

    if median_ffl < 0.8:
        warn_msgs.append(
            f"Median fraction_full_length {median_ffl:.1%} < 80%. "
            f"Significant synthesis truncation detected."
        )

    summary: dict[str, Any] = {
        "median_fraction_full_length": median_ffl,
        "fraction_oligos_pass": frac_pass,
        "n_unique_designed_lengths": int(len(unique_lengths)),
        "warnings": warn_msgs,
        "pass": median_ffl >= 0.80,
    }
    return result, summary


# ---------------------------------------------------------------------------
# Tool 8: plasmid_depth_summary
# ---------------------------------------------------------------------------


def plasmid_depth_summary(
    plasmid_count_path: str | Path,
) -> tuple[pd.DataFrame, dict]:
    """
    Question answered:
        What is the overall sequencing depth of the plasmid DNA library, and is
        barcode coverage sufficient for reliable RNA/DNA normalisation?

    Inputs:
        plasmid_count_path : path to plasmid DNA count TSV

    Outputs:
        DataFrame — one row per barcode:
            barcode, oligo_id, dna_count, log10_dna_count, replicate (if present)
        summary dict — total_counts, n_barcodes, n_oligos, median_count,
            mean_count, pct_barcodes_zero, pct_barcodes_lt5, warnings, pass

    Pass/fail criteria:
        PASS  iff  median dna_count >= 10  AND  pct_barcodes_zero < 0.10
    """
    params = PlasmidDepthSummaryInput(plasmid_count_path=str(plasmid_count_path))

    df = _load_plasmid_counts(params.plasmid_count_path)

    result = df.copy()
    result["log10_dna_count"] = np.log10(result["dna_count"].clip(lower=1))

    warn_msgs: list[str] = []
    n = len(result)
    n_zero = int((result["dna_count"] == 0).sum())
    n_lt5 = int((result["dna_count"] < 5).sum())
    pct_zero = n_zero / n if n else 0.0
    pct_lt5 = n_lt5 / n if n else 0.0
    median_count = float(result["dna_count"].median())

    if pct_zero >= 0.10:
        warn_msgs.append(
            f"{pct_zero:.1%} of barcodes have zero plasmid counts — "
            f"possible insufficient sequencing depth."
        )

    summary: dict[str, Any] = {
        "total_counts": int(result["dna_count"].sum()),
        "n_barcodes": int(n),
        "n_oligos": int(result["oligo_id"].nunique()),
        "median_count": median_count,
        "mean_count": float(result["dna_count"].mean()),
        "pct_barcodes_zero": float(pct_zero),
        "pct_barcodes_lt5": float(pct_lt5),
        "warnings": warn_msgs,
        "pass": median_count >= 10 and pct_zero < 0.10,
    }
    return result, summary


# ---------------------------------------------------------------------------
# Tool 9: variant_family_coverage
# ---------------------------------------------------------------------------


def variant_family_coverage(
    mapping_table_path: str | Path,
    design_manifest_path: str | Path,
) -> tuple[pd.DataFrame, dict]:
    """
    Question answered:
        For CRE-seq designs that include variant families (a reference element plus
        its motif-knockouts or single-bp mutants), are all family members — especially
        the reference — recovered in the sequenced library?

    Inputs:
        mapping_table_path   : path to barcode→oligo mapping TSV
        design_manifest_path : path to design manifest with parent_element_id column

    Outputs:
        DataFrame — one row per family (keyed by parent_element_id):
            parent_element_id, n_variants_designed, n_variants_recovered,
            reference_recovered (bool), family_complete (bool),
            missing_variants (list, stored as str)
        summary dict — n_families, fraction_families_complete,
            fraction_families_reference_missing, warnings, pass

    Pass/fail criteria:
        PASS  iff  fraction_families_complete >= 0.80
               AND fraction_families_reference_missing == 0

    CRE-seq-specific notes:
        - Losing the reference sequence while retaining its knockouts makes
          delta-score (variant effect) calculation impossible for that family.
          This is a hard failure because there is no analysis-side rescue.
        - parent_element_id is null for top-level reference sequences.  Variants
          point to their reference via parent_element_id = reference oligo_id.
    """
    params = VariantFamilyCoverageInput(
        mapping_table_path=str(mapping_table_path),
        design_manifest_path=str(design_manifest_path),
    )

    mapping = _load_mapping_table(params.mapping_table_path)
    manifest = _load_design_manifest(params.design_manifest_path)

    if (
        "parent_element_id" not in manifest.columns
        or manifest["parent_element_id"].isna().all()
    ):
        logger.info("No parent_element_id links found — no variant families to evaluate.")
        empty = pd.DataFrame(
            columns=[
                "parent_element_id",
                "n_variants_designed",
                "n_variants_recovered",
                "reference_recovered",
                "family_complete",
                "missing_variants",
            ]
        )
        return empty, {
            "n_families": 0,
            "fraction_families_complete": None,
            "fraction_families_reference_missing": None,
            "warnings": ["No variant families found in design manifest."],
            "pass": True,
        }

    recovered_oligos: set[str] = set(mapping["oligo_id"].unique())

    # Variants: rows with non-null parent_element_id
    variants = manifest[manifest["parent_element_id"].notna()].copy()
    family_ids = variants["parent_element_id"].unique()

    rows = []
    for parent_id in family_ids:
        fam_variants = variants[variants["parent_element_id"] == parent_id]
        variant_ids = list(fam_variants["oligo_id"].unique())

        n_designed = len(variant_ids)
        recovered_variants = [v for v in variant_ids if v in recovered_oligos]
        missing = [v for v in variant_ids if v not in recovered_oligos]

        ref_recovered = parent_id in recovered_oligos
        n_recovered = len(recovered_variants)
        family_complete = ref_recovered and len(missing) == 0

        rows.append(
            {
                "parent_element_id": parent_id,
                "n_variants_designed": n_designed,
                "n_variants_recovered": n_recovered,
                "reference_recovered": ref_recovered,
                "family_complete": family_complete,
                "missing_variants": str(missing),
            }
        )

    result = pd.DataFrame(rows)

    n_fam = len(result)
    frac_complete = float(result["family_complete"].mean()) if n_fam else 1.0
    frac_ref_missing = (
        float((~result["reference_recovered"]).mean()) if n_fam else 0.0
    )

    warn_msgs: list[str] = []
    if frac_ref_missing > 0:
        n_ref_missing = int((~result["reference_recovered"]).sum())
        warn_msgs.append(
            f"{n_ref_missing} families have their reference oligo missing — "
            f"variant-effect scores are undefined for these families."
        )
    if frac_complete < 0.8:
        warn_msgs.append(
            f"Only {frac_complete:.1%} of families are fully recovered "
            f"(threshold 80%)."
        )

    summary: dict[str, Any] = {
        "n_families": int(n_fam),
        "fraction_families_complete": float(frac_complete),
        "fraction_families_reference_missing": float(frac_ref_missing),
        "warnings": warn_msgs,
        "pass": frac_complete >= 0.80 and frac_ref_missing == 0.0,
    }
    return result, summary


# ---------------------------------------------------------------------------
# Tool 10: library_summary_report  (meta-tool)
# ---------------------------------------------------------------------------


def library_summary_report(
    mapping_table_path: str | Path,
    plasmid_count_path: str | Path,
    design_manifest_path: str | Path | None = None,
    thresholds_config: dict | None = None,
) -> dict[str, tuple[pd.DataFrame, dict]]:
    """
    Question answered:
        Comprehensive one-shot QC report: runs all applicable library-QC tools and
        returns a structured dict of results with an overall pass/fail.

    Inputs:
        mapping_table_path   : path to barcode→oligo mapping TSV
        plasmid_count_path   : path to plasmid DNA count TSV
        design_manifest_path : optional; if absent, tools requiring the manifest
                               are skipped
        thresholds_config    : optional dict of per-tool threshold overrides
                               (keys: tool name, values: dict of param overrides)

    Outputs:
        dict keyed by tool name →  (DataFrame, summary_dict)
        Plus a top-level key ``"_report"`` with:
            overall_pass, failed_checks, warnings, skipped_tools

    CRE-seq-specific notes:
        - variant_family_coverage, oligo_recovery, gc_content_bias, and oligo_length_qc
          require a design manifest; they are skipped with a logged reason if it is
          not provided.
        - All individual tool summaries are accessible under their tool name key.
    """
    params = LibrarySummaryReportInput(
        mapping_table_path=str(mapping_table_path),
        plasmid_count_path=str(plasmid_count_path),
        design_manifest_path=str(design_manifest_path) if design_manifest_path else None,
        thresholds_config=thresholds_config,
    )

    tc = params.thresholds_config or {}
    has_manifest = params.design_manifest_path is not None

    results: dict[str, tuple[pd.DataFrame, dict]] = {}
    skipped: list[dict[str, str]] = []
    all_warnings: list[str] = []
    failed_checks: list[str] = []

    def _run(name: str, fn, *args, **kwargs):
        try:
            df, summ = fn(*args, **kwargs)
            results[name] = (df, summ)
            if not summ.get("pass", True):
                failed_checks.append(name)
            all_warnings.extend(summ.get("warnings", []))
            logger.info("  [%s] pass=%s", name, summ.get("pass"))
        except Exception as exc:
            logger.error("  [%s] ERROR: %s", name, exc)
            results[name] = (pd.DataFrame(), {"error": str(exc), "pass": False})
            failed_checks.append(name)

    logger.info("library_summary_report: starting CRE-seq library QC")

    _run(
        "barcode_complexity",
        barcode_complexity,
        params.mapping_table_path,
        **tc.get("barcode_complexity", {}),
    )
    _run(
        "barcode_collision_analysis",
        barcode_collision_analysis,
        params.mapping_table_path,
        **tc.get("barcode_collision_analysis", {}),
    )
    _run(
        "synthesis_error_profile",
        synthesis_error_profile,
        params.mapping_table_path,
        params.design_manifest_path,
        **tc.get("synthesis_error_profile", {}),
    )
    _run(
        "plasmid_depth_summary",
        plasmid_depth_summary,
        params.plasmid_count_path,
        **tc.get("plasmid_depth_summary", {}),
    )
    _run(
        "barcode_uniformity",
        barcode_uniformity,
        params.plasmid_count_path,
        **tc.get("barcode_uniformity", {}),
    )

    manifest_tools = [
        ("oligo_recovery", oligo_recovery, [params.mapping_table_path, params.design_manifest_path]),
        ("gc_content_bias", gc_content_bias, [params.mapping_table_path, params.design_manifest_path]),
        ("oligo_length_qc", oligo_length_qc, [params.mapping_table_path, params.design_manifest_path]),
        ("variant_family_coverage", variant_family_coverage, [params.mapping_table_path, params.design_manifest_path]),
    ]

    for name, fn, args in manifest_tools:
        if not has_manifest:
            reason = f"design_manifest_path not provided"
            skipped.append({"tool": name, "reason": reason})
            logger.info("  [%s] SKIPPED — %s", name, reason)
        else:
            _run(name, fn, *args, **tc.get(name, {}))

    overall_pass = len(failed_checks) == 0

    results["_report"] = (
        pd.DataFrame(),
        {
            "overall_pass": overall_pass,
            "failed_checks": failed_checks,
            "warnings": all_warnings,
            "skipped_tools": skipped,
        },
    )

    logger.info(
        "library_summary_report: done. overall_pass=%s, failed=%s",
        overall_pass,
        failed_checks,
    )
    return results


# ---------------------------------------------------------------------------
# FUTURE TOOLS (OTHER ASSAYS):
#
# lentiviral_integration_site_bias(mapping_table_path, genome_bam_path)
#   — Detect hotspot integration in lentiMPRA; not relevant for episomal CRE-seq.
#
# umi_deduplication_efficiency(umi_count_path)
#   — PCR duplicate rate from UMI tags; lentiMPRA protocols often add UMIs,
#     CRE-seq protocols typically do not.
#
# starr_seq_fragment_coverage(bam_path, design_manifest_path)
#   — Assess coverage uniformity along long STARR-seq inserts (500–700 bp);
#     incompatible with CRE-seq's short fixed-length oligo model.
#
# sure_long_insert_qc(mapping_table_path)
#   — Self-Transcribing Active Regulatory Region sequencing uses random shear
#     fragments with variable lengths; oligo_length_qc assumptions do not apply.
# ---------------------------------------------------------------------------
