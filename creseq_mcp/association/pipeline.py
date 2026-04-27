"""
creseq_mcp/processing/pipeline.py
==================================
Raw-data processing: FASTQ + barcode reference library → QC-ready TSVs.

Input
-----
- Plasmid-DNA FASTQ (gzipped or plain)
- Barcode reference library TSV with columns:
    oligo_id, barcode, sequence, designed_category
    optional: variant_family

Read structure assumed
---------------------
  3-prime barcode (default):  [oligo_sequence ...][barcode]
  5-prime barcode:             [barcode][oligo_sequence ...]

Output (all written to UPLOAD_DIR by process_and_save)
------
- mapping_table.tsv  — barcode, oligo_id, cigar, md, n_reads
- plasmid_counts.tsv — barcode, oligo_id, dna_count
- design_manifest.tsv — oligo_id, sequence, designed_category[, variant_family]
"""

from __future__ import annotations

import gzip
import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import pandas as pd

logger = logging.getLogger(__name__)

REFERENCE_REQUIRED_COLS = {"oligo_id", "barcode", "sequence", "designed_category"}


# ---------------------------------------------------------------------------
# FASTQ parsing
# ---------------------------------------------------------------------------


def _open_fastq(path: str | Path):
    path = Path(path)
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path, "r")


def _parse_fastq(path: str | Path) -> Iterator[tuple[str, str, str]]:
    """Yield (header, sequence, quality) for each read."""
    with _open_fastq(path) as fh:
        while True:
            header = fh.readline().strip()
            if not header:
                break
            seq = fh.readline().strip()
            fh.readline()  # '+'
            qual = fh.readline().strip()
            yield header[1:], seq, qual


# ---------------------------------------------------------------------------
# Barcode matching
# ---------------------------------------------------------------------------


def _hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def _match_barcode(
    bc: str,
    index: dict[str, tuple[str, str]],
    max_mismatch: int,
) -> str | None:
    """Return the reference barcode key if bc matches within max_mismatch, else None."""
    if bc in index:
        return bc
    if max_mismatch > 0:
        for ref_bc in index:
            if len(ref_bc) == len(bc) and _hamming(bc, ref_bc) <= max_mismatch:
                return ref_bc
    return None


# ---------------------------------------------------------------------------
# CIGAR / MD generation
# ---------------------------------------------------------------------------


def _make_cigar_md(obs: str, ref: str) -> tuple[str, str]:
    """
    Build a CIGAR string and MD tag from observed vs designed sequence.

    - Aligned region (min length): all M, mismatches captured in MD.
    - Length difference: soft-clipped (S) appended to CIGAR.
    """
    n = min(len(obs), len(ref))
    clip = abs(len(obs) - len(ref))

    md_parts: list[str] = []
    run_match = 0

    for o, r in zip(obs[:n], ref[:n]):
        if o == r:
            run_match += 1
        else:
            if run_match:
                md_parts.append(str(run_match))
                run_match = 0
            md_parts.append(r)  # reference base at mismatch position

    if run_match:
        md_parts.append(str(run_match))

    cigar = f"{n}M" + (f"{clip}S" if clip else "")
    md = "".join(md_parts) if md_parts else str(n)
    return cigar, md


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def process_library(
    fastq_path: str | Path,
    reference_path: str | Path,
    *,
    barcode_len: int = 10,
    barcode_end: str = "3prime",
    max_mismatch: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Process a CRE-seq plasmid-DNA FASTQ against a barcode reference library.

    Parameters
    ----------
    fastq_path      : Path to plasmid-DNA FASTQ (.fastq or .fastq.gz)
    reference_path  : Path to barcode reference library TSV
    barcode_len     : Length of barcode (default 10)
    barcode_end     : Where the barcode sits — "3prime" (default) or "5prime"
    max_mismatch    : Allowed mismatches for barcode matching (default 1)

    Returns
    -------
    (mapping_table, plasmid_counts, design_manifest)
    """
    ref_df = pd.read_csv(reference_path, sep="\t")
    missing = REFERENCE_REQUIRED_COLS - set(ref_df.columns)
    if missing:
        raise ValueError(f"Reference library missing required columns: {missing}")

    # barcode -> (oligo_id, designed_sequence)
    barcode_index: dict[str, tuple[str, str]] = {
        str(row.barcode): (str(row.oligo_id), str(row.sequence))
        for row in ref_df.itertuples(index=False)
    }

    read_counts: dict[str, int] = defaultdict(int)
    n_total = 0
    n_unmapped = 0

    for _, seq, _ in _parse_fastq(fastq_path):
        n_total += 1
        if barcode_end == "3prime":
            bc_obs = seq[-barcode_len:]
            oligo_obs = seq[:-barcode_len]
        else:
            bc_obs = seq[:barcode_len]
            oligo_obs = seq[barcode_len:]

        matched = _match_barcode(bc_obs, barcode_index, max_mismatch)
        if matched:
            read_counts[(matched, oligo_obs)] += 1
        else:
            n_unmapped += 1

    logger.info(
        "Processed %d reads: %d mapped (%.1f%%), %d unmapped",
        n_total,
        n_total - n_unmapped,
        100 * (n_total - n_unmapped) / max(n_total, 1),
        n_unmapped,
    )

    # Build mapping table — one row per unique (barcode, oligo_obs) combination
    rows = []
    for (bc, oligo_obs), n_reads in read_counts.items():
        oligo_id, designed_seq = barcode_index[bc]
        cigar, md = _make_cigar_md(oligo_obs, designed_seq)
        rows.append({
            "barcode": bc,
            "oligo_id": oligo_id,
            "cigar": cigar,
            "md": md,
            "n_reads": n_reads,
        })

    mapping_table = (
        pd.DataFrame(rows)
        if rows
        else pd.DataFrame(columns=["barcode", "oligo_id", "cigar", "md", "n_reads"])
    )

    # Plasmid counts — aggregate n_reads per barcode
    plasmid_counts = (
        mapping_table.groupby(["barcode", "oligo_id"], as_index=False)["n_reads"]
        .sum()
        .rename(columns={"n_reads": "dna_count"})
    )

    # Design manifest — straight from reference
    manifest_cols = ["oligo_id", "sequence", "designed_category"]
    if "variant_family" in ref_df.columns:
        manifest_cols.append("variant_family")
    design_manifest = ref_df[manifest_cols].drop_duplicates("oligo_id")

    return mapping_table, plasmid_counts, design_manifest


def process_and_save(
    fastq_path: str | Path,
    reference_path: str | Path,
    upload_dir: Path,
    **kwargs,
) -> dict:
    """Run process_library and write all three TSVs to upload_dir."""
    mapping_table, plasmid_counts, design_manifest = process_library(
        fastq_path, reference_path, **kwargs
    )

    mapping_table.to_csv(upload_dir / "mapping_table.tsv", sep="\t", index=False)
    plasmid_counts.to_csv(upload_dir / "plasmid_counts.tsv", sep="\t", index=False)
    design_manifest.to_csv(upload_dir / "design_manifest.tsv", sep="\t", index=False)

    return {
        "total_reads": int(mapping_table["n_reads"].sum()),
        "unique_barcodes": len(plasmid_counts),
        "oligos_represented": mapping_table["oligo_id"].nunique(),
        "oligos_in_reference": len(design_manifest),
    }
