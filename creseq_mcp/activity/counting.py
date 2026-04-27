"""
creseq_mcp/processing/counting.py
===================================
Count barcode occurrences in DNA/RNA FASTQs using the mapping table from ASSOCIATION.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd

from creseq_mcp.association.pipeline import _hamming, _parse_fastq

logger = logging.getLogger(__name__)


def _count_fastq(
    fastq_path: Path,
    known_barcodes: set[str],
    barcode_len: int,
    barcode_end: str,
    max_mismatch: int = 0,
) -> tuple[dict[str, int], int, int]:
    counts: dict[str, int] = defaultdict(int)
    n_total = n_matched = 0

    for _, seq, _ in _parse_fastq(fastq_path):
        n_total += 1
        bc = seq[-barcode_len:] if barcode_end == "3prime" else seq[:barcode_len]
        if bc in known_barcodes:
            counts[bc] += 1
            n_matched += 1
        elif max_mismatch > 0:
            for ref in known_barcodes:
                if len(ref) == len(bc) and _hamming(bc, ref) <= max_mismatch:
                    counts[ref] += 1
                    n_matched += 1
                    break

    logger.info(
        "%s: %d/%d reads matched (%.1f%%)",
        fastq_path.name,
        n_matched,
        n_total,
        100 * n_matched / max(n_total, 1),
    )
    return dict(counts), n_total, n_matched


def process_dna_counting(
    fastq_path: str | Path,
    mapping_table_path: str | Path,
    upload_dir: Path,
    *,
    barcode_len: int = 20,
    barcode_end: str = "3prime",
    max_mismatch: int = 0,
) -> dict:
    """Count DNA barcodes → overwrite plasmid_counts.tsv with real counts."""
    mapping = pd.read_csv(mapping_table_path, sep="\t")
    known = set(mapping["barcode"].unique())

    raw, n_total, n_matched = _count_fastq(
        Path(fastq_path), known, barcode_len, barcode_end, max_mismatch
    )

    plasmid = mapping[["barcode", "oligo_id"]].drop_duplicates().copy()
    plasmid["dna_count"] = plasmid["barcode"].map(raw).fillna(0).astype(int)
    plasmid.to_csv(upload_dir / "plasmid_counts.tsv", sep="\t", index=False)

    return {
        "total_reads": n_total,
        "matched_reads": n_matched,
        "match_rate": round(n_matched / max(n_total, 1), 4),
        "barcodes_with_counts": int((plasmid["dna_count"] > 0).sum()),
        "median_dna_count": float(plasmid["dna_count"].median()),
    }


def process_rna_counting(
    fastq_paths: list[str | Path],
    mapping_table_path: str | Path,
    upload_dir: Path,
    *,
    rep_names: list[str] | None = None,
    barcode_len: int = 20,
    barcode_end: str = "3prime",
    max_mismatch: int = 0,
) -> dict:
    """Count RNA barcodes across replicates → write rna_counts.tsv."""
    mapping = pd.read_csv(mapping_table_path, sep="\t")
    known = set(mapping["barcode"].unique())

    if rep_names is None:
        rep_names = [f"rep{i + 1}" for i in range(len(fastq_paths))]

    base = mapping[["barcode", "oligo_id"]].drop_duplicates().copy()

    per_rep = []
    for fastq_path, rep in zip(fastq_paths, rep_names):
        raw, n_total, n_matched = _count_fastq(
            Path(fastq_path), known, barcode_len, barcode_end, max_mismatch
        )
        base[f"rna_count_{rep}"] = base["barcode"].map(raw).fillna(0).astype(int)
        per_rep.append({"rep": rep, "total_reads": n_total, "matched_reads": n_matched})

    base.to_csv(upload_dir / "rna_counts.tsv", sep="\t", index=False)

    rep_cols = [f"rna_count_{r}" for r in rep_names]
    return {
        "replicates": rep_names,
        "total_barcodes": len(base),
        "median_rna_count": float(base[rep_cols].median().median()),
        "per_replicate": per_rep,
    }
