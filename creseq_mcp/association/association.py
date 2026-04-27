"""
creseq_mcp/processing/association.py
=====================================
Built-in lentiMPRA / CRE-seq association step.

Replaces Nextflow/MPRAflow for the one-time library build step:
  R1 FASTQ + design FASTA → mapping_table.tsv

Protocol note
-------------
In the Ahituv-lab lentiMPRA protocol (and many others), the random barcode
is sequenced as the i5 index read during paired-end Illumina sequencing.
It appears in every FASTQ header after the final colon:

  @instrument:run:flowcell:lane:tile:x:y 1:N:0:BARCODE+i7index

R1 reads the oligo insert (used for alignment).
R2 reads the oligo from the other end (optional, improves alignment).
Neither R1 nor R2 sequence itself contains the barcode.

Pipeline
--------
1. Parse R1 headers  → raw barcode per read
2. STARCODE          → cluster barcodes within edit-distance 1 (error correction)
3. mappy / minimap2  → align R1 sequences to design FASTA → oligo_id per read
4. Join              → (clustered barcode, oligo_id) per read
5. Filter            → min_cov reads AND min_frac mapping to same oligo
6. Write             → mapping_table.tsv, plasmid_counts.tsv, design_manifest.tsv

Outputs
-------
mapping_table.tsv   : barcode, oligo_id, n_reads, cigar, md
plasmid_counts.tsv  : barcode, oligo_id, dna_count
design_manifest.tsv : oligo_id, sequence, designed_category [, variant_family]

CIGAR / MD are placeholder perfect-match strings; synthesis_error_profile
will report trivially perfect quality.  Real per-read CIGAR requires a
full BAM-level aligner (BWA / STAR) which is out of scope here.
"""

from __future__ import annotations

import gzip
import logging
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FASTQ helpers
# ---------------------------------------------------------------------------

def _open_fastq(path: Path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def _iter_fastq(path: Path):
    """Yield (read_name, barcode_i5, sequence) for each read.

    Barcode source (in priority order):
      1. Separate barcode FASTQ passed via fastq_bc (see _load_bc_fastq).
      2. i5 index embedded in R1 header: @name 1:N:0:I5BARCODE+I7INDEX
    """
    with _open_fastq(path) as fh:
        while True:
            header = fh.readline()
            if not header:
                break
            seq  = fh.readline().strip()
            fh.readline()   # +
            fh.readline()   # qual

            name = header.split()[0][1:]     # strip @
            barcode = None
            parts = header.split()
            if len(parts) >= 2:
                idx = parts[1].split(":")[-1]  # last colon-field
                barcode = idx.split("+")[0]    # i5 before +
            yield name, barcode, seq


def _load_bc_fastq(path: Path) -> dict[str, str]:
    """Load a separate barcode FASTQ → {read_name: barcode_sequence}.

    Used when the i5 barcode is delivered as its own index read file
    (ENCODE format) rather than embedded in R1/R2 headers.
    """
    bc_map: dict[str, str] = {}
    with _open_fastq(path) as fh:
        while True:
            header = fh.readline()
            if not header:
                break
            seq  = fh.readline().strip()
            fh.readline()   # +
            fh.readline()   # qual
            name = header.split()[0][1:]
            bc_map[name] = seq
    return bc_map


# ---------------------------------------------------------------------------
# STARCODE barcode clustering
# ---------------------------------------------------------------------------

def _starcode_available() -> bool:
    return shutil.which("starcode") is not None


def _cluster_barcodes(
    barcodes: list[str],
    dist: int = 1,
) -> dict[str, str]:
    """
    Run STARCODE sphere-clustering on barcodes.
    Returns {raw_barcode: centroid_barcode}.
    Falls back to identity mapping if STARCODE is not installed.
    """
    if not _starcode_available():
        logger.warning("starcode not found — skipping barcode clustering")
        return {b: b for b in set(barcodes)}

    bc_input = "\n".join(barcodes) + "\n"
    result = subprocess.run(
        ["starcode", "--dist", str(dist), "--sphere", "--print-clusters"],
        input=bc_input,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.warning("starcode failed (code %d) — skipping clustering", result.returncode)
        return {b: b for b in set(barcodes)}

    raw_to_centroid: dict[str, str] = {}
    for line in result.stdout.strip().splitlines():
        if not line:
            continue
        parts = line.split("\t")
        centroid = parts[0]
        raw_to_centroid[centroid] = centroid
        if len(parts) >= 3:
            for member in parts[2].split(","):
                raw_to_centroid[member] = centroid

    # any barcode not seen in output (shouldn't happen) → identity
    for b in set(barcodes):
        if b not in raw_to_centroid:
            raw_to_centroid[b] = b

    return raw_to_centroid


# ---------------------------------------------------------------------------
# mappy (minimap2) alignment
# ---------------------------------------------------------------------------

def _align_reads(
    sequences: list[tuple[str, str]],  # [(read_name, seq), ...]
    design_fasta: Path,
    mapq_threshold: int,
) -> dict[str, str]:
    """
    Align sequences to design FASTA using minimap2 (mappy).
    Returns {read_name: oligo_id} for reads that pass mapq_threshold.
    """
    try:
        import mappy as mp
    except ImportError:
        raise RuntimeError(
            "mappy is required for built-in association. "
            "Install with: conda install -c conda-forge mappy"
        )

    aligner = mp.Aligner(str(design_fasta), preset="sr", best_n=1)
    if not aligner:
        raise RuntimeError(f"Failed to build mappy index from {design_fasta}")

    assignments: dict[str, str] = {}
    for name, seq in sequences:
        hits = list(aligner.map(seq))
        if not hits:
            continue
        best = hits[0]
        if best.mapq >= mapq_threshold:
            assignments[name] = best.ctg   # contig name = oligo_id from FASTA

    return assignments


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

def _filter_assignments(
    read_assignments: list[tuple[str, str]],  # [(clustered_barcode, oligo_id), ...]
    min_cov: int,
    min_frac: float,
) -> pd.DataFrame:
    """
    Apply coverage and fraction filters.

    For each barcode:
      - total reads across all oligos ≥ min_cov
      - fraction mapping to the top oligo ≥ min_frac

    Returns DataFrame: barcode, oligo_id, n_reads
    """
    # Count (barcode, oligo_id) pairs
    pair_counts: Counter = Counter(read_assignments)

    # Group by barcode
    bc_to_oligos: dict[str, Counter] = defaultdict(Counter)
    for (bc, oid), count in pair_counts.items():
        bc_to_oligos[bc][oid] += count

    rows = []
    for bc, oligo_counts in bc_to_oligos.items():
        total = sum(oligo_counts.values())
        top_oligo, top_count = oligo_counts.most_common(1)[0]
        if total >= min_cov and (top_count / total) >= min_frac:
            rows.append({"barcode": bc, "oligo_id": top_oligo, "n_reads": total})

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["barcode", "oligo_id", "n_reads"]
    )


# ---------------------------------------------------------------------------
# Design manifest builder
# ---------------------------------------------------------------------------

def _open_fasta(path: Path):
    """Open a FASTA file, handling gzip regardless of extension."""
    with open(path, "rb") as f:
        magic = f.read(2)
    return gzip.open(path, "rt") if magic == b"\x1f\x8b" else open(path)


def _build_design_manifest(design_fasta: Path, labels_path: Path | None) -> pd.DataFrame:
    """
    Build design_manifest from design FASTA (sequences) + labels TSV (categories).
    """
    # Parse FASTA
    records = []
    current_id = current_seq = None
    with _open_fasta(design_fasta) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if current_id:
                    records.append({"oligo_id": current_id, "sequence": current_seq})
                current_id = line[1:].split()[0]
                current_seq = ""
            elif current_id:
                current_seq += line
    if current_id:
        records.append({"oligo_id": current_id, "sequence": current_seq})

    manifest = pd.DataFrame(records)

    if labels_path and Path(labels_path).exists():
        labels = pd.read_csv(labels_path, sep="\t")
        needed = {"oligo_id", "designed_category"}
        if needed.issubset(labels.columns):
            manifest = manifest.merge(
                labels[list(needed) + [c for c in labels.columns
                                       if c not in needed and c != "sequence"]],
                on="oligo_id", how="left",
            )
            manifest["designed_category"] = manifest["designed_category"].fillna("other")
        else:
            logger.warning("labels TSV missing required columns %s", needed - set(labels.columns))
            manifest["designed_category"] = "other"
    else:
        manifest["designed_category"] = "other"

    return manifest


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_association(
    fastq_r1: Path,
    design_fasta: Path,
    outdir: Path,
    *,
    fastq_r2: Path | None = None,
    fastq_bc: Path | None = None,
    labels_path: Path | None = None,
    min_cov: int = 3,
    min_frac: float = 0.5,
    mapq_threshold: int = 20,
    starcode_dist: int = 1,
) -> dict:
    """
    Run the built-in association step.

    Parameters
    ----------
    fastq_r1        : R1 FASTQ (oligo reads)
    design_fasta    : FASTA of all designed oligo sequences
    outdir          : Directory to write output TSVs
    fastq_r2        : R2 FASTQ (optional; paired oligo reads for better alignment)
    fastq_bc        : Separate barcode index FASTQ (ENCODE format: i5 read as its
                      own file). When provided, barcodes are taken from here instead
                      of the R1 header. Read names must match R1.
    labels_path     : TSV with oligo_id + designed_category columns
    min_cov         : Minimum reads per barcode-oligo pair (default 3)
    min_frac        : Minimum fraction mapping to same oligo (default 0.5)
    mapq_threshold  : Minimum minimap2 mapping quality (default 20)
    starcode_dist   : STARCODE edit-distance for clustering (default 1)

    Returns
    -------
    dict with summary statistics and output file paths.
    Writes mapping_table.tsv, plasmid_counts.tsv, design_manifest.tsv to outdir.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info("Association: parsing R1 sequences")
    # Load separate barcode FASTQ first if provided (ENCODE format)
    bc_from_file: dict[str, str] = {}
    if fastq_bc is not None:
        logger.info("Association: loading barcodes from separate index FASTQ %s", fastq_bc)
        bc_from_file = _load_bc_fastq(Path(fastq_bc))
        logger.info("  %d barcodes loaded from index FASTQ", len(bc_from_file))

    names, raw_barcodes, sequences = [], [], []
    for name, header_bc, seq in _iter_fastq(fastq_r1):
        # Prefer explicit barcode file; fall back to header-embedded i5
        bc = bc_from_file.get(name) if bc_from_file else header_bc
        if bc:
            names.append(name)
            raw_barcodes.append(bc)
            sequences.append(seq)

    n_reads = len(names)
    source = "index FASTQ" if bc_from_file else "R1 header"
    logger.info("  %d reads with i5 barcode (source: %s)", n_reads, source)

    # STARCODE clustering
    logger.info("Association: clustering barcodes with STARCODE (dist=%d)", starcode_dist)
    raw_to_centroid = _cluster_barcodes(raw_barcodes, dist=starcode_dist)
    clustered = [raw_to_centroid.get(b, b) for b in raw_barcodes]
    n_unique_raw = len(set(raw_barcodes))
    n_unique_clustered = len(set(clustered))
    logger.info("  %d → %d unique barcodes after clustering", n_unique_raw, n_unique_clustered)

    # Align R1 (and optionally merge with R2) to design FASTA
    logger.info("Association: aligning reads to design FASTA")
    read_seqs = list(zip(names, sequences))

    if fastq_r2 is not None:
        logger.info("  merging R2 reads for paired-end alignment")
        r2_seqs = {name: seq for name, _, seq in _iter_fastq(fastq_r2)}
        # For each R1 read, also submit R2 with a modified name
        pe_seqs = []
        for name, seq in read_seqs:
            pe_seqs.append((name, seq))
            if name in r2_seqs:
                pe_seqs.append((name + "/2", r2_seqs[name]))
        assignments_raw = _align_reads(pe_seqs, design_fasta, mapq_threshold)
        # Prefer R1 alignment; use R2 only if R1 didn't align
        assignments: dict[str, str] = {}
        for name in names:
            if name in assignments_raw:
                assignments[name] = assignments_raw[name]
            elif name + "/2" in assignments_raw:
                assignments[name] = assignments_raw[name + "/2"]
    else:
        assignments = _align_reads(read_seqs, design_fasta, mapq_threshold)

    n_aligned = len(assignments)
    logger.info("  %d/%d reads aligned (mapq≥%d)", n_aligned, n_reads, mapq_threshold)

    # Join barcode + oligo_id
    paired = [
        (clustered[i], assignments[names[i]])
        for i in range(n_reads)
        if names[i] in assignments
    ]

    # Filter
    logger.info("Association: filtering (min_cov=%d, min_frac=%.2f)", min_cov, min_frac)
    mapping_df = _filter_assignments(paired, min_cov=min_cov, min_frac=min_frac)
    n_barcodes = len(mapping_df)
    n_oligos = mapping_df["oligo_id"].nunique() if n_barcodes else 0
    logger.info("  %d barcodes → %d oligos after filtering", n_barcodes, n_oligos)

    # Add placeholder CIGAR / MD
    _bc_len_str = mapping_df["barcode"].str.len().astype(str)
    mapping_df["cigar"] = _bc_len_str + "M"
    mapping_df["md"] = _bc_len_str

    # Design manifest
    manifest = _build_design_manifest(design_fasta, labels_path)

    # plasmid_counts: use n_reads from mapping as dna_count placeholder
    plasmid_counts = mapping_df[["barcode", "oligo_id"]].copy()
    plasmid_counts["dna_count"] = mapping_df["n_reads"]

    # Write outputs
    mapping_df[["barcode", "oligo_id", "n_reads", "cigar", "md"]].to_csv(
        outdir / "mapping_table.tsv", sep="\t", index=False
    )
    plasmid_counts.to_csv(outdir / "plasmid_counts.tsv", sep="\t", index=False)
    manifest.to_csv(outdir / "design_manifest.tsv", sep="\t", index=False)

    return {
        "n_reads_total": n_reads,
        "n_reads_aligned": n_aligned,
        "pct_aligned": round(n_aligned / n_reads * 100, 1) if n_reads else 0,
        "n_barcodes_raw": n_unique_raw,
        "n_barcodes_clustered": n_unique_clustered,
        "n_barcodes_passing_filter": n_barcodes,
        "n_oligos_covered": n_oligos,
        "n_oligos_in_design": len(manifest),
        "warnings": [] if n_barcodes > 0 else ["No barcodes passed filter — check input files"],
        "pass": n_barcodes > 0,
    }
