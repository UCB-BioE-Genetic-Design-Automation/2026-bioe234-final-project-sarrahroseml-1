"""
creseq_mcp/processing/mpraflow.py
==================================
Wrapper around the MPRAflow ASSOCIATION workflow (shendurelab/MPRAflow).

MPRAflow aligns paired-end reads to the design FASTA with BWA, then links
barcodes to oligos and filters out low-coverage / ambiguous barcodes.

Requires: nextflow + conda (MPRAflow manages its own bioinformatics deps).

Note on the kircherlab fork
---------------------------
The original kircherlab/MPRAflow (Nextflow) repo was removed by the Kircher
lab, who now maintain kircherlab/MPRAsnakeflow (Snakemake).  This wrapper
targets the Shendure-lab Nextflow fork (shendurelab/MPRAflow), which is the
last maintained Nextflow version.

Parameter mapping vs. kircherlab version:
  kircherlab --fastq_oligo  →  shendurelab --fastq-insert  (R1, oligo reads)
  kircherlab --fastq_bc     →  shendurelab --fastq-bc       (R2, barcode reads)
  kircherlab -entry ASSOCIATION  →  shendurelab -main-script association.nf

Output format
-------------
The kircherlab version wrote an assigned_counts.tsv.
The shendurelab version writes a pickle:
  {name}_filtered_coords_to_barcodes.pickle
  → dict  {oligo_id: set(barcode, ...)}

convert_to_qc_format() inverts this to produce mapping_table.tsv,
plasmid_counts.tsv, and design_manifest.tsv in the standard format.
n_reads is set to min_cov for all barcodes (they all passed the coverage
filter, but per-barcode read counts are not in the filtered pickle).
"""

from __future__ import annotations

import logging
import pickle
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from creseq_mcp.processing.pipeline import _make_cigar_md

logger = logging.getLogger(__name__)


def _nextflow_bin() -> str | None:
    """Return path to nextflow binary, checking ~/bin if not on PATH."""
    found = shutil.which("nextflow")
    if found:
        return found
    local = Path.home() / "bin" / "nextflow"
    return str(local) if local.exists() else None


def is_available() -> bool:
    return _nextflow_bin() is not None


def run_mpraflow(
    fastq_bc: Path,
    fastq_oligo: Path,
    design_fasta: Path,
    outdir: Path,
    *,
    name: str = "library",
    profile: str = "conda",
    fastq_oligo_pe: Path | None = None,
    min_cov: int = 3,
    min_frac: float = 0.5,
) -> Path:
    """
    Run shendurelab/MPRAflow ASSOCIATION workflow.

    Parameters
    ----------
    fastq_bc       : FASTQ with barcode reads (R2)
    fastq_oligo    : FASTQ with oligo reads (R1, --fastq-insert)
    design_fasta   : FASTA of designed oligo sequences
    outdir         : Directory for MPRAflow output
    name           : Library name (used in output filenames)
    profile        : Nextflow profile — "conda" (default) or "docker"
    fastq_oligo_pe : Optional R2 oligo FASTQ for paired-end insert alignment
    min_cov        : Minimum barcode coverage to keep (default 3)
    min_frac       : Minimum fraction mapping to same oligo (default 0.5)

    Returns
    -------
    Path to the filtered_coords_to_barcodes pickle produced by MPRAflow.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        _nextflow_bin(), "run", "shendurelab/MPRAflow",
        "-main-script", "association.nf",
        "--name", name,
        "--fastq-insert", str(fastq_oligo),
        "--fastq-bc", str(fastq_bc),
        "--design", str(design_fasta),
        "--outdir", str(outdir),
        "--min-cov", str(min_cov),
        "--min-frac", str(min_frac),
        "-profile", profile,
    ]

    if fastq_oligo_pe is not None:
        cmd += ["--fastq-insertPE", str(fastq_oligo_pe)]

    logger.info("Running MPRAflow: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=outdir)

    if result.returncode != 0:
        raise RuntimeError(
            f"MPRAflow exited with code {result.returncode}.\n"
            f"stderr:\n{result.stderr[-3000:]}"
        )

    # Output: {outdir}/{name}/{name}_filtered_coords_to_barcodes.pickle
    candidates = list(outdir.glob(f"**/{name}_filtered_coords_to_barcodes.pickle"))
    if not candidates:
        raise FileNotFoundError(
            f"MPRAflow pickle output not found under {outdir}. "
            "Check the MPRAflow log for errors."
        )

    return candidates[0]


def _pickle_to_dataframe(pickle_path: Path, min_cov: int = 3) -> pd.DataFrame:
    """
    Convert shendurelab MPRAflow pickle to a flat DataFrame.

    The pickle is {oligo_id: set(barcode, ...)} — every barcode in the set
    passed the min_cov and min_frac filters.  n_reads is set to min_cov as a
    conservative placeholder (the actual per-barcode counts from the
    association run are not stored in the filtered pickle).

    Returns DataFrame with columns: barcode, oligo_id, n_reads, cigar, md
    """
    with open(pickle_path, "rb") as fh:
        coords_to_barcodes: dict = pickle.load(fh)

    rows = []
    for oligo_id, barcodes in coords_to_barcodes.items():
        for bc in barcodes:
            rows.append({"barcode": bc, "oligo_id": str(oligo_id)})

    if not rows:
        return pd.DataFrame(columns=["barcode", "oligo_id", "n_reads", "cigar", "md"])

    df = pd.DataFrame(rows)
    df["n_reads"] = min_cov
    df["cigar"] = df["barcode"].apply(lambda bc: f"{len(bc)}M")
    df["md"] = df["barcode"].apply(lambda bc: str(len(bc)))
    return df


def convert_to_qc_format(
    pickle_path: Path,
    reference_path: Path,
    upload_dir: Path,
    min_cov: int = 3,
) -> dict:
    """
    Convert MPRAflow filtered pickle → mapping_table, plasmid_counts,
    design_manifest TSVs in upload_dir.

    Note: CIGAR/MD are placeholder perfect-match strings; synthesis_error_profile
    and oligo_length_qc will report trivially perfect synthesis quality.
    n_reads = min_cov for all barcodes (conservative lower bound).
    """
    mapping_table = _pickle_to_dataframe(pickle_path, min_cov=min_cov)

    plasmid_counts = mapping_table[["barcode", "oligo_id"]].copy()
    plasmid_counts["dna_count"] = mapping_table["n_reads"]

    ref_df = pd.read_csv(reference_path, sep="\t")
    manifest_cols = [c for c in ["oligo_id", "sequence", "designed_category", "variant_family"]
                     if c in ref_df.columns]
    design_manifest = ref_df[manifest_cols].drop_duplicates("oligo_id")

    mapping_table.to_csv(upload_dir / "mapping_table.tsv", sep="\t", index=False)
    plasmid_counts.to_csv(upload_dir / "plasmid_counts.tsv", sep="\t", index=False)
    design_manifest.to_csv(upload_dir / "design_manifest.tsv", sep="\t", index=False)

    return {
        "total_barcodes": len(mapping_table),
        "unique_barcodes": len(plasmid_counts),
        "oligos_represented": mapping_table["oligo_id"].nunique(),
        "oligos_in_reference": len(design_manifest),
    }


def process_and_save(
    fastq_bc: Path,
    fastq_oligo: Path,
    design_fasta: Path,
    reference_path: Path,
    upload_dir: Path,
    *,
    name: str = "library",
    profile: str = "conda",
    fastq_oligo_pe: Path | None = None,
    min_cov: int = 3,
    min_frac: float = 0.5,
) -> dict:
    """Run MPRAflow then convert output to QC-ready TSVs."""
    mpraflow_out = upload_dir / "mpraflow_out"
    pickle_path = run_mpraflow(
        fastq_bc, fastq_oligo, design_fasta, mpraflow_out,
        name=name, profile=profile,
        fastq_oligo_pe=fastq_oligo_pe,
        min_cov=min_cov, min_frac=min_frac,
    )
    return convert_to_qc_format(pickle_path, reference_path, upload_dir, min_cov=min_cov)
