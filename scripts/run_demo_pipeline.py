"""
End-to-end demo: generator → normalize → call_active → extract_sequences →
motif_enrichment → plots.

All intermediate and final outputs are written to ``demo_outputs/`` at the
project root.  The pipeline reads the generator's output (per-barcode counts +
per-oligo design manifest), bridges the schema joints between the per-barcode
tables and the per-element analysis tools, and stitches the modules together.

Schema joints handled here:
  - per-barcode plasmid + RNA counts → per-element counts (sum across barcodes,
    sum across replicates) for ``normalize_activity``;
  - ``log2_activity`` → ``mean_activity`` rename, plus per-element std + barcode
    counts pulled from the underlying per-barcode log2 values, so
    ``call_active_elements_empirical`` gets a complete per-element table;
  - manifest column ``oligo_id`` → ``element_id`` for ``extract_sequences``.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEMO_DIR = ROOT / "demo_outputs"
DEMO_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR = Path.home() / ".creseq" / "uploads"


def _hr(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ─── Step 1: Generator ───────────────────────────────────────────────────────

_hr("STEP 1 · Run generator")
import subprocess

gen_proc = subprocess.run(
    [sys.executable, str(ROOT / "scripts" / "generate_test_data.py")],
    capture_output=True, text=True, check=True,
)
print(gen_proc.stdout.strip().splitlines()[-1])
print("Generator outputs in:", UPLOAD_DIR)
for f in sorted(UPLOAD_DIR.glob("*.tsv")):
    print(f"  {f.name:25s}  {f.stat().st_size:>9,} bytes")


# ─── Steps 2+3: normalize + activity calling ─────────────────────────────────

_hr("STEP 2+3 · normalize + activity_report")
from creseq_mcp.activity.normalize import activity_report

manifest = pd.read_csv(UPLOAD_DIR / "design_manifest.tsv", sep="\t")

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    results_df, act_summary = activity_report(
        dna_counts_path=UPLOAD_DIR / "plasmid_counts.tsv",
        rna_counts_path=UPLOAD_DIR / "rna_counts.tsv",
        design_manifest_path=UPLOAD_DIR / "design_manifest.tsv",
        upload_dir=DEMO_DIR,
    )

classified_path = str(DEMO_DIR / "activity_results.tsv")
print(json.dumps(
    {k: v for k, v in act_summary.items() if k not in ("warnings", "replicates")},
    indent=2, default=str,
))
if act_summary.get("warnings"):
    print("Warnings:", act_summary["warnings"])
for w in caught:
    print(f"  [pyWarning] {w.category.__name__}: {w.message}")

neg_ids = manifest.loc[
    manifest["designed_category"] == "negative_control", "oligo_id"
].astype(str).tolist()
print(f"Negative controls: {len(neg_ids)} IDs (e.g. {neg_ids[:3]}...)")


# ─── Step 4: extract_sequences_to_fasta ──────────────────────────────────────

_hr("STEP 4 · extract_sequences_to_fasta")
from creseq_mcp.motifs.enrichment import extract_sequences_to_fasta

extract_result = extract_sequences_to_fasta(
    classified_table=classified_path,
    sequence_source=str(UPLOAD_DIR / "design_manifest.tsv"),
    active_output=str(DEMO_DIR / "active.fa"),
    background_output=str(DEMO_DIR / "background.fa"),
)
print(json.dumps(extract_result, indent=2))


# ─── Step 5: motif_enrichment (real JASPAR — ~30s) ───────────────────────────

_hr("STEP 5 · motif_enrichment (JASPAR2024 CORE Vertebrates)")
from creseq_mcp.motifs.enrichment import motif_enrichment

enrich_result = motif_enrichment(
    active_fasta=extract_result["active_fasta"],
    background_fasta=extract_result["background_fasta"],
    motif_database="JASPAR2024",
    score_threshold=0.85,
    output_path=str(DEMO_DIR / "motif_enrichment.tsv"),
)
print(f"Enrichment table → {Path(enrich_result['enrichment_table']).name}")
print(enrich_result["summary"])

# Quick inspection of the top of the table
enr_df = pd.read_csv(enrich_result["enrichment_table"], sep="\t")
print(f"\nMotifs evaluated: {len(enr_df)} | FDR<0.05: {(enr_df['fdr']<0.05).sum()}")
print(enr_df.head(8).to_string(index=False))


# ─── Step 6: plot_creseq ─────────────────────────────────────────────────────

_hr("STEP 6 · plot_creseq")
from creseq_mcp.plots.plots import plot_creseq

volcano = plot_creseq(
    data_file=classified_path,
    plot_type="volcano",
    output_path=str(DEMO_DIR / "volcano.png"),
    neg_control_ids=neg_ids,
)
print(f"Volcano → {Path(volcano['plot_path']).name}")
print("  ", volcano["description"])

dotplot = plot_creseq(
    data_file=enrich_result["enrichment_table"],
    plot_type="motif_dotplot",
    output_path=str(DEMO_DIR / "motif_dotplot.png"),
)
print(f"Motif dot plot → {Path(dotplot['plot_path']).name}")
print("  ", dotplot["description"])


# ─── Wrap up ─────────────────────────────────────────────────────────────────

_hr("DONE · contents of demo_outputs/")
for f in sorted(DEMO_DIR.iterdir()):
    print(f"  {f.name:35s}  {f.stat().st_size:>9,} bytes")
