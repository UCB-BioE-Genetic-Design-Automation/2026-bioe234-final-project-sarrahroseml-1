"""
creseq_mcp/variants/delta_scores.py
=====================================
Variant effect scoring for lentiMPRA / CRE-seq.

For each variant family, computes delta = mutant_log2_ratio - reference_log2_ratio,
then tests significance via z-test across all deltas (BH FDR).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm
from statsmodels.stats.multitest import multipletests

_LOCUS_RE = re.compile(r"\[([^\]]+)\]")


def _add_variant_cols(manifest: pd.DataFrame) -> pd.DataFrame:
    """
    Derive variant_family and is_reference from oligo_id when those columns
    are absent.  Handles the lentiMPRA naming convention:
      R:<TF>_<coords>_[<locus>]  →  reference allele, family = <locus>
      A:<TF>_<coords>_[<locus>]  →  alternate allele, family = <locus>
      C:<TF>_<coords>_[<locus>]  →  control allele,   family = <locus>
      seq#####                   →  no variant family (NaN)
    Falls back to designed_category == "reference" when prefix is ambiguous.
    """
    manifest = manifest.copy()

    def _family(oid: str) -> str | None:
        m = _LOCUS_RE.search(oid)
        return m.group(1) if m else None

    def _is_ref(row) -> bool | None:
        oid = str(row.get("oligo_id", ""))
        if oid.startswith("R:"):
            return True
        if oid.startswith("A:") or oid.startswith("C:"):
            return False
        cat = str(row.get("designed_category", ""))
        if cat == "reference":
            return True
        if cat in ("alternate", "control"):
            return False
        return None

    manifest["variant_family"] = manifest["oligo_id"].apply(_family)
    manifest["is_reference"] = manifest.apply(_is_ref, axis=1)
    return manifest


def compute_variant_delta_scores(
    activity_results_path: str | Path,
    design_manifest_path: str | Path,
    upload_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    For each variant family, compute delta = mutant_log2_ratio - reference_log2_ratio.
    Tests significance via z-test across all deltas (BH FDR).
    Saves variant_delta_scores.tsv when upload_dir is provided.

    If the manifest lacks variant_family / is_reference columns they are
    automatically derived from the oligo_id using the R:/A:/C: prefix
    convention used by lentiMPRA design tables.
    """
    results = pd.read_csv(activity_results_path, sep="\t")
    manifest = pd.read_csv(design_manifest_path, sep="\t")

    if not {"variant_family", "is_reference"}.issubset(manifest.columns):
        manifest = _add_variant_cols(manifest)

    results = results.drop(
        columns=[c for c in ("variant_family", "is_reference") if c in results.columns]
    )
    df = results.merge(
        manifest[["oligo_id", "variant_family", "is_reference"]],
        on="oligo_id", how="left",
    )
    if df["is_reference"].dtype != bool:
        df["is_reference"] = (
            df["is_reference"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
            .fillna(False)
            .astype(bool)
        )

    families = df["variant_family"].dropna().unique()
    rows = []
    skipped = 0

    for fam in families:
        fam_df = df[df["variant_family"] == fam]
        ref_rows = fam_df[fam_df["is_reference"]]
        if len(ref_rows) == 0:
            skipped += 1
            continue
        ref_log2 = float(ref_rows["log2_ratio"].iloc[0])
        mutants = fam_df[~fam_df["is_reference"]]
        for _, row in mutants.iterrows():
            rows.append({
                "oligo_id": row["oligo_id"],
                "variant_family": fam,
                "ref_log2": ref_log2,
                "mutant_log2": float(row["log2_ratio"]),
                "delta_log2": float(row["log2_ratio"]) - ref_log2,
            })

    if not rows:
        return pd.DataFrame(), {
            "n_families": int(len(families)),
            "n_families_skipped": skipped,
            "n_mutants": 0,
            "warnings": ["No variant families with recovered references found."],
            "pass": False,
        }

    delta_df = pd.DataFrame(rows)

    all_deltas = delta_df["delta_log2"].values
    delta_mean = float(all_deltas.mean())
    delta_std = max(float(all_deltas.std(ddof=1)), 1e-9)
    z_scores = (all_deltas - delta_mean) / delta_std
    pvals = list(1.0 - _norm.cdf(z_scores))
    _, fdrs, _, _ = multipletests(pvals, method="fdr_bh")

    delta_df["pval"] = pvals
    delta_df["fdr"] = fdrs
    delta_df["significant"] = delta_df["fdr"] < 0.05

    if upload_dir is not None:
        delta_df.to_csv(Path(upload_dir) / "variant_delta_scores.tsv", sep="\t", index=False)

    return delta_df, {
        "n_families": int(len(families)),
        "n_families_skipped": skipped,
        "n_mutants": int(len(delta_df)),
        "n_significant": int(delta_df["significant"].sum()),
        "median_abs_delta": round(float(delta_df["delta_log2"].abs().median()), 4),
        "warnings": [],
        "pass": True,
    }
