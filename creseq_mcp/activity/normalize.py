"""
creseq_mcp/activity/normalize.py
=================================
Normalization and activity calling for lentiMPRA / CRE-seq.

Steps
-----
1. RPM-style size-factor normalization (per sample)
2. log2(RNA/DNA) per barcode, averaged across replicates
3. Collapse to per-oligo median (requiring min_barcodes)
4. Activity calling: empirical null (median/MAD, BH FDR)
   Falls back to log2_ratio > 1 threshold when controls are absent.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PSEUDO = 0.5  # pseudocount added before log2


def normalize_and_compute_ratios(
    dna_counts_path: str | Path,
    rna_counts_path: str | Path,
    design_manifest_path: str | Path | None = None,
    *,
    min_barcodes: int = 2,
) -> tuple[pd.DataFrame, dict]:
    """
    Normalize DNA and RNA counts → per-barcode log2(RNA/DNA) → collapse to per-oligo.
    """
    dna = pd.read_csv(dna_counts_path, sep="\t")
    rna = pd.read_csv(rna_counts_path, sep="\t")

    rep_cols = [c for c in rna.columns if c.startswith("rna_count_")]
    if not rep_cols:
        raise ValueError("rna_counts.tsv has no rna_count_* columns")

    merged = dna[["barcode", "oligo_id", "dna_count"]].merge(
        rna[["barcode"] + rep_cols], on="barcode", how="inner"
    )

    dna_sf = max(merged["dna_count"].sum() / 1e6, 1e-9)
    merged["norm_dna"] = (merged["dna_count"] + _PSEUDO) / dna_sf

    log2_cols = []
    for col in rep_cols:
        sf = max(merged[col].sum() / 1e6, 1e-9)
        norm_col = f"norm_{col}"
        merged[norm_col] = (merged[col] + _PSEUDO) / sf
        log2_col = f"log2_{col}"
        merged[log2_col] = np.log2(merged[norm_col] / merged["norm_dna"])
        log2_cols.append(log2_col)

    merged["log2_ratio"] = merged[log2_cols].mean(axis=1)

    oligo_df = (
        merged.groupby("oligo_id")
        .agg(**{
            "n_barcodes": ("barcode", "count"),
            "median_dna": ("dna_count", "median"),
            "log2_ratio": ("log2_ratio", "median"),
            **{col: (col, "median") for col in log2_cols},
        })
        .reset_index()
    )

    oligo_df = oligo_df[oligo_df["n_barcodes"] >= min_barcodes].copy()

    if design_manifest_path and Path(design_manifest_path).exists():
        manifest = pd.read_csv(design_manifest_path, sep="\t")
        oligo_df = oligo_df.merge(manifest, on="oligo_id", how="left")

    return oligo_df, {
        "n_barcodes_merged": len(merged),
        "n_oligos_after_filter": len(oligo_df),
        "min_barcodes_filter": min_barcodes,
        "replicates": rep_cols,
        "median_log2_ratio": float(oligo_df["log2_ratio"].median()),
    }


def _call_activity(
    oligo_df: pd.DataFrame,
    *,
    neg_ctrl_category: str = "negative_control",
    fdr_threshold: float = 0.05,
) -> tuple[pd.DataFrame, dict]:
    """
    Classify CREs using the empirical null (median/MAD + BH FDR).
    Falls back to log2_ratio > 1 when <3 controls are present.
    """
    from creseq_mcp.activity.classify import call_active_elements_empirical

    df = oligo_df.copy()

    activity_table = df.rename(columns={"oligo_id": "element_id", "log2_ratio": "mean_activity"})
    if "n_barcodes" not in activity_table.columns:
        activity_table["n_barcodes"] = pd.NA

    neg_ctrl_ids: list[str] = []
    if "designed_category" in df.columns:
        neg_ctrl_ids = (
            df.loc[df["designed_category"] == neg_ctrl_category, "oligo_id"]
            .dropna()
            .tolist()
        )

    if len(neg_ctrl_ids) >= 3:
        classified, summary = call_active_elements_empirical(
            activity_table, neg_ctrl_ids, fdr_threshold
        )
        classified = classified.rename(columns={"element_id": "oligo_id", "mean_activity": "log2_ratio"})
        keep_cols = [c for c in ("active", "pvalue", "fdr", "zscore", "fold_over_controls") if c in classified.columns]
        df = df.merge(classified[["oligo_id"] + keep_cols], on="oligo_id", how="left")
        return df, {
            "method": "empirical_median_mad",
            "n_neg_controls": len(neg_ctrl_ids),
            "fdr_threshold": fdr_threshold,
            "n_active": int(df["active"].sum()),
            "n_inactive": int((~df["active"]).sum()),
            "activity_rate": round(float(df["active"].mean()), 4),
            "warnings": summary.get("warnings", []),
        }

    df["pvalue"] = np.nan
    df["fdr"] = np.nan
    df["active"] = df["log2_ratio"] > 1.0
    return df, {
        "method": "threshold_log2gt1",
        "n_active": int(df["active"].sum()),
        "n_inactive": int((~df["active"]).sum()),
        "activity_rate": round(float(df["active"].mean()), 4),
    }


def activity_report(
    dna_counts_path: str | Path,
    rna_counts_path: str | Path,
    design_manifest_path: str | Path | None = None,
    upload_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Full pipeline: normalize → empirical classifier → save activity_results.tsv."""
    oligo_df, norm_summary = normalize_and_compute_ratios(
        dna_counts_path, rna_counts_path, design_manifest_path
    )
    results_df, call_summary = _call_activity(oligo_df)

    if upload_dir is not None:
        results_df.to_csv(upload_dir / "activity_results.tsv", sep="\t", index=False)

    return results_df, {**norm_summary, **call_summary}
