"""
tests/qc/test_activity.py
=========================
Tests for creseq_mcp/qc/activity.py.

Coverage:
  - normalize_and_compute_ratios: happy path, min_barcodes filter, manifest merge
  - call_activity: z-test path with neg controls, fallback threshold path
  - activity_report: end-to-end save
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from creseq_mcp.qc.activity import (
    normalize_and_compute_ratios,
    call_activity,
    activity_report,
)


# ---------------------------------------------------------------------------
# Minimal fixture builders
# ---------------------------------------------------------------------------

def _write_plasmid(tmp_path, rows):
    p = tmp_path / "plasmid_counts.tsv"
    pd.DataFrame(rows).to_csv(p, sep="\t", index=False)
    return str(p)


def _write_rna(tmp_path, rows):
    p = tmp_path / "rna_counts.tsv"
    pd.DataFrame(rows).to_csv(p, sep="\t", index=False)
    return str(p)


def _write_manifest(tmp_path, rows):
    p = tmp_path / "design_manifest.tsv"
    pd.DataFrame(rows).to_csv(p, sep="\t", index=False)
    return str(p)


def _minimal_data(tmp_path, n_oligos=20, n_barcodes=5, seed=0):
    """Build minimal plasmid + RNA count TSVs with n_oligos * n_barcodes rows."""
    rng = np.random.default_rng(seed)
    plasmid_rows, rna_rows = [], []
    for i in range(n_oligos):
        oid = f"OLG{i:03d}"
        for j in range(n_barcodes):
            bc = f"BC{i:03d}{j:02d}"
            dna = int(rng.integers(20, 100))
            r1 = int(rng.integers(10, 200))
            r2 = int(rng.integers(10, 200))
            plasmid_rows.append({"barcode": bc, "oligo_id": oid, "dna_count": dna})
            rna_rows.append({"barcode": bc, "oligo_id": oid, "rna_count_rep1": r1, "rna_count_rep2": r2})
    return _write_plasmid(tmp_path, plasmid_rows), _write_rna(tmp_path, rna_rows)


# ---------------------------------------------------------------------------
# normalize_and_compute_ratios
# ---------------------------------------------------------------------------

class TestNormalizeAndComputeRatios:
    def test_happy_path(self, tmp_path):
        dna_p, rna_p = _minimal_data(tmp_path)
        df, summary = normalize_and_compute_ratios(dna_p, rna_p)

        assert isinstance(df, pd.DataFrame)
        assert "log2_ratio" in df.columns
        assert "oligo_id" in df.columns
        assert len(df) == 20
        assert summary["n_oligos_after_filter"] == 20

    def test_min_barcodes_filter(self, tmp_path):
        # 3 barcodes per oligo, min_barcodes=4 → all filtered out
        dna_p, rna_p = _minimal_data(tmp_path, n_barcodes=3)
        df, summary = normalize_and_compute_ratios(dna_p, rna_p, min_barcodes=4)
        assert len(df) == 0
        assert summary["n_oligos_after_filter"] == 0

    def test_manifest_merge_adds_category(self, tmp_path):
        dna_p, rna_p = _minimal_data(tmp_path)
        manifest_rows = [{"oligo_id": f"OLG{i:03d}", "designed_category": "negative_control"} for i in range(20)]
        manifest_p = _write_manifest(tmp_path, manifest_rows)

        df, _ = normalize_and_compute_ratios(dna_p, rna_p, manifest_p)
        assert "designed_category" in df.columns
        assert (df["designed_category"] == "negative_control").all()

    def test_no_rna_rep_cols_raises(self, tmp_path):
        dna_p = _write_plasmid(tmp_path, [{"barcode": "AAAA", "oligo_id": "O1", "dna_count": 10}])
        rna_p = _write_rna(tmp_path, [{"barcode": "AAAA", "oligo_id": "O1", "bad_col": 5}])
        with pytest.raises(ValueError, match="rna_count_"):
            normalize_and_compute_ratios(dna_p, rna_p)


# ---------------------------------------------------------------------------
# call_activity
# ---------------------------------------------------------------------------

class TestCallActivity:
    def _make_oligo_df(self, n=60, neg_mean=-0.5, pos_mean=2.5, seed=1):
        rng = np.random.default_rng(seed)
        rows = []
        for i in range(n):
            cat = "negative_control" if i < 20 else "positive_control" if i < 30 else "test_element"
            base = neg_mean if cat == "negative_control" else pos_mean if cat == "positive_control" else rng.choice([neg_mean, pos_mean])
            rows.append({"oligo_id": f"O{i}", "log2_ratio": float(base + rng.normal(0, 0.2)), "designed_category": cat})
        return pd.DataFrame(rows)

    def test_z_test_path(self):
        df = self._make_oligo_df()
        result, summary = call_activity(df)

        assert summary["method"] == "z_test_vs_neg_ctrl"
        assert summary["n_active"] > 0
        assert "fdr" in result.columns
        assert "active" in result.columns

    def test_fallback_threshold(self):
        # No designed_category column → fallback to log2>1 threshold
        df = self._make_oligo_df()
        df = df.drop(columns=["designed_category"])
        result, summary = call_activity(df)

        assert summary["method"] == "threshold_log2gt1"
        assert result["active"].dtype == bool

    def test_too_few_neg_controls_fallback(self):
        # Only 2 neg controls → fewer than 3 required → fallback
        rows = [{"oligo_id": f"O{i}", "log2_ratio": float(i * 0.1), "designed_category": "negative_control" if i < 2 else "test_element"} for i in range(20)]
        df = pd.DataFrame(rows)
        result, summary = call_activity(df)
        assert summary["method"] == "threshold_log2gt1"


# ---------------------------------------------------------------------------
# activity_report
# ---------------------------------------------------------------------------

class TestActivityReport:
    def test_saves_tsv(self, tmp_path):
        dna_p, rna_p = _minimal_data(tmp_path, n_oligos=30, n_barcodes=5)
        manifest_rows = [{"oligo_id": f"OLG{i:03d}", "designed_category": "negative_control" if i < 10 else "test_element"} for i in range(30)]
        manifest_p = _write_manifest(tmp_path, manifest_rows)

        df, summary = activity_report(dna_p, rna_p, manifest_p, upload_dir=tmp_path)

        out_file = tmp_path / "activity_results.tsv"
        assert out_file.exists()
        saved = pd.read_csv(out_file, sep="\t")
        assert "log2_ratio" in saved.columns
        assert "active" in saved.columns
        assert len(saved) == len(df)

    def test_no_manifest(self, tmp_path):
        dna_p, rna_p = _minimal_data(tmp_path)
        df, summary = activity_report(dna_p, rna_p, upload_dir=tmp_path)
        assert "log2_ratio" in df.columns
        assert "n_oligos_after_filter" in summary
