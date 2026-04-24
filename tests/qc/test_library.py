"""
tests/qc/test_library.py
========================
Pytest tests for creseq_mcp/qc/library.py.

Coverage:
  - Happy path for every tool
  - CreSeqAssumptionWarning on wrong barcode length (15 bp)
  - CreSeqAssumptionWarning on wrong oligo length (500 bp)
  - variant_family_coverage fails when reference is missing
  - ValueError on empty input
  - Threshold pass vs. fail cases
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from creseq_mcp.qc.library import (
    CreSeqAssumptionWarning,
    barcode_collision_analysis,
    barcode_complexity,
    barcode_uniformity,
    gc_content_bias,
    library_summary_report,
    oligo_length_qc,
    oligo_recovery,
    plasmid_depth_summary,
    synthesis_error_profile,
    variant_family_coverage,
)


# ===========================================================================
# 1. barcode_complexity
# ===========================================================================


class TestBarcodeComplexity:
    def test_happy_path(self, mapping_table_path):
        df, summ = barcode_complexity(mapping_table_path)

        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) >= {
            "oligo_id",
            "n_barcodes",
            "n_perfect_barcodes",
            "n_error_barcodes",
            "median_reads_per_barcode",
        }
        assert len(df) == 500  # one row per oligo
        assert "pass" in summ
        assert isinstance(summ["pass"], bool)
        assert summ["median_barcodes_per_oligo"] >= 10  # healthy library passes
        assert summ["pass"] is True

    def test_fraction_thresholds_present(self, mapping_table_path):
        _, summ = barcode_complexity(mapping_table_path)
        for t in [5, 10, 25, 50, 100]:
            key = f"fraction_oligos_gte_{t}_barcodes"
            assert key in summ
            assert 0.0 <= summ[key] <= 1.0

    def test_wrong_barcode_length_warns(self, long_barcode_mapping_path):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            df, summ = barcode_complexity(long_barcode_mapping_path)

        cseq_warns = [w for w in caught if issubclass(w.category, CreSeqAssumptionWarning)]
        assert len(cseq_warns) >= 1
        assert any("15" in str(w.message) or "barcode" in str(w.message).lower() for w in cseq_warns)
        assert summ["median_barcode_length_bp"] == pytest.approx(15, abs=0.5)
        assert len(summ["warnings"]) >= 1

    def test_min_reads_filter(self, mapping_table_path):
        df_low, _ = barcode_complexity(mapping_table_path, min_reads_per_barcode=1)
        df_high, _ = barcode_complexity(mapping_table_path, min_reads_per_barcode=50)
        # Higher filter → fewer or equal barcodes per oligo
        assert df_high["n_barcodes"].sum() <= df_low["n_barcodes"].sum()

    def test_empty_input_raises(self, empty_mapping_path):
        with pytest.raises(ValueError, match="empty|No barcodes"):
            barcode_complexity(empty_mapping_path)

    def test_fail_case(self, tmp_path):
        """Library with only 2 barcodes per oligo should fail median >= 10 check."""
        import numpy as np
        rng = np.random.default_rng(99)
        rows = []
        for i in range(20):
            for _ in range(2):
                rows.append(
                    {
                        "barcode": "".join(rng.choice(list("ACGT"), 10)),
                        "oligo_id": f"oligo_{i:03d}",
                        "n_reads": 10,
                        "cigar": "84M",
                        "md": "84",
                    }
                )
        p = tmp_path / "sparse.tsv"
        pd.DataFrame(rows).to_csv(p, sep="\t", index=False)
        _, summ = barcode_complexity(str(p))
        assert summ["pass"] is False


# ===========================================================================
# 2. oligo_recovery
# ===========================================================================


class TestOligoRecovery:
    def test_happy_path(self, mapping_table_path, design_manifest_path):
        df, summ = oligo_recovery(mapping_table_path, design_manifest_path)

        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) >= {
            "oligo_id",
            "designed_category",
            "designed",
            "recovered",
            "n_barcodes",
        }
        for t in [5, 10, 25]:
            assert f"passes_threshold_{t}" in df.columns

        assert "recovery_by_category" in summ
        assert summ["pass"] is True  # healthy library passes

    def test_category_breakdown(self, mapping_table_path, design_manifest_path):
        _, summ = oligo_recovery(mapping_table_path, design_manifest_path)
        cats = summ["recovery_by_category"]
        for expected_cat in ["test_element", "positive_control"]:
            assert expected_cat in cats
            assert cats[expected_cat]["n_designed"] > 0

    def test_low_recovery_fails(self, low_recovery_mapping_path, design_manifest_path):
        _, summ = oligo_recovery(low_recovery_mapping_path, design_manifest_path)
        # 50% recovery across all categories should fail positive_control threshold
        assert summ["pass"] is False
        assert len(summ["warnings"]) >= 1

    def test_custom_thresholds(self, mapping_table_path, design_manifest_path):
        df, _ = oligo_recovery(
            mapping_table_path, design_manifest_path, thresholds=[1, 5]
        )
        assert "passes_threshold_1" in df.columns
        assert "passes_threshold_5" in df.columns
        assert "passes_threshold_10" not in df.columns

    def test_empty_input_raises(self, empty_mapping_path, design_manifest_path):
        with pytest.raises(ValueError):
            oligo_recovery(empty_mapping_path, design_manifest_path)


# ===========================================================================
# 3. synthesis_error_profile
# ===========================================================================


class TestSynthesisErrorProfile:
    def test_happy_path(self, mapping_table_path):
        df, summ = synthesis_error_profile(mapping_table_path)

        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) >= {
            "oligo_id",
            "n_barcodes",
            "n_perfect_barcodes",
            "perfect_fraction",
            "mean_mismatches",
        }
        assert summ["pass"] is True  # 85% perfect → median perfect_fraction >> 0.5

    def test_with_manifest(self, mapping_table_path, design_manifest_path):
        df, summ = synthesis_error_profile(mapping_table_path, design_manifest_path)
        # Should succeed; GC spearman reported if scipy available
        assert "pass" in summ

    def test_wrong_oligo_length_manifest_warns(
        self, mapping_table_path, long_oligo_manifest_path
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            synthesis_error_profile(mapping_table_path, long_oligo_manifest_path)

        cseq_warns = [w for w in caught if issubclass(w.category, CreSeqAssumptionWarning)]
        assert len(cseq_warns) >= 1

    def test_fail_case(self, tmp_path):
        """Library where all barcodes have a mismatch → perfect_fraction = 0 → fail."""
        rows = [
            {
                "barcode": "ACGTACGTAC",
                "oligo_id": f"oligo_{i:03d}",
                "n_reads": 10,
                "cigar": "84M",
                "md": f"42A41",  # one mismatch per barcode
            }
            for i in range(50)
            for _ in range(5)
        ]
        p = tmp_path / "mismatch.tsv"
        pd.DataFrame(rows).to_csv(p, sep="\t", index=False)
        _, summ = synthesis_error_profile(str(p))
        assert summ["median_perfect_fraction"] == pytest.approx(0.0)
        assert summ["pass"] is False

    def test_empty_input_raises(self, empty_mapping_path):
        with pytest.raises(ValueError):
            synthesis_error_profile(empty_mapping_path)


# ===========================================================================
# 4. barcode_collision_analysis
# ===========================================================================


class TestBarcodeCollisionAnalysis:
    def test_happy_path_no_collisions(self, mapping_table_path):
        df, summ = barcode_collision_analysis(mapping_table_path)

        assert isinstance(df, pd.DataFrame)
        assert "barcode" in df.columns
        assert "n_oligos_mapped" in df.columns
        # In synthetic data with random 10-mer barcodes, collisions should be rare
        assert summ["collision_rate"] < 0.03
        assert summ["pass"] is True

    def test_high_collision_fails(self, tmp_path):
        """Reusing barcodes across oligos → high collision rate."""
        rows = []
        # 5 barcodes each mapped to 10 different oligos
        shared_bcs = ["AAAAAAAAAA", "CCCCCCCCCC", "GGGGGGGGGG", "TTTTTTTTTT", "ACGTACGTAC"]
        for bc in shared_bcs:
            for i in range(10):
                rows.append(
                    {
                        "barcode": bc,
                        "oligo_id": f"oligo_{i:03d}",
                        "n_reads": 5,
                        "cigar": "84M",
                        "md": "84",
                    }
                )
        # Add some unique barcodes so total > 0
        for i in range(2):
            rows.append(
                {
                    "barcode": f"AAAAAA{i:04d}",
                    "oligo_id": "oligo_0000",
                    "n_reads": 5,
                    "cigar": "84M",
                    "md": "84",
                }
            )
        p = tmp_path / "collisions.tsv"
        pd.DataFrame(rows).to_csv(p, sep="\t", index=False)
        df, summ = barcode_collision_analysis(str(p), min_read_support=2)
        assert summ["collision_rate"] > 0.03
        assert summ["pass"] is False
        assert len(df) == 5  # all 5 shared barcodes collide

    def test_empty_input_raises(self, empty_mapping_path):
        with pytest.raises(ValueError):
            barcode_collision_analysis(empty_mapping_path)


# ===========================================================================
# 5. barcode_uniformity
# ===========================================================================


class TestBarcodeUniformity:
    def test_happy_path(self, plasmid_count_path):
        df, summ = barcode_uniformity(plasmid_count_path)

        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) >= {
            "oligo_id",
            "n_barcodes",
            "total_dna_count",
            "gini_coefficient",
            "effective_barcodes",
        }
        assert 0.0 <= summ["median_gini"] <= 1.0
        assert summ["pass"] is True  # uniformly distributed counts pass

    def test_skewed_counts_fail(self, tmp_path):
        """One barcode dominates → high Gini → fail."""
        rows = []
        for oligo_id in [f"oligo_{i:03d}" for i in range(20)]:
            # One barcode with 10,000 counts, rest with 1
            for j in range(10):
                rows.append(
                    {
                        "barcode": f"ACGT{j:06d}",
                        "oligo_id": oligo_id,
                        "dna_count": 10000 if j == 0 else 1,
                    }
                )
        p = tmp_path / "skewed.tsv"
        pd.DataFrame(rows).to_csv(p, sep="\t", index=False)
        _, summ = barcode_uniformity(str(p))
        assert summ["median_gini"] > 0.30
        assert summ["pass"] is False

    def test_min_barcodes_filter(self, plasmid_count_path):
        # Fixture has 15-25 barcodes/oligo; threshold=20 should exclude some oligos
        _, summ_low = barcode_uniformity(plasmid_count_path, min_barcodes_per_oligo=1)
        _, summ_high = barcode_uniformity(plasmid_count_path, min_barcodes_per_oligo=20)
        assert summ_high["n_oligos_evaluated"] <= summ_low["n_oligos_evaluated"]

    def test_empty_input_raises(self, tmp_path):
        p = tmp_path / "empty_plasmid.tsv"
        pd.DataFrame(columns=["barcode", "oligo_id", "dna_count"]).to_csv(
            p, sep="\t", index=False
        )
        with pytest.raises(ValueError):
            barcode_uniformity(str(p))


# ===========================================================================
# 6. gc_content_bias
# ===========================================================================


class TestGcContentBias:
    def test_happy_path(self, mapping_table_path, design_manifest_path):
        df, summ = gc_content_bias(mapping_table_path, design_manifest_path)

        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) >= {
            "gc_bin_center",
            "gc_bin_lo",
            "gc_bin_hi",
            "n_designed",
            "n_recovered",
            "recovery_rate",
        }
        assert "pass" in summ
        assert "gc_bias_detected" in summ

    def test_wrong_oligo_length_warns(self, mapping_table_path, long_oligo_manifest_path):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gc_content_bias(mapping_table_path, long_oligo_manifest_path)

        cseq_warns = [w for w in caught if issubclass(w.category, CreSeqAssumptionWarning)]
        assert len(cseq_warns) >= 1

    def test_no_gc_content_raises(self, mapping_table_path, tmp_path):
        manifest = pd.DataFrame(
            {
                "oligo_id": [f"oligo_{i:04d}" for i in range(10)],
                "designed_category": ["test_element"] * 10,
            }
        )
        p = tmp_path / "no_gc_manifest.tsv"
        manifest.to_csv(p, sep="\t", index=False)
        with pytest.raises(ValueError, match="gc_content"):
            gc_content_bias(mapping_table_path, str(p))


# ===========================================================================
# 7. oligo_length_qc
# ===========================================================================


class TestOligoLengthQc:
    def test_happy_path(self, mapping_table_path, design_manifest_path):
        df, summ = oligo_length_qc(mapping_table_path, design_manifest_path)

        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) >= {
            "oligo_id",
            "designed_length",
            "observed_length_mode",
            "fraction_full_length",
            "fraction_truncated_gt10bp",
            "n_barcodes",
        }
        assert summ["pass"] is True
        assert summ["median_fraction_full_length"] > 0.8

    def test_wrong_oligo_length_warns(self, mapping_table_path, long_oligo_manifest_path):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            oligo_length_qc(mapping_table_path, long_oligo_manifest_path)

        cseq_warns = [w for w in caught if issubclass(w.category, CreSeqAssumptionWarning)]
        assert len(cseq_warns) >= 1

    def test_mixed_length_design_warns(self, mapping_table_path, tmp_path):
        """Non-fixed-length designs should raise CreSeqAssumptionWarning."""
        import numpy as np
        rng = np.random.default_rng(77)
        rows = [
            {
                "oligo_id": f"oligo_{i:04d}",
                "sequence": "A" * (84 + i % 5),  # lengths 84-88
                "length": 84 + i % 5,
                "gc_content": 0.5,
                "designed_category": "test_element",
                "parent_element_id": None,
            }
            for i in range(50)
        ]
        p = tmp_path / "mixed_len_manifest.tsv"
        pd.DataFrame(rows).to_csv(p, sep="\t", index=False)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            oligo_length_qc(mapping_table_path, str(p))

        cseq_warns = [w for w in caught if issubclass(w.category, CreSeqAssumptionWarning)]
        assert len(cseq_warns) >= 1
        assert any("length" in str(w.message).lower() for w in cseq_warns)

    def test_no_length_column_raises(self, mapping_table_path, tmp_path):
        manifest = pd.DataFrame({"oligo_id": [f"oligo_{i:04d}" for i in range(10)]})
        p = tmp_path / "no_len.tsv"
        manifest.to_csv(p, sep="\t", index=False)
        with pytest.raises(ValueError, match="length"):
            oligo_length_qc(mapping_table_path, str(p))


# ===========================================================================
# 8. plasmid_depth_summary
# ===========================================================================


class TestPlasmidDepthSummary:
    def test_happy_path(self, plasmid_count_path):
        df, summ = plasmid_depth_summary(plasmid_count_path)

        assert isinstance(df, pd.DataFrame)
        assert "dna_count" in df.columns
        assert "log10_dna_count" in df.columns
        assert summ["pass"] is True
        assert summ["median_count"] >= 10
        assert summ["pct_barcodes_zero"] < 0.10

    def test_fail_low_counts(self, tmp_path):
        """Library with median count = 1 should fail."""
        rows = [
            {"barcode": f"ACGT{i:06d}", "oligo_id": f"oligo_{i % 50:03d}", "dna_count": 1}
            for i in range(500)
        ]
        p = tmp_path / "low_counts.tsv"
        pd.DataFrame(rows).to_csv(p, sep="\t", index=False)
        _, summ = plasmid_depth_summary(str(p))
        assert summ["pass"] is False

    def test_fail_many_zeros(self, tmp_path):
        """Library with 20% zeros should fail."""
        rows = [
            {
                "barcode": f"ACGT{i:06d}",
                "oligo_id": f"oligo_{i % 50:03d}",
                "dna_count": 0 if i % 5 == 0 else 100,
            }
            for i in range(500)
        ]
        p = tmp_path / "zeros.tsv"
        pd.DataFrame(rows).to_csv(p, sep="\t", index=False)
        _, summ = plasmid_depth_summary(str(p))
        assert summ["pct_barcodes_zero"] == pytest.approx(0.20, abs=0.01)
        assert summ["pass"] is False

    def test_empty_input_raises(self, tmp_path):
        p = tmp_path / "empty_plasmid2.tsv"
        pd.DataFrame(columns=["barcode", "oligo_id", "dna_count"]).to_csv(
            p, sep="\t", index=False
        )
        with pytest.raises(ValueError):
            plasmid_depth_summary(str(p))


# ===========================================================================
# 9. variant_family_coverage
# ===========================================================================


class TestVariantFamilyCoverage:
    def test_happy_path(self, mapping_table_path, design_manifest_path):
        df, summ = variant_family_coverage(mapping_table_path, design_manifest_path)

        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) >= {
            "parent_element_id",
            "n_variants_designed",
            "n_variants_recovered",
            "reference_recovered",
            "family_complete",
            "missing_variants",
        }
        assert summ["n_families"] == 30  # 30 knockout → test_element families
        assert summ["pass"] is True  # healthy library: all refs present

    def test_missing_reference_fails(self, missing_ref_mapping_path, design_manifest_path):
        df, summ = variant_family_coverage(missing_ref_mapping_path, design_manifest_path)

        assert summ["fraction_families_reference_missing"] > 0.0
        assert summ["pass"] is False
        assert len(summ["warnings"]) >= 1
        assert any("reference" in w.lower() for w in summ["warnings"])

    def test_no_families_returns_empty(self, mapping_table_path, tmp_path):
        """Manifest with no parent_element_id links → no families, pass=True."""
        manifest = pd.DataFrame(
            {
                "oligo_id": [f"oligo_{i:04d}" for i in range(50)],
                "designed_category": ["test_element"] * 50,
                "parent_element_id": [None] * 50,
            }
        )
        p = tmp_path / "no_families.tsv"
        manifest.to_csv(p, sep="\t", index=False)
        df, summ = variant_family_coverage(mapping_table_path, str(p))
        assert len(df) == 0
        assert summ["n_families"] == 0
        assert summ["pass"] is True

    def test_empty_mapping_raises(self, empty_mapping_path, design_manifest_path):
        with pytest.raises(ValueError):
            variant_family_coverage(empty_mapping_path, design_manifest_path)


# ===========================================================================
# 10. library_summary_report
# ===========================================================================


class TestLibrarySummaryReport:
    def test_happy_path_with_manifest(
        self, mapping_table_path, plasmid_count_path, design_manifest_path
    ):
        results = library_summary_report(
            mapping_table_path, plasmid_count_path, design_manifest_path
        )

        assert "_report" in results
        _, report_summ = results["_report"]
        assert "overall_pass" in report_summ
        assert isinstance(report_summ["failed_checks"], list)
        assert isinstance(report_summ["warnings"], list)
        assert isinstance(report_summ["skipped_tools"], list)

        # All 9 tools + _report
        tool_keys = [k for k in results if not k.startswith("_")]
        assert len(tool_keys) == 9

    def test_no_manifest_skips_manifest_tools(
        self, mapping_table_path, plasmid_count_path
    ):
        results = library_summary_report(mapping_table_path, plasmid_count_path)

        _, report_summ = results["_report"]
        skipped_names = {s["tool"] for s in report_summ["skipped_tools"]}
        for expected in [
            "oligo_recovery",
            "gc_content_bias",
            "oligo_length_qc",
            "variant_family_coverage",
        ]:
            assert expected in skipped_names

    def test_overall_pass_type(self, mapping_table_path, plasmid_count_path):
        results = library_summary_report(mapping_table_path, plasmid_count_path)
        _, report_summ = results["_report"]
        assert isinstance(report_summ["overall_pass"], bool)

    def test_empty_mapping_propagates_failures(
        self, empty_mapping_path, plasmid_count_path
    ):
        results = library_summary_report(empty_mapping_path, plasmid_count_path)
        _, report_summ = results["_report"]
        # Mapping-dependent tools should fail; overall_pass = False
        assert report_summ["overall_pass"] is False
        assert len(report_summ["failed_checks"]) >= 1
