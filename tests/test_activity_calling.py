"""
tests/test_activity_calling.py
==============================
Tests for creseq_mcp/activity_calling.py — full-featured empirical-null
activity caller.

Coverage (12 cases, one skipped pending GLM implementation):
  - Core correctness: positive controls active, controls excluded, inactives
    rejected, moderate-active recall.
  - FDR calibration under the null.
  - Robustness to outlier control.
  - Edge cases: too-few-controls warning, missing IDs raise, empty input.
  - Output schema: column names, summary keys.
  - Method concordance: empirical vs. GLM (skipped — GLM not implemented).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creseq_mcp.activity.classify import (
    call_active_elements,
    call_active_elements_empirical,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def activity_fixture() -> pd.DataFrame:
    """
    100-element synthetic activity table with a known structure:
      - 20 negative controls   ~ N(0.0, 0.3)
      - 10 positive controls   ~ N(3.0, 0.3)
      - 50 moderate active     ~ N(1.5, 0.4)
      - 20 inactive elements   ~ N(0.1, 0.3)
    """
    rng = np.random.default_rng(42)
    rows = []

    for i in range(20):
        rows.append({
            "element_id": f"neg_ctrl_{i:03d}",
            "mean_activity": float(rng.normal(0.0, 0.3)),
            "std_activity": 0.3,
            "n_barcodes": 15,
        })
    for i in range(10):
        rows.append({
            "element_id": f"pos_ctrl_{i:03d}",
            "mean_activity": float(rng.normal(3.0, 0.3)),
            "std_activity": 0.3,
            "n_barcodes": 15,
        })
    for i in range(50):
        rows.append({
            "element_id": f"active_{i:03d}",
            "mean_activity": float(rng.normal(1.5, 0.4)),
            "std_activity": 0.4,
            "n_barcodes": 15,
        })
    for i in range(20):
        rows.append({
            "element_id": f"inactive_{i:03d}",
            "mean_activity": float(rng.normal(0.1, 0.3)),
            "std_activity": 0.3,
            "n_barcodes": 15,
        })

    return pd.DataFrame(rows)


@pytest.fixture
def neg_control_ids() -> list[str]:
    return [f"neg_ctrl_{i:03d}" for i in range(20)]


# ---------------------------------------------------------------------------
# Core correctness
# ---------------------------------------------------------------------------


def test_positive_controls_called_active(activity_fixture, neg_control_ids):
    """Positive controls (activity ~3.0) should all be called active."""
    result, _ = call_active_elements_empirical(activity_fixture, neg_control_ids)
    pos = result[result["element_id"].str.startswith("pos_ctrl")]
    assert pos["active"].all(), "Not all positive controls called active"


def test_negative_controls_excluded_from_testing(activity_fixture, neg_control_ids):
    """Negative controls should have NaN p-values (not tested against themselves)."""
    result, _ = call_active_elements_empirical(activity_fixture, neg_control_ids)
    neg = result[result["element_id"].str.startswith("neg_ctrl")]
    assert neg["pvalue"].isna().all()
    assert neg["fdr"].isna().all()
    assert neg["zscore"].isna().all()
    assert neg["fold_over_controls"].isna().all()
    assert not neg["active"].any()


def test_inactive_elements_not_called_active(activity_fixture, neg_control_ids):
    """Inactive elements (activity ~0.1) should rarely be called active."""
    result, _ = call_active_elements_empirical(activity_fixture, neg_control_ids)
    inactive = result[result["element_id"].str.startswith("inactive")]
    false_pos_rate = inactive["active"].mean()
    assert false_pos_rate <= 0.15, f"Too many false positives: {false_pos_rate:.2f}"


def test_moderate_active_high_recall(activity_fixture, neg_control_ids):
    """Most moderate-active elements (activity ~1.5) should be called active."""
    result, _ = call_active_elements_empirical(activity_fixture, neg_control_ids)
    actives = result[result["element_id"].str.startswith("active_")]
    recall = actives["active"].mean()
    assert recall >= 0.75, f"Recall too low for moderate actives: {recall:.2f}"


# ---------------------------------------------------------------------------
# FDR calibration
# ---------------------------------------------------------------------------


def test_fdr_calibration_under_null():
    """
    When all elements come from the same null distribution, the BH FDR
    threshold should hold (calling rate ≤ ~threshold + slack).
    """
    rng = np.random.default_rng(123)
    rows = []
    for i in range(200):
        prefix = "neg_ctrl" if i < 50 else "test"
        rows.append({
            "element_id": f"{prefix}_{i:03d}",
            "mean_activity": float(rng.normal(0.0, 0.3)),
            "std_activity": 0.3,
            "n_barcodes": 15,
        })
    df = pd.DataFrame(rows)
    neg_ids = [f"neg_ctrl_{i:03d}" for i in range(50)]

    result, _ = call_active_elements_empirical(df, neg_ids, fdr_threshold=0.05)
    test = result[~result["element_id"].str.startswith("neg_ctrl")]
    false_pos_rate = test["active"].mean()
    assert false_pos_rate <= 0.10, f"FDR not calibrated: {false_pos_rate:.2f}"


# ---------------------------------------------------------------------------
# Null distribution robustness
# ---------------------------------------------------------------------------


def test_robust_null_with_outlier_controls():
    """
    A single outlier negative control should not break the null estimate
    (median/MAD is robust; mean/std would inflate the scale).
    """
    rng = np.random.default_rng(456)
    rows = []
    for i in range(19):
        rows.append({
            "element_id": f"neg_ctrl_{i:03d}",
            "mean_activity": float(rng.normal(0.0, 0.3)),
            "std_activity": 0.3,
            "n_barcodes": 15,
        })
    rows.append({
        "element_id": "neg_ctrl_019",
        "mean_activity": 8.0,  # extreme outlier
        "std_activity": 0.3,
        "n_barcodes": 15,
    })
    rows.append({
        "element_id": "test_active",
        "mean_activity": 2.5,
        "std_activity": 0.3,
        "n_barcodes": 15,
    })
    df = pd.DataFrame(rows)
    neg_ids = [f"neg_ctrl_{i:03d}" for i in range(20)]

    result, _summary = call_active_elements_empirical(df, neg_ids)
    test_row = result[result["element_id"] == "test_active"]
    assert bool(test_row["active"].iloc[0]), (
        "Outlier control broke the null estimate — median/MAD should resist it"
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_too_few_controls_warns(activity_fixture):
    """Should warn when fewer than 20 negative controls are provided."""
    few = [f"neg_ctrl_{i:03d}" for i in range(5)]
    with pytest.warns(UserWarning, match="fewer than 20"):
        call_active_elements_empirical(activity_fixture, few)


def test_missing_controls_raises(activity_fixture):
    """Should raise ValueError when no supplied control IDs exist in the table."""
    fake = ["does_not_exist_001", "does_not_exist_002"]
    with pytest.raises(ValueError, match="not found"):
        call_active_elements_empirical(activity_fixture, fake)


def test_empty_table_raises():
    """Should raise ValueError on empty input."""
    empty = pd.DataFrame(columns=["element_id", "mean_activity", "std_activity", "n_barcodes"])
    with pytest.raises(ValueError):
        call_active_elements_empirical(empty, ["neg_ctrl_000"])


# ---------------------------------------------------------------------------
# Output format
# ---------------------------------------------------------------------------


def test_output_columns(activity_fixture, neg_control_ids):
    """Output should have the expected columns."""
    result, _ = call_active_elements_empirical(activity_fixture, neg_control_ids)
    expected = {
        "element_id", "mean_activity", "std_activity", "n_barcodes",
        "active", "pvalue", "fdr", "fold_over_controls", "zscore",
    }
    assert expected.issubset(set(result.columns))


def test_summary_fields(activity_fixture, neg_control_ids):
    """Summary dict should have the required top-level keys."""
    _, summary = call_active_elements_empirical(activity_fixture, neg_control_ids)
    required = {
        "n_total_elements", "n_negative_controls", "n_test_elements",
        "n_active", "n_inactive", "fdr_threshold", "method",
        "null_distribution", "active_summary",
    }
    assert required.issubset(set(summary.keys()))


# ---------------------------------------------------------------------------
# Method concordance
# ---------------------------------------------------------------------------


def test_empirical_and_glm_agree(activity_fixture, neg_control_ids, tmp_path):
    """Empirical and GLM methods should agree on >80% of calls (well-powered)."""
    pytest.skip(
        "GLM method not yet implemented — enable when activity_calling.py "
        "has both methods"
    )
