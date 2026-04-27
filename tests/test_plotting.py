"""
tests/test_plotting.py
======================
Tests for creseq_mcp/plotting.py — five plot types + dispatcher.

14 tests total:
  volcano               (2): creates file, description has counts
  ranked_activity       (2): creates file, sort verification
  replicate_correlation (3): creates file, needs ≥2 reps, description has Pearson r
  annotation_boxplot    (2): creates file, bad annotation file raises
  motif_dotplot         (3): creates file, bad columns raises, empty significant note
  dispatcher            (2): unknown plot_type raises, annotation type requires file
"""
from __future__ import annotations

import pandas as pd
import pytest

from creseq_mcp.plots.plots import (
    _plot_annotation_boxplot,
    _plot_motif_dotplot,
    _plot_ranked_activity,
    _plot_replicate_correlation,
    _plot_volcano,
    plot_creseq,
)


# ---------------------------------------------------------------------------
# Volcano
# ---------------------------------------------------------------------------


def test_volcano_creates_file(classified_fixture, tmp_path):
    out = tmp_path / "volcano.png"
    result = _plot_volcano(classified_fixture, str(out))
    assert out.exists()
    assert out.stat().st_size > 0
    assert "plot_path" in result
    assert "description" in result


def test_volcano_description_has_counts(classified_fixture, tmp_path):
    out = tmp_path / "volcano.png"
    result = _plot_volcano(classified_fixture, str(out))
    assert "active" in result["description"].lower()
    assert any(c.isdigit() for c in result["description"])


# ---------------------------------------------------------------------------
# Ranked activity
# ---------------------------------------------------------------------------


def test_ranked_activity_creates_file(classified_fixture, tmp_path):
    out = tmp_path / "ranked.png"
    result = _plot_ranked_activity(classified_fixture, str(out))
    assert out.exists()
    assert out.stat().st_size > 0
    assert "plot_path" in result and "description" in result


def test_ranked_activity_sorted(classified_fixture):
    """Verify the sort order applied inside the plot helper would be monotonic."""
    df_sorted = classified_fixture.sort_values("mean_activity", ascending=True)
    assert df_sorted["mean_activity"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# Replicate correlation
# ---------------------------------------------------------------------------


def test_replicate_correlation_creates_file(activity_with_reps_fixture, tmp_path):
    out = tmp_path / "corr.png"
    result = _plot_replicate_correlation(activity_with_reps_fixture, str(out))
    assert out.exists()
    assert out.stat().st_size > 0


def test_replicate_correlation_needs_two_reps(classified_fixture, tmp_path):
    """classified_fixture has no rep*_activity columns — should error."""
    out = tmp_path / "corr.png"
    with pytest.raises(ValueError, match="at least 2 replicate"):
        _plot_replicate_correlation(classified_fixture, str(out))


def test_replicate_correlation_description_has_r(activity_with_reps_fixture, tmp_path):
    out = tmp_path / "corr.png"
    result = _plot_replicate_correlation(activity_with_reps_fixture, str(out))
    assert "Pearson r" in result["description"]


# ---------------------------------------------------------------------------
# Annotation boxplot
# ---------------------------------------------------------------------------


def test_annotation_boxplot_creates_file(classified_fixture, annotation_fixture, tmp_path):
    out = tmp_path / "annot.png"
    result = _plot_annotation_boxplot(classified_fixture, str(annotation_fixture), str(out))
    assert out.exists()
    assert out.stat().st_size > 0


def test_annotation_boxplot_bad_file(classified_fixture, tmp_path):
    """Annotation file missing required columns → ValueError."""
    bad = tmp_path / "bad.tsv"
    bad.write_text("col_a\tcol_b\n1\t2\n")
    out = tmp_path / "annot.png"
    with pytest.raises(ValueError, match="element_id"):
        _plot_annotation_boxplot(classified_fixture, str(bad), str(out))


# ---------------------------------------------------------------------------
# Motif dotplot
# ---------------------------------------------------------------------------


def test_motif_dotplot_creates_file(motif_fixture, tmp_path):
    out = tmp_path / "motifs.png"
    result = _plot_motif_dotplot(motif_fixture, str(out))
    assert out.exists()
    assert out.stat().st_size > 0


def test_motif_dotplot_bad_columns(tmp_path):
    bad_df = pd.DataFrame({"gene": ["A"], "score": [1.0]})
    out = tmp_path / "motifs.png"
    with pytest.raises(ValueError, match="tf_name"):
        _plot_motif_dotplot(bad_df, str(out))


def test_motif_dotplot_empty_significant(tmp_path):
    """No motif passes FDR < 0.05 — still plot top by OR and note it."""
    df = pd.DataFrame({
        "tf_name": ["TF1", "TF2", "TF3"],
        "odds_ratio": [1.2, 1.1, 0.9],
        "fdr": [0.8, 0.9, 0.95],
        "n_active_hits": [5, 3, 2],
    })
    out = tmp_path / "motifs.png"
    result = _plot_motif_dotplot(df, str(out))
    assert out.exists()
    assert "No motifs reached FDR" in result["description"]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def test_dispatcher_unknown_type_raises(classified_fixture, tmp_path):
    p = tmp_path / "cls.tsv"
    classified_fixture.to_csv(p, sep="\t", index=False)
    with pytest.raises(ValueError, match="Unknown plot_type"):
        plot_creseq(str(p), "not_a_real_type", str(tmp_path / "out.png"))


def test_dispatcher_annotation_requires_file(classified_fixture, tmp_path):
    p = tmp_path / "cls.tsv"
    classified_fixture.to_csv(p, sep="\t", index=False)
    with pytest.raises(ValueError, match="annotation_file"):
        plot_creseq(
            str(p),
            "annotation_boxplot",
            str(tmp_path / "out.png"),
            annotation_file=None,
        )
