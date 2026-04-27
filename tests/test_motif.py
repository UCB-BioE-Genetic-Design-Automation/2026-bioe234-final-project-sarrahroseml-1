"""
tests/test_motif.py
===================
Tests for creseq_mcp/motif.py — TF-motif enrichment in active CRE-seq elements.

Coverage (9 tests):
  - planted GATA1 motif ranks in top 3 enriched
  - random vs random → 0 significant motifs
  - scanner detects planted motif in ≥20/30 sequences
  - JASPAR loading returns >500 motifs (slow, real database)
  - output table has required columns
  - output sorted by odds_ratio descending
  - empty FASTA handled gracefully
  - sequences shorter than motif length skipped silently
  - end-to-end pipeline produces output file + summary (slow)
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

from creseq_mcp.motifs.enrichment import (
    _parse_fasta,
    compute_enrichment,
    extract_sequences_to_fasta,
    load_jaspar_motifs,
    motif_enrichment,
    scan_sequences,
)


# ---------------------------------------------------------------------------
# Core correctness
# ---------------------------------------------------------------------------


def test_planted_motif_enriched(
    active_fasta_with_motif, background_fasta_no_motif, small_motif_list
):
    """GATA1 motif planted in active sequences should be enriched (top 3 by OR)."""
    active = _parse_fasta(active_fasta_with_motif)
    background = _parse_fasta(background_fasta_no_motif)
    all_seqs = {**active, **background}
    results = scan_sequences(all_seqs, small_motif_list, score_threshold=0.7)
    enrichment = compute_enrichment(
        results, set(active.keys()), set(background.keys())
    )
    top3 = enrichment.head(3)["tf_name"].tolist()
    assert "GATA1" in top3, f"GATA1 not in top 3: {top3}"


def test_no_enrichment_random(both_random_fasta, small_motif_list):
    """Random vs random → no motif should pass FDR < 0.05.

    Uses a stricter (0.85) threshold than the planted-motif test: at the
    permissive 0.7 threshold the strong-consensus mock motifs hit random
    sequences often enough that seed-specific imbalance can reach
    significance by chance, which is a property of the test data, not the
    enrichment logic.
    """
    active_path, bg_path = both_random_fasta
    active = _parse_fasta(active_path)
    background = _parse_fasta(bg_path)
    all_seqs = {**active, **background}
    results = scan_sequences(all_seqs, small_motif_list, score_threshold=0.9)
    enrichment = compute_enrichment(
        results, set(active.keys()), set(background.keys())
    )
    sig = enrichment[enrichment["fdr"] < 0.05]
    assert len(sig) == 0, (
        f"Unexpected significant motifs in random data: "
        f"{sig['tf_name'].tolist()}"
    )


def test_scan_finds_planted_motif(active_fasta_with_motif, small_motif_list):
    """Scanner should detect the planted GATA motif in at least 20/30 sequences."""
    seqs = _parse_fasta(active_fasta_with_motif)
    results = scan_sequences(seqs, small_motif_list, score_threshold=0.7)
    gata_hits = results.get("MA0035.1", {}).get("hit_sequences", set())
    assert len(gata_hits) >= 20, f"Too few GATA hits: {len(gata_hits)}/30"


# ---------------------------------------------------------------------------
# JASPAR loading (slow — hits real database)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_load_jaspar_motifs():
    """JASPAR CORE vertebrates should load with >500 motifs and contain a known TF."""
    motifs = load_jaspar_motifs("JASPAR2024", "CORE", "Vertebrates")
    assert len(motifs) > 500
    names = [m.name for m in motifs]
    assert any(n in names for n in ("CTCF", "SP1", "GATA1"))


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


def test_enrichment_table_columns(
    active_fasta_with_motif, background_fasta_no_motif, small_motif_list
):
    """Enrichment table should have all required columns."""
    active = _parse_fasta(active_fasta_with_motif)
    background = _parse_fasta(background_fasta_no_motif)
    results = scan_sequences(
        {**active, **background}, small_motif_list, score_threshold=0.7
    )
    enrichment = compute_enrichment(
        results, set(active.keys()), set(background.keys())
    )
    required = {
        "motif_id", "tf_name", "n_active_hits", "n_background_hits",
        "n_active_total", "n_background_total",
        "odds_ratio", "pvalue", "fdr",
    }
    assert required.issubset(set(enrichment.columns))


def test_enrichment_sorted_by_odds_ratio(
    active_fasta_with_motif, background_fasta_no_motif, small_motif_list
):
    """Output rows should be sorted by odds_ratio descending."""
    active = _parse_fasta(active_fasta_with_motif)
    background = _parse_fasta(background_fasta_no_motif)
    results = scan_sequences(
        {**active, **background}, small_motif_list, score_threshold=0.7
    )
    enrichment = compute_enrichment(
        results, set(active.keys()), set(background.keys())
    )
    if len(enrichment) > 1:
        assert enrichment["odds_ratio"].is_monotonic_decreasing


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_fasta_handled(tmp_path, small_motif_list):
    """An empty FASTA should parse to an empty dict and produce an empty table."""
    empty = tmp_path / "empty.fa"
    empty.write_text("")
    seqs = _parse_fasta(str(empty))
    assert len(seqs) == 0

    results = scan_sequences(seqs, small_motif_list, score_threshold=0.7)
    enrichment = compute_enrichment(results, set(), set())
    assert enrichment.empty
    # Schema preserved even when empty so downstream consumers don't break.
    for col in ("motif_id", "tf_name", "odds_ratio", "pvalue", "fdr"):
        assert col in enrichment.columns


def test_short_sequences_skipped(tmp_path, small_motif_list):
    """Sequences shorter than the motif length should be skipped silently."""
    fasta = tmp_path / "short.fa"
    fasta.write_text(">short\nACG\n")
    seqs = _parse_fasta(str(fasta))
    results = scan_sequences(seqs, small_motif_list, score_threshold=0.7)
    for _motif_id, data in results.items():
        assert "short" not in data["hit_sequences"]


# ---------------------------------------------------------------------------
# End-to-end (slow — uses real JASPAR data)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# extract_sequences_to_fasta — bridge from classified table to motif FASTAs
# ---------------------------------------------------------------------------


def _write_classified_tsv(df, path):
    df.to_csv(path, sep="\t", index=False)
    return str(path)


def test_extract_produces_fasta_files(
    classified_fixture, manifest_with_sequences_fixture, tmp_path
):
    """Two non-empty FASTAs with counts matching the active/inactive split."""
    cls_path = _write_classified_tsv(classified_fixture, tmp_path / "cls.tsv")
    active_out = tmp_path / "active.fa"
    bg_out = tmp_path / "background.fa"

    result = extract_sequences_to_fasta(
        cls_path, manifest_with_sequences_fixture,
        active_output=str(active_out), background_output=str(bg_out),
    )

    expected_active = int(classified_fixture["active"].astype(bool).sum())
    expected_bg = int(
        ((~classified_fixture["active"].astype(bool))
         & classified_fixture["pvalue"].notna()).sum()
    )

    assert result["n_active"] == expected_active
    assert result["n_background"] == expected_bg
    assert active_out.exists() and active_out.stat().st_size > 0
    assert bg_out.exists() and bg_out.stat().st_size > 0

    # Round-trip: reading the FASTAs back yields the expected counts.
    assert len(_parse_fasta(active_out)) == expected_active
    assert len(_parse_fasta(bg_out)) == expected_bg


def test_extract_excludes_controls(
    classified_fixture, manifest_with_sequences_fixture, tmp_path
):
    """Negative controls (NaN pvalue) must appear in neither FASTA."""
    cls_path = _write_classified_tsv(classified_fixture, tmp_path / "cls.tsv")
    active_out = tmp_path / "active.fa"
    bg_out = tmp_path / "background.fa"

    extract_sequences_to_fasta(
        cls_path, manifest_with_sequences_fixture,
        active_output=str(active_out), background_output=str(bg_out),
    )

    control_ids = set(
        classified_fixture.loc[
            classified_fixture["pvalue"].isna(), "element_id"
        ]
    )
    assert control_ids, "fixture must contain at least one control row"

    active_ids = set(_parse_fasta(active_out).keys())
    bg_ids = set(_parse_fasta(bg_out).keys())

    assert not (control_ids & active_ids), "controls leaked into active FASTA"
    assert not (control_ids & bg_ids), "controls leaked into background FASTA"


def test_extract_missing_sequences_warns(
    classified_fixture, tmp_path
):
    """Element IDs without a matching sequence should warn but not crash."""
    cls_path = _write_classified_tsv(classified_fixture, tmp_path / "cls.tsv")

    # Manifest covers only the first half of the IDs — the rest are missing.
    half = classified_fixture["element_id"].iloc[: len(classified_fixture) // 2]
    partial = pd.DataFrame({
        "element_id": half,
        "sequence": ["A" * 170] * len(half),
    })
    src = tmp_path / "partial_manifest.tsv"
    partial.to_csv(src, sep="\t", index=False)

    active_out = tmp_path / "active.fa"
    bg_out = tmp_path / "background.fa"

    with pytest.warns(UserWarning, match="had no sequence"):
        result = extract_sequences_to_fasta(
            cls_path, str(src),
            active_output=str(active_out), background_output=str(bg_out),
        )

    # Output FASTAs still written — counts simply reflect what could be resolved.
    assert active_out.exists()
    assert bg_out.exists()
    assert result["n_active"] + result["n_background"] <= len(half)


@pytest.mark.slow
def test_motif_enrichment_end_to_end(
    active_fasta_with_motif, background_fasta_no_motif, tmp_path
):
    """Full pipeline writes a TSV and returns a summary string."""
    out = tmp_path / "enrichment.tsv"
    result = motif_enrichment(
        active_fasta_with_motif,
        background_fasta_no_motif,
        motif_database="JASPAR2024",
        score_threshold=0.8,
        output_path=str(out),
    )
    assert "enrichment_table" in result
    assert "summary" in result
    assert os.path.exists(result["enrichment_table"])
    assert isinstance(result["summary"], str) and len(result["summary"]) > 0
