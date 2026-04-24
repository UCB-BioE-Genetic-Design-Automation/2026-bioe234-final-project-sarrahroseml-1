"""
tests/stats/test_library.py
============================
Tests for creseq_mcp/stats/library.py.

Coverage:
  - call_active_elements: happy path, fallback background, low-control warning
  - rank_cre_candidates: rank ordering, top_element, q_col fallback
  - motif_enrichment_summary: missing column raises ValueError
  - search_pubmed: mocked HTTP response returns DataFrame with title column
  - search_jaspar_motif: mocked HTTP response returns matrix_id
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from creseq_mcp.stats.library import (
    call_active_elements,
    motif_enrichment_summary,
    rank_cre_candidates,
    search_jaspar_motif,
    search_pubmed,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _activity_tsv(tmp_path: Path, n=60, seed=7) -> str:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        if i < 20:
            cat, base = "negative_control", -0.4
        elif i < 30:
            cat, base = "positive_control", 3.0
        else:
            cat, base = "test_element", float(rng.choice([-0.2, 2.5]))
        rows.append({
            "oligo_id": f"O{i:03d}",
            "log2_ratio": base + float(rng.normal(0, 0.15)),
            "designed_category": cat,
        })
    p = tmp_path / "activity_results.tsv"
    pd.DataFrame(rows).to_csv(p, sep="\t", index=False)
    return str(p)


# ---------------------------------------------------------------------------
# call_active_elements
# ---------------------------------------------------------------------------

class TestCallActiveElements:
    def test_happy_path(self, tmp_path):
        p = _activity_tsv(tmp_path)
        df, summary = call_active_elements(p, activity_col="log2_ratio")

        assert "active" in df.columns
        assert "q_value" in df.columns
        assert summary["n_active"] > 0
        assert 0.0 <= summary["fraction_active"] <= 1.0

    def test_fallback_no_neg_ctrl(self, tmp_path):
        rows = [{"oligo_id": f"O{i}", "log2_ratio": float(i * 0.2)} for i in range(20)]
        p = tmp_path / "act.tsv"
        pd.DataFrame(rows).to_csv(p, sep="\t", index=False)

        df, summary = call_active_elements(str(p), activity_col="log2_ratio")
        # fallback uses bottom 20% as background, should still produce active calls
        assert "active" in df.columns

    def test_low_control_warning(self, tmp_path):
        rows = [
            {"oligo_id": "N1", "log2_ratio": -0.3, "designed_category": "negative_control"},
            {"oligo_id": "N2", "log2_ratio": -0.2, "designed_category": "negative_control"},
            {"oligo_id": "T1", "log2_ratio": 2.5, "designed_category": "test_element"},
        ]
        p = tmp_path / "few.tsv"
        pd.DataFrame(rows).to_csv(p, sep="\t", index=False)

        _, summary = call_active_elements(str(p), activity_col="log2_ratio")
        assert any("fewer than 5" in w.lower() for w in summary["warnings"])


# ---------------------------------------------------------------------------
# rank_cre_candidates
# ---------------------------------------------------------------------------

class TestRankCrCandidates:
    def test_rank_ordering(self, tmp_path):
        p = _activity_tsv(tmp_path)
        df, summary = rank_cre_candidates(p, top_n=10, activity_col="log2_ratio", q_col="q_value")

        assert list(df["rank"]) == list(range(1, len(df) + 1))
        assert len(df) == 10
        assert summary["top_element"] is not None

    def test_missing_q_col_defaults_to_one(self, tmp_path):
        # Table has no q_value column → should not raise, q defaults to 1.0
        rows = [{"oligo_id": f"O{i}", "log2_ratio": float(i)} for i in range(20)]
        p = tmp_path / "no_q.tsv"
        pd.DataFrame(rows).to_csv(p, sep="\t", index=False)

        df, summary = rank_cre_candidates(str(p), top_n=5, activity_col="log2_ratio", q_col="q_value")
        assert len(df) == 5
        assert (df["q_value"] == 1.0).all()

    def test_top_n_larger_than_table(self, tmp_path):
        rows = [{"oligo_id": f"O{i}", "log2_ratio": float(i)} for i in range(5)]
        p = tmp_path / "small.tsv"
        pd.DataFrame(rows).to_csv(p, sep="\t", index=False)

        df, summary = rank_cre_candidates(str(p), top_n=50, activity_col="log2_ratio")
        assert len(df) == 5


# ---------------------------------------------------------------------------
# motif_enrichment_summary
# ---------------------------------------------------------------------------

class TestMotifEnrichmentSummary:
    def test_missing_motif_col_raises(self, tmp_path):
        rows = [{"oligo_id": "O1", "log2_ratio": 1.0, "active": True}]
        p = tmp_path / "no_motif.tsv"
        pd.DataFrame(rows).to_csv(p, sep="\t", index=False)

        with pytest.raises(ValueError, match="top_motif"):
            motif_enrichment_summary(str(p))

    def test_happy_path(self, tmp_path):
        rows = [
            {"oligo_id": f"O{i}", "active": i % 2 == 0, "top_motif": "SP1" if i % 3 == 0 else "GATA1"}
            for i in range(30)
        ]
        p = tmp_path / "motifs.tsv"
        pd.DataFrame(rows).to_csv(p, sep="\t", index=False)

        df, summary = motif_enrichment_summary(str(p))
        assert "enrichment_ratio" in df.columns
        assert summary["n_motifs_tested"] == 2


# ---------------------------------------------------------------------------
# search_pubmed (mocked)
# ---------------------------------------------------------------------------

class TestSearchPubmed:
    def _mock_response(self, ids, summaries):
        esearch_data = {"esearchresult": {"idlist": ids}}
        esummary_data = {"result": summaries}

        call_count = 0

        def fake_get(url, params=None, timeout=15, headers=None):
            nonlocal call_count
            mock = MagicMock()
            mock.raise_for_status = MagicMock()
            if call_count == 0:
                mock.json.return_value = esearch_data
            else:
                mock.json.return_value = esummary_data
            call_count += 1
            return mock

        return fake_get

    def test_returns_dataframe_with_title(self):
        ids = ["12345678"]
        summaries = {
            "12345678": {
                "title": "GATA1 enhancer MPRA study",
                "fulljournalname": "Nature Genetics",
                "pubdate": "2024",
                "authors": [{"name": "Smith A"}],
            }
        }
        fake_get = self._mock_response(ids, summaries)

        with patch("creseq_mcp.stats.library.requests.get", side_effect=fake_get), \
             patch("creseq_mcp.stats.library.time.sleep"):
            df, summary = search_pubmed("GATA1 enhancer MPRA", max_results=1)

        assert len(df) == 1
        assert df.iloc[0]["title"] == "GATA1 enhancer MPRA study"
        assert summary["n_results"] == 1

    def test_no_results(self):
        fake_get = self._mock_response([], {})

        with patch("creseq_mcp.stats.library.requests.get", side_effect=fake_get), \
             patch("creseq_mcp.stats.library.time.sleep"):
            df, summary = search_pubmed("xyzzy gibberish query", max_results=5)

        assert len(df) == 0
        assert summary["n_results"] == 0


# ---------------------------------------------------------------------------
# search_jaspar_motif (mocked)
# ---------------------------------------------------------------------------

class TestSearchJasparMotif:
    def test_returns_matrix_id(self):
        mock_data = {
            "results": [
                {
                    "matrix_id": "MA0139.1",
                    "name": "CTCF",
                    "collection": "CORE",
                    "tax_group": "vertebrates",
                    "species": [{"name": "Homo sapiens"}],
                    "class": ["C2H2 zinc finger factors"],
                    "family": ["CTCF-related"],
                }
            ]
        }

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = mock_data

        with patch("creseq_mcp.stats.library.requests.get", return_value=mock_resp):
            df, summary = search_jaspar_motif("CTCF")

        assert len(df) == 1
        assert df.iloc[0]["matrix_id"] == "MA0139.1"
        assert summary["n_results"] == 1

    def test_no_results_warning(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"results": []}

        with patch("creseq_mcp.stats.library.requests.get", return_value=mock_resp):
            df, summary = search_jaspar_motif("NOTAREALFACTOR999")

        assert len(df) == 0
        assert any("No JASPAR" in w for w in summary["warnings"])
