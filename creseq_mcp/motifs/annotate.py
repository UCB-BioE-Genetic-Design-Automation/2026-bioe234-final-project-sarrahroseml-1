"""
creseq_mcp/qc/motifs.py
========================
Lightweight PWM-based motif annotation for CRE-seq oligo sequences.

Fetches position frequency matrices from JASPAR REST API, converts to
log-odds PWMs, and scans each oligo sequence to assign a top_motif column.

Default TF list is curated for liver/HepG2 relevance (the primary CRE-seq
cell type in the Ahituv lab), but any list of JASPAR-searchable TF names
can be passed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

_JASPAR_URL = "https://jaspar.elixir.no/api/v1/matrix/"

_DEFAULT_TFS = [
    "HNF4A", "FOXA2", "FOXA1", "HNF1A", "CEBPA",
    "GATA4", "SP1", "SP3", "NRF2", "CTCF",
    "ATF3", "JUNB", "FOS", "MYC", "MAX",
    "E2F1", "TP53", "KLF4", "YY1", "RXRA",
]


# ---------------------------------------------------------------------------
# JASPAR fetch helpers
# ---------------------------------------------------------------------------

def _fetch_jaspar_pfm(tf_name: str, species: int = 9606) -> dict[str, Any] | None:
    """Return the first JASPAR PFM for tf_name, or None on failure."""
    try:
        resp = requests.get(
            _JASPAR_URL,
            params={"search": tf_name, "species": species, "collection": "CORE",
                    "format": "json", "page_size": 1},
            timeout=10,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            return None
        matrix_id = results[0].get("matrix_id")
        if not matrix_id:
            return None
        detail = requests.get(
            f"{_JASPAR_URL}{matrix_id}/",
            timeout=10,
            headers={"Accept": "application/json"},
        )
        detail.raise_for_status()
        data = detail.json()
        pfm = data.get("pfm") or data.get("matrix")
        if not pfm:
            return None
        return {"tf_name": tf_name, "matrix_id": matrix_id, "pfm": pfm}
    except Exception as exc:
        logger.debug("JASPAR fetch failed for %s: %s", tf_name, exc)
        return None


def _pfm_to_pwm(pfm: dict[str, list[float]], pseudocount: float = 0.1) -> np.ndarray:
    """
    Convert a JASPAR PFM dict {A:[...], C:[...], G:[...], T:[...]}
    to a (4, L) log-odds PWM (background = 0.25 uniform).
    Returns shape (4, L), rows in ACGT order.
    """
    keys = ["A", "C", "G", "T"]
    matrix = np.array([pfm[k] for k in keys], dtype=float)  # (4, L)
    matrix += pseudocount
    col_sums = matrix.sum(axis=0, keepdims=True)
    freq = matrix / col_sums
    pwm = np.log2(freq / 0.25)
    return pwm


def _scan_sequence(seq: str, pwm: np.ndarray) -> float:
    """Return the maximum PWM score over all positions in seq (both strands)."""
    _nt = {"A": 0, "C": 1, "G": 2, "T": 3}
    _comp = {"A": "T", "T": "A", "C": "G", "G": "C"}
    L = pwm.shape[1]
    n = len(seq)
    if n < L:
        return float("-inf")

    best = float("-inf")
    for strand_seq in [seq, "".join(_comp.get(b, "N") for b in reversed(seq))]:
        for start in range(n - L + 1):
            window = strand_seq[start: start + L]
            score = 0.0
            valid = True
            for pos, base in enumerate(window):
                idx = _nt.get(base)
                if idx is None:
                    valid = False
                    break
                score += pwm[idx, pos]
            if valid and score > best:
                best = score
    return best


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def annotate_top_motifs(
    activity_results_path: str | Path,
    design_manifest_path: str | Path,
    tf_names: list[str] | None = None,
    upload_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Annotate each oligo with its highest-scoring JASPAR motif.

    Fetches PWMs from JASPAR for each TF in tf_names, scans oligo sequences,
    assigns top_motif = name of the TF with the best PWM score.
    Updates activity_results.tsv in-place (adds top_motif column).

    Parameters
    ----------
    activity_results_path:
        Path to activity_results.tsv (must have oligo_id column).
    design_manifest_path:
        Path to design_manifest.tsv (must have oligo_id + sequence columns).
    tf_names:
        List of JASPAR TF names. Defaults to _DEFAULT_TFS (HepG2-relevant).
    upload_dir:
        If provided, saves updated activity_results.tsv there.
    """
    if tf_names is None:
        tf_names = _DEFAULT_TFS

    activity_df = pd.read_csv(activity_results_path, sep="\t")
    manifest_df = pd.read_csv(design_manifest_path, sep="\t")

    if "sequence" not in manifest_df.columns:
        raise ValueError("design_manifest.tsv must have a 'sequence' column for motif scanning.")

    seq_map = dict(zip(manifest_df["oligo_id"], manifest_df["sequence"]))

    # Fetch PWMs
    pwms: dict[str, np.ndarray] = {}
    fetched, failed = 0, 0
    for tf in tf_names:
        result = _fetch_jaspar_pfm(tf)
        if result is not None:
            pwms[tf] = _pfm_to_pwm(result["pfm"])
            fetched += 1
        else:
            failed += 1
            logger.warning("Could not fetch JASPAR PWM for %s", tf)

    if not pwms:
        return activity_df, {
            "n_tfs_fetched": 0,
            "n_tfs_failed": failed,
            "warnings": ["No JASPAR PWMs could be fetched. Check network or tf_names."],
            "pass": False,
        }

    # Scan sequences
    pwm_tfs = list(pwms.keys())
    top_motifs = []
    for oid in activity_df["oligo_id"]:
        seq = seq_map.get(oid, "")
        if not seq:
            top_motifs.append(None)
            continue
        scores = {tf: _scan_sequence(seq.upper(), pwms[tf]) for tf in pwm_tfs}
        top_motifs.append(max(scores, key=scores.get))

    activity_df["top_motif"] = top_motifs

    if upload_dir is not None:
        out = Path(upload_dir) / "activity_results.tsv"
        activity_df.to_csv(out, sep="\t", index=False)

    n_annotated = int(pd.Series(top_motifs).notna().sum())
    motif_counts = pd.Series(top_motifs).dropna().value_counts().to_dict()

    return activity_df, {
        "n_tfs_fetched": fetched,
        "n_tfs_failed": failed,
        "n_oligos_annotated": n_annotated,
        "top_motif_counts": motif_counts,
        "warnings": [f"Could not fetch PWM for {failed} TF(s)."] if failed else [],
        "pass": fetched > 0,
    }
