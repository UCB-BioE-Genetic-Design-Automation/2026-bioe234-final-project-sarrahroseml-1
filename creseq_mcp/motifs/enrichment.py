"""
creseq_mcp/motif.py
===================
TF binding-motif enrichment for CRE-seq active vs. background sequence sets.

Pipeline
--------
1. Load PWMs from JASPAR via ``pyjaspar`` (no network required after install —
   pyjaspar bundles a SQLite database).
2. Score each sequence against each PWM on both strands; record presence/
   absence using a relative score threshold.
3. Build a per-motif 2×2 contingency table (active × motif-present) and run
   one-sided Fisher's exact test (alternative='greater') with BH-FDR.

Output schema
-------------
``[motif_id, tf_name, n_active_hits, n_background_hits, n_active_total,
n_background_total, odds_ratio, pvalue, fdr]``
sorted by ``odds_ratio`` descending; infinite ORs capped at 999.0.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _write_fasta(records: list[tuple[str, str]], path: str | Path) -> None:
    """Write ``[(id, seq), ...]`` to a FASTA file (one entry per record)."""
    with open(path, "w") as fh:
        for record_id, seq in records:
            fh.write(f">{record_id}\n{seq}\n")


def extract_sequences_to_fasta(
    classified_table: str | Path,
    sequence_source: str | Path,
    active_output: str | Path = "active.fa",
    background_output: str | Path = "background.fa",
) -> dict:
    """
    Bridge ``call_active_elements`` → ``motif_enrichment``.

    Splits a classified-elements TSV into two FASTAs ready for motif scanning:
      - ``active``: rows where ``active == True``
      - ``background``: rows where ``active == False`` AND ``pvalue`` is not
        NaN (i.e., inactive *test* elements — negative controls, which carry
        NaN pvalues, are excluded so they don't pollute the background null).

    Sequences are pulled from *sequence_source* (a TSV with ``element_id`` and
    ``sequence`` columns).  Element IDs missing from the source are warned
    about and skipped (no crash).

    Returns
    -------
    dict with keys ``active_fasta``, ``background_fasta``, ``n_active``,
    ``n_background``.
    """
    classified = pd.read_csv(classified_table, sep="\t")
    sources = pd.read_csv(sequence_source, sep="\t")

    # Normalize oligo_id → element_id in both tables
    if "element_id" not in classified.columns and "oligo_id" in classified.columns:
        classified = classified.rename(columns={"oligo_id": "element_id"})
    if "element_id" not in sources.columns and "oligo_id" in sources.columns:
        sources = sources.rename(columns={"oligo_id": "element_id"})

    for col in ("element_id", "active", "pvalue"):
        if col not in classified.columns:
            raise ValueError(
                f"classified_table missing required column {col!r}; "
                f"got {list(classified.columns)}"
            )
    for col in ("element_id", "sequence"):
        if col not in sources.columns:
            raise ValueError(
                f"sequence_source missing required column {col!r}; "
                f"got {list(sources.columns)}"
            )

    seq_lookup = dict(
        zip(sources["element_id"].astype(str), sources["sequence"].astype(str))
    )

    is_active = classified["active"].astype(bool)
    is_test = classified["pvalue"].notna()
    active_ids = classified.loc[is_active, "element_id"].astype(str).tolist()
    background_ids = classified.loc[is_test & ~is_active, "element_id"].astype(str).tolist()

    def _resolve(ids: list[str]) -> tuple[list[tuple[str, str]], list[str]]:
        kept: list[tuple[str, str]] = []
        missing: list[str] = []
        for eid in ids:
            seq = seq_lookup.get(eid)
            if seq is None or (isinstance(seq, float) and pd.isna(seq)):
                missing.append(eid)
                continue
            kept.append((eid, seq))
        return kept, missing

    active_records, active_missing = _resolve(active_ids)
    background_records, background_missing = _resolve(background_ids)

    total_missing = len(active_missing) + len(background_missing)
    if total_missing:
        sample = (active_missing + background_missing)[:5]
        warnings.warn(
            f"{total_missing} element_id(s) had no sequence in the source "
            f"and were skipped. Examples: {sample}",
            UserWarning,
            stacklevel=2,
        )

    _write_fasta(active_records, active_output)
    _write_fasta(background_records, background_output)

    return {
        "active_fasta": str(active_output),
        "background_fasta": str(background_output),
        "n_active": len(active_records),
        "n_background": len(background_records),
    }


def _parse_fasta(fasta_path: str | Path) -> dict[str, str]:
    """Parse a FASTA file into ``{record_id: sequence}`` (uppercase)."""
    from Bio import SeqIO

    sequences: dict[str, str] = {}
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        sequences[record.id] = str(record.seq).upper()
    return sequences


# ---------------------------------------------------------------------------
# JASPAR loading
# ---------------------------------------------------------------------------


def load_jaspar_motifs(
    release: str = "JASPAR2024",
    collection: str = "CORE",
    tax_group: str = "Vertebrates",
) -> list[Any]:
    """
    Load JASPAR PWM profiles via pyjaspar.

    Returns a list of ``Bio.motifs.jaspar.Motif`` objects exposing
    ``.matrix_id``, ``.name``, and ``.counts``.
    """
    from pyjaspar import jaspardb

    jdb = jaspardb(release=release)
    return jdb.fetch_motifs(
        collection=[collection],
        tax_group=[tax_group],
        all_versions=False,
    )


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------


def _build_pssm(jaspar_motif: Any):
    """Build a PSSM (log-odds PWM) from a JASPAR/mock motif's counts dict."""
    from Bio import motifs as bio_motifs

    m = bio_motifs.Motif(alphabet="ACGT", counts=jaspar_motif.counts)
    pwm = m.counts.normalize(pseudocounts=0.5)
    return pwm.log_odds()


def scan_sequences(
    sequences: dict[str, str],
    motif_list: list[Any],
    score_threshold: float = 0.8,
) -> dict[str, dict[str, Any]]:
    """
    Scan each sequence against each motif on both strands.

    A sequence is recorded as a "hit" for a motif if any forward- or
    reverse-strand position scores at or above
    ``min_score + score_threshold * (max_score - min_score)``.

    Parameters
    ----------
    sequences : ``{seq_id: ACGT-string}``
    motif_list : list of motifs with ``.matrix_id``, ``.name``, ``.counts``
    score_threshold : float in [0, 1], fraction of the dynamic range above
        the minimum PSSM score.  0.8 follows the JASPAR recommendation.

    Returns
    -------
    ``{motif_id: {"tf_name": str, "hit_sequences": set[str]}}``
    """
    from Bio.Seq import Seq

    results: dict[str, dict[str, Any]] = {}

    for jaspar_motif in motif_list:
        motif_id = jaspar_motif.matrix_id
        tf_name = jaspar_motif.name

        pssm = _build_pssm(jaspar_motif)
        max_score = float(pssm.max)
        min_score = float(pssm.min)
        threshold = min_score + score_threshold * (max_score - min_score)
        motif_len = pssm.length

        hit_seqs: set[str] = set()
        for seq_id, seq_str in sequences.items():
            if len(seq_str) < motif_len:
                continue

            try:
                seq_obj = Seq(seq_str)
            except Exception:
                continue

            hit = False
            # Forward strand
            try:
                for _pos, _score in pssm.search(
                    seq_obj, threshold=threshold, both=False
                ):
                    hit = True
                    break  # presence/absence — one hit is enough
            except Exception:
                # Non-ACGT character or other issue — skip silently.
                pass

            # Reverse complement (only if no forward hit yet)
            if not hit:
                try:
                    rc_seq = seq_obj.reverse_complement()
                    for _pos, _score in pssm.search(
                        rc_seq, threshold=threshold, both=False
                    ):
                        hit = True
                        break
                except Exception:
                    pass

            if hit:
                hit_seqs.add(seq_id)

        results[motif_id] = {
            "tf_name": tf_name,
            "hit_sequences": hit_seqs,
        }

    return results


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------


_ENRICHMENT_COLUMNS = [
    "motif_id",
    "tf_name",
    "n_active_hits",
    "n_background_hits",
    "n_active_total",
    "n_background_total",
    "odds_ratio",
    "pvalue",
    "fdr",
]


def compute_enrichment(
    scan_results: dict[str, dict[str, Any]],
    active_ids: set[str],
    background_ids: set[str],
) -> pd.DataFrame:
    """
    For each motif, run a one-sided Fisher's exact test for enrichment in
    active vs. background sequences and apply BH-FDR.

    Motifs with zero hits in both sets are dropped.  Infinite odds ratios
    (zero cells) are capped at 999.0 for clean output.  Output is sorted by
    ``odds_ratio`` descending.
    """
    n_active = len(active_ids)
    n_background = len(background_ids)

    rows: list[dict[str, Any]] = []
    for motif_id, data in scan_results.items():
        hit_seqs = data["hit_sequences"]
        a = len(hit_seqs & active_ids)
        c = len(hit_seqs & background_ids)
        b = n_active - a
        d = n_background - c

        if a + c == 0:
            continue

        odds_ratio, pvalue = fisher_exact(
            [[a, b], [c, d]], alternative="greater"
        )
        if not np.isfinite(odds_ratio):
            odds_ratio = 999.0

        rows.append({
            "motif_id": motif_id,
            "tf_name": data["tf_name"],
            "n_active_hits": int(a),
            "n_background_hits": int(c),
            "n_active_total": int(n_active),
            "n_background_total": int(n_background),
            "odds_ratio": float(odds_ratio),
            "pvalue": float(pvalue),
        })

    if not rows:
        return pd.DataFrame(columns=_ENRICHMENT_COLUMNS)

    df = pd.DataFrame(rows)
    _, fdr_values, _, _ = multipletests(df["pvalue"].to_numpy(), method="fdr_bh")
    df["fdr"] = fdr_values

    df = df.sort_values(["fdr", "odds_ratio"], ascending=[True, False]).reset_index(drop=True)
    return df[_ENRICHMENT_COLUMNS]


# ---------------------------------------------------------------------------
# MCP entry point
# ---------------------------------------------------------------------------


def motif_enrichment(
    active_fasta: str | Path,
    background_fasta: str | Path,
    motif_database: str = "JASPAR2024",
    collection: str = "CORE",
    tax_group: str = "Vertebrates",
    score_threshold: float = 0.8,
    output_path: str | Path | None = None,
) -> dict:
    """
    Full motif-enrichment pipeline.

    Loads JASPAR motifs, parses both FASTAs, scans every sequence on both
    strands, computes per-motif Fisher enrichment + BH-FDR, writes the table
    to ``<active_fasta>_motif_enrichment.tsv`` (or *output_path*), and returns
    a summary of the top 10 significant motifs (or a "nothing significant"
    note).
    """
    motifs = load_jaspar_motifs(
        release=motif_database, collection=collection, tax_group=tax_group
    )

    active_seqs = _parse_fasta(active_fasta)
    background_seqs = _parse_fasta(background_fasta)
    all_seqs = {**active_seqs, **background_seqs}

    scan_results = scan_sequences(all_seqs, motifs, score_threshold=score_threshold)

    enrichment_df = compute_enrichment(
        scan_results,
        active_ids=set(active_seqs.keys()),
        background_ids=set(background_seqs.keys()),
    )

    if output_path is None:
        active_path = Path(active_fasta)
        out_path = active_path.with_name(active_path.stem + "_motif_enrichment.tsv")
    else:
        out_path = Path(output_path)
    enrichment_df.to_csv(out_path, sep="\t", index=False)

    sig = enrichment_df[enrichment_df["fdr"] < 0.05]
    if len(sig):
        top = sig.head(10)
        lines = [
            f"  {row.tf_name}: OR={row.odds_ratio:.1f}, FDR={row.fdr:.1e}, "
            f"hits={row.n_active_hits}/{row.n_active_total}"
            for _, row in top.iterrows()
        ]
        summary = (
            f"{len(sig)} motifs enriched at FDR < 0.05. "
            f"Top {len(top)}:\n" + "\n".join(lines)
        )
    else:
        summary = "No motifs reached FDR < 0.05 significance."

    return {
        "enrichment_table": str(out_path),
        "summary": summary,
    }
