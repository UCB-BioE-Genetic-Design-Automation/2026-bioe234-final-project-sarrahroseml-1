"""
tests/conftest.py
=================
Shared pytest fixtures for CRE-seq QC tests.

Synthetic data parameters (mirroring a small but realistic CRE-seq library):
  - 500 designed oligos, 84 bp each
  - 10 bp barcodes, ~20 barcodes/oligo  →  ~10,000 rows in mapping table
  - designed_categories: test_element (380), scrambled_control (50),
    motif_knockout (30), positive_control (20), negative_control (20)
  - 30 motif_knockout oligos form 30 families, each with one test_element parent
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_OLIGOS = 500
N_TEST = 380
N_SCRAMBLED = 50
N_KNOCKOUT = 30
N_POS_CTRL = 20
N_NEG_CTRL = 20
OLIGO_LEN = 84
BC_LEN = 10
MEAN_BC_PER_OLIGO = 20
NUCLEOTIDES = ["A", "C", "G", "T"]

OLIGO_IDS = [f"oligo_{i:04d}" for i in range(N_OLIGOS)]


def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def _rand_seq(length: int, rng: np.random.Generator) -> str:
    return "".join(rng.choice(NUCLEOTIDES, size=length))


def _gc(seq: str) -> float:
    return (seq.count("G") + seq.count("C")) / len(seq) if seq else 0.0


def _perfect_cigar_md(oligo_len: int):
    return f"{oligo_len}M", str(oligo_len)


def _mismatch_cigar_md(oligo_len: int, rng: np.random.Generator):
    pos = int(rng.integers(1, oligo_len - 1))
    return f"{oligo_len}M", f"{pos}A{oligo_len - pos - 1}"


# ---------------------------------------------------------------------------
# Design manifest
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def design_manifest_df() -> pd.DataFrame:
    """500-oligo design manifest with categories and variant-family links."""
    rng = _rng(44)
    test_ids = OLIGO_IDS[:N_TEST]
    scrambled_ids = OLIGO_IDS[N_TEST : N_TEST + N_SCRAMBLED]
    knockout_ids = OLIGO_IDS[N_TEST + N_SCRAMBLED : N_TEST + N_SCRAMBLED + N_KNOCKOUT]
    pos_ids = OLIGO_IDS[N_TEST + N_SCRAMBLED + N_KNOCKOUT : N_TEST + N_SCRAMBLED + N_KNOCKOUT + N_POS_CTRL]
    neg_ids = OLIGO_IDS[N_TEST + N_SCRAMBLED + N_KNOCKOUT + N_POS_CTRL :]

    # Each knockout maps to the first N_KNOCKOUT test elements
    parent_map = {ko: test_ids[i] for i, ko in enumerate(knockout_ids)}

    rows = []
    for oligo_id in OLIGO_IDS:
        seq = _rand_seq(OLIGO_LEN, rng)
        if oligo_id in test_ids:
            cat = "test_element"
            parent = None
        elif oligo_id in scrambled_ids:
            cat = "scrambled_control"
            parent = None
        elif oligo_id in knockout_ids:
            cat = "motif_knockout"
            parent = parent_map[oligo_id]
        elif oligo_id in pos_ids:
            cat = "positive_control"
            parent = None
        else:
            cat = "negative_control"
            parent = None

        rows.append(
            {
                "oligo_id": oligo_id,
                "sequence": seq,
                "length": OLIGO_LEN,
                "gc_content": _gc(seq),
                "designed_category": cat,
                "parent_element_id": parent,
            }
        )

    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def design_manifest_path(tmp_path_factory, design_manifest_df) -> str:
    p = tmp_path_factory.mktemp("data") / "design_manifest.tsv"
    design_manifest_df.to_csv(p, sep="\t", index=False)
    return str(p)


# ---------------------------------------------------------------------------
# Mapping table (healthy)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def mapping_df() -> pd.DataFrame:
    """Healthy mapping table: ~20 barcodes/oligo, 85% perfect alignments."""
    rng = _rng(42)
    rows = []
    for oligo_id in OLIGO_IDS:
        n_bc = int(rng.integers(15, 26))  # 15–25 barcodes
        for _ in range(n_bc):
            bc = _rand_seq(BC_LEN, rng)
            n_reads = int(rng.integers(5, 101))
            if rng.random() < 0.85:
                cigar, md = _perfect_cigar_md(OLIGO_LEN)
            else:
                cigar, md = _mismatch_cigar_md(OLIGO_LEN, rng)
            rows.append(
                {
                    "barcode": bc,
                    "oligo_id": oligo_id,
                    "n_reads": n_reads,
                    "cigar": cigar,
                    "md": md,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def mapping_table_path(tmp_path_factory, mapping_df) -> str:
    p = tmp_path_factory.mktemp("data") / "mapping_table.tsv"
    mapping_df.to_csv(p, sep="\t", index=False)
    return str(p)


# ---------------------------------------------------------------------------
# Plasmid count table
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def plasmid_df(mapping_df) -> pd.DataFrame:
    """Plasmid count table derived from the mapping table."""
    rng = _rng(43)
    df = mapping_df[["barcode", "oligo_id"]].copy()
    # Use a narrow count range so per-oligo Gini stays < 0.30 (healthy library)
    df["dna_count"] = rng.integers(50, 151, size=len(df)).astype(int)
    return df


@pytest.fixture(scope="session")
def plasmid_count_path(tmp_path_factory, plasmid_df) -> str:
    p = tmp_path_factory.mktemp("data") / "plasmid_counts.tsv"
    plasmid_df.to_csv(p, sep="\t", index=False)
    return str(p)


# ---------------------------------------------------------------------------
# Edge-case fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def long_barcode_mapping_path(tmp_path_factory) -> str:
    """Mapping table with 15 bp barcodes (outside CRE-seq 8–12 bp window)."""
    rng = _rng(50)
    rows = []
    for oligo_id in OLIGO_IDS[:50]:
        for _ in range(20):
            bc = _rand_seq(15, rng)  # lentiMPRA-length barcodes
            cigar, md = _perfect_cigar_md(OLIGO_LEN)
            rows.append(
                {
                    "barcode": bc,
                    "oligo_id": oligo_id,
                    "n_reads": 20,
                    "cigar": cigar,
                    "md": md,
                }
            )
    p = tmp_path_factory.mktemp("data") / "long_bc_mapping.tsv"
    pd.DataFrame(rows).to_csv(p, sep="\t", index=False)
    return str(p)


@pytest.fixture(scope="session")
def long_oligo_manifest_path(tmp_path_factory) -> str:
    """Design manifest with 500 bp oligos (outside CRE-seq 84–200 bp window)."""
    rng = _rng(51)
    rows = [
        {
            "oligo_id": f"oligo_{i:04d}",
            "sequence": _rand_seq(500, rng),
            "length": 500,
            "gc_content": 0.5,
            "designed_category": "test_element",
            "parent_element_id": None,
        }
        for i in range(50)
    ]
    p = tmp_path_factory.mktemp("data") / "long_oligo_manifest.tsv"
    pd.DataFrame(rows).to_csv(p, sep="\t", index=False)
    return str(p)


@pytest.fixture(scope="session")
def missing_ref_mapping_path(tmp_path_factory, design_manifest_df) -> str:
    """
    Mapping table where the reference oligos of several families are absent,
    but their knockouts are present.
    """
    rng = _rng(52)
    knockouts = design_manifest_df[design_manifest_df["designed_category"] == "motif_knockout"]
    parent_ids = set(knockouts["parent_element_id"].dropna().unique())

    rows = []
    for oligo_id in OLIGO_IDS:
        if oligo_id in parent_ids:
            continue  # drop all reference oligos → families lack their reference
        for _ in range(15):
            bc = _rand_seq(BC_LEN, rng)
            cigar, md = _perfect_cigar_md(OLIGO_LEN)
            rows.append(
                {
                    "barcode": bc,
                    "oligo_id": oligo_id,
                    "n_reads": 20,
                    "cigar": cigar,
                    "md": md,
                }
            )
    p = tmp_path_factory.mktemp("data") / "missing_ref_mapping.tsv"
    pd.DataFrame(rows).to_csv(p, sep="\t", index=False)
    return str(p)


@pytest.fixture(scope="session")
def low_recovery_mapping_path(tmp_path_factory, design_manifest_df) -> str:
    """Mapping table where only 50% of oligos are present (should fail oligo_recovery)."""
    rng = _rng(53)
    recovered = set(OLIGO_IDS[::2])  # every other oligo
    rows = []
    for oligo_id in OLIGO_IDS:
        if oligo_id not in recovered:
            continue
        for _ in range(12):
            bc = _rand_seq(BC_LEN, rng)
            cigar, md = _perfect_cigar_md(OLIGO_LEN)
            rows.append(
                {
                    "barcode": bc,
                    "oligo_id": oligo_id,
                    "n_reads": 20,
                    "cigar": cigar,
                    "md": md,
                }
            )
    p = tmp_path_factory.mktemp("data") / "low_recovery_mapping.tsv"
    pd.DataFrame(rows).to_csv(p, sep="\t", index=False)
    return str(p)


@pytest.fixture(scope="session")
def empty_mapping_path(tmp_path_factory) -> str:
    """Empty mapping table (header only)."""
    p = tmp_path_factory.mktemp("data") / "empty_mapping.tsv"
    pd.DataFrame(
        columns=["barcode", "oligo_id", "n_reads", "cigar", "md"]
    ).to_csv(p, sep="\t", index=False)
    return str(p)
