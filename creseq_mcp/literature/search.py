"""
creseq_mcp/stats/library.py
===========================

Stats and interpretation tools for CRE-seq / MPRA activity data.

This module operates AFTER library QC. It takes barcode-level or element-level
DNA/RNA count tables, computes log2 RNA/DNA activity, calls active elements,
ranks CRE candidates, and prepares top hits/motifs for downstream RAG-style
literature interpretation.

All public functions return:
    tuple[pd.DataFrame, dict]

This matches the existing QC tool convention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import time
import requests


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Path does not exist: {path}")

    name = path.name.lower()

    if name.endswith(".tsv") or name.endswith(".txt"):
        return pd.read_csv(path, sep="\t")
    if name.endswith(".tsv.gz") or name.endswith(".txt.gz"):
        return pd.read_csv(path, sep="\t", compression="gzip")
    if name.endswith(".csv"):
        return pd.read_csv(path)
    if name.endswith(".csv.gz"):
        return pd.read_csv(path, compression="gzip")

    raise ValueError(f"Unsupported file type: {path.suffix}. Use CSV/TSV/TXT, optionally gzipped.")

def _check_cols(df: pd.DataFrame, required: set[str], source: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{source}: missing required columns {missing}. "
            f"Found columns: {list(df.columns)}"
        )



# ---------------------------------------------------------------------------
# rank_cre_candidates
# ---------------------------------------------------------------------------

def rank_cre_candidates(
    activity_table_path: str | Path,
    top_n: int = 20,
    activity_col: str = "log2_ratio",
    q_col: str = "fdr",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Rank CREs by activity strength and statistical confidence.

    Score logic:
        high activity is good
        low q-value is good
        low DNA coverage is penalized if present
    """

    df = _read_table(activity_table_path)
    _check_cols(df, {activity_col}, str(activity_table_path))

    result = df.copy()

    if q_col not in result.columns:
        result[q_col] = 1.0

    result[activity_col] = pd.to_numeric(result[activity_col], errors="coerce").fillna(0)
    result[q_col] = pd.to_numeric(result[q_col], errors="coerce").fillna(1.0).clip(1e-12, 1)

    result["confidence_score"] = -np.log10(result[q_col])
    result["rank_score"] = result[activity_col] + 0.25 * result["confidence_score"]

    if "low_dna_coverage" in result.columns:
        result.loc[result["low_dna_coverage"] == True, "rank_score"] -= 1.0

    result = result.sort_values("rank_score", ascending=False).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)

    top = result.head(top_n).copy()

    summary = {
        "n_ranked": int(len(result)),
        "top_n": int(top_n),
        "top_element": str(top.iloc[0].get("element_id", top.iloc[0].get("oligo_id", "unknown")))
        if len(top)
        else None,
        "median_top_activity": float(top[activity_col].median()) if len(top) else None,
        "warnings": [],
        "pass": True,
    }

    return top, summary


# ---------------------------------------------------------------------------
# Tool 4: motif_enrichment_summary
# ---------------------------------------------------------------------------

def motif_enrichment_summary(
    activity_table_path: str | Path,
    motif_col: str = "top_motif",
    active_col: str = "active",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Lightweight motif enrichment summary.

    This does not run MEME/HOMER/FIMO. It summarizes motifs already annotated
    in the table and computes active-vs-inactive enrichment ratios.
    """

    df = _read_table(activity_table_path)
    _check_cols(df, {motif_col, active_col}, str(activity_table_path))

    rows = []
    motifs = sorted(m for m in df[motif_col].dropna().unique() if str(m).lower() != "none")

    active_df = df[df[active_col] == True]
    inactive_df = df[df[active_col] == False]

    for motif in motifs:
        active_rate = float((active_df[motif_col] == motif).mean()) if len(active_df) else 0.0
        inactive_rate = float((inactive_df[motif_col] == motif).mean()) if len(inactive_df) else 0.0

        enrichment = (active_rate + 1e-6) / (inactive_rate + 1e-6)

        rows.append(
            {
                "motif": motif,
                "active_fraction": active_rate,
                "inactive_fraction": inactive_rate,
                "enrichment_ratio": enrichment,
                "n_active_with_motif": int((active_df[motif_col] == motif).sum()),
                "n_inactive_with_motif": int((inactive_df[motif_col] == motif).sum()),
            }
        )

    result = pd.DataFrame(rows).sort_values("enrichment_ratio", ascending=False)

    summary = {
        "n_motifs_tested": int(len(result)),
        "top_enriched_motif": str(result.iloc[0]["motif"]) if len(result) else None,
        "warnings": [],
        "pass": True,
    }

    return result, summary


# ---------------------------------------------------------------------------
# Tool 5: prepare_rag_context
# ---------------------------------------------------------------------------

def prepare_rag_context(
    ranked_table_path: str | Path,
    top_n: int = 10,
    motif_col: str = "top_motif",
    target_cell_type: str | None = None,
    off_target_cell_type: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Prepare a compact table that an LLM/RAG layer can use for literature search.

    This does not call external APIs directly. It creates suggested search queries
    for PubMed/JASPAR/ENCODE-style lookups.
    """

    df = _read_table(ranked_table_path)
    top = df.head(top_n).copy()

    queries = []

    if motif_col in top.columns:
        motifs = [m for m in top[motif_col].dropna().unique() if str(m).lower() != "none"]
        for motif in motifs:
            if target_cell_type and off_target_cell_type:
                queries.append(
                    f"{motif} transcription factor {target_cell_type} enhancer activity "
                    f"off target {off_target_cell_type}"
                )
            elif target_cell_type:
                queries.append(f"{motif} transcription factor {target_cell_type} enhancer activity")
            else:
                queries.append(f"{motif} transcription factor enhancer MPRA CRE-seq")

    top["rag_search_terms"] = "; ".join(queries[:5]) if queries else ""

    summary = {
        "n_top_elements": int(len(top)),
        "suggested_queries": queries[:10],
        "target_cell_type": target_cell_type,
        "off_target_cell_type": off_target_cell_type,
        "warnings": [],
        "pass": True,
    }

    return top, summary



# ---------------------------------------------------------------------------
# Literature/API retrieval helpers
# ---------------------------------------------------------------------------

def _safe_get_json(
    url: str,
    params: dict[str, Any] | None = None,
    timeout: int = 15,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Small helper for GET-based JSON APIs.

    Returns a dictionary with either:
        {"ok": True, "data": ...}
    or:
        {"ok": False, "error": "...", "url": "..."}
    """

    try:
        response = requests.get(
            url,
            params=params,
            timeout=timeout,
            headers=headers or {"Accept": "application/json"},
        )
        response.raise_for_status()
        return {"ok": True, "data": response.json()}
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "url": url,
            "params": params or {},
        }


def search_pubmed(
    query: str,
    max_results: int = 5,
    email: str | None = None,
    api_key: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Search PubMed using NCBI E-utilities.

    API flow:
        1. ESearch finds PubMed IDs for the query.
        2. ESummary retrieves title/journal/date metadata for those IDs.

    Notes:
        - email is recommended by NCBI for responsible API use.
        - api_key is optional but helps with rate limits.
    """

    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    esearch_params: dict[str, Any] = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": max_results,
        "sort": "relevance",
    }

    if email:
        esearch_params["email"] = email
    if api_key:
        esearch_params["api_key"] = api_key

    esearch = _safe_get_json(f"{base}/esearch.fcgi", esearch_params)

    if not esearch["ok"]:
        return pd.DataFrame(), {
            "query": query,
            "n_results": 0,
            "warnings": [f"PubMed ESearch failed: {esearch['error']}"],
            "pass": False,
        }

    ids = esearch["data"].get("esearchresult", {}).get("idlist", [])

    if not ids:
        return pd.DataFrame(), {
            "query": query,
            "n_results": 0,
            "warnings": ["No PubMed results found."],
            "pass": True,
        }

    # Be polite between NCBI requests.
    time.sleep(0.34)

    esummary_params: dict[str, Any] = {
        "db": "pubmed",
        "id": ",".join(ids),
        "retmode": "json",
    }

    if email:
        esummary_params["email"] = email
    if api_key:
        esummary_params["api_key"] = api_key

    esummary = _safe_get_json(f"{base}/esummary.fcgi", esummary_params)

    if not esummary["ok"]:
        return pd.DataFrame({"pmid": ids}), {
            "query": query,
            "n_results": len(ids),
            "warnings": [f"PubMed ESummary failed: {esummary['error']}"],
            "pass": False,
        }

    result_obj = esummary["data"].get("result", {})
    rows = []

    for pmid in ids:
        item = result_obj.get(pmid, {})
        rows.append(
            {
                "source": "PubMed",
                "query": query,
                "pmid": pmid,
                "title": item.get("title"),
                "journal": item.get("fulljournalname"),
                "pubdate": item.get("pubdate"),
                "authors": ", ".join(
                    author.get("name", "")
                    for author in item.get("authors", [])[:5]
                    if isinstance(author, dict)
                ),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            }
        )

    df = pd.DataFrame(rows)

    summary = {
        "query": query,
        "n_results": int(len(df)),
        "warnings": [],
        "pass": True,
    }

    return df, summary


def search_jaspar_motif(
    tf_name: str,
    species: int = 9606,
    collection: str = "CORE",
    max_results: int = 5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Search JASPAR for transcription factor motif matrix profiles.

    species=9606 means human.
    """

    url = "https://jaspar.elixir.no/api/v1/matrix/"

    params = {
        "search": tf_name,
        "species": species,
        "collection": collection,
        "format": "json",
        "page_size": max_results,
    }

    response = _safe_get_json(url, params=params)

    if not response["ok"]:
        return pd.DataFrame(), {
            "tf_name": tf_name,
            "n_results": 0,
            "warnings": [f"JASPAR search failed: {response['error']}"],
            "pass": False,
        }

    data = response["data"]
    results = data.get("results", []) if isinstance(data, dict) else []

    rows = []
    for item in results[:max_results]:
        matrix_id = item.get("matrix_id") or item.get("matrix_id_base")
        rows.append(
            {
                "source": "JASPAR",
                "tf_name": tf_name,
                "matrix_id": matrix_id,
                "name": item.get("name"),
                "collection": item.get("collection"),
                "tax_group": item.get("tax_group"),
                "species": item.get("species"),
                "class": item.get("class"),
                "family": item.get("family"),
                "url": f"https://jaspar.elixir.no/matrix/{matrix_id}/" if matrix_id else None,
            }
        )

    df = pd.DataFrame(rows)

    summary = {
        "tf_name": tf_name,
        "n_results": int(len(df)),
        "warnings": [],
        "pass": True,
    }

    if len(df) == 0:
        summary["warnings"].append("No JASPAR motif records found.")

    return df, summary


def search_encode_tf(
    tf_name: str,
    cell_type: str | None = None,
    max_results: int = 5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Search ENCODE for TF-related experiment records.

    This is intentionally broad because ENCODE metadata terms vary.
    It works best for TF names like GATA1, TAL1, CTCF, SPI1, HNF4A.
    """

    url = "https://www.encodeproject.org/search/"

    params: dict[str, Any] = {
        "type": "Experiment",
        "searchTerm": tf_name,
        "format": "json",
        "limit": max_results,
    }

    if cell_type:
        params["searchTerm"] = f"{tf_name} {cell_type}"

    response = _safe_get_json(url, params=params)

    if not response["ok"]:
        return pd.DataFrame(), {
            "tf_name": tf_name,
            "cell_type": cell_type,
            "n_results": 0,
            "warnings": [f"ENCODE search failed: {response['error']}"],
            "pass": False,
        }

    graph = response["data"].get("@graph", [])

    rows = []
    for item in graph[:max_results]:
        accession = item.get("accession")
        biosample = item.get("biosample_ontology", {}) or {}
        target = item.get("target", {}) or {}

        rows.append(
            {
                "source": "ENCODE",
                "tf_name": tf_name,
                "cell_type_query": cell_type,
                "accession": accession,
                "assay_title": item.get("assay_title"),
                "target_label": target.get("label") if isinstance(target, dict) else None,
                "biosample_term_name": biosample.get("term_name") if isinstance(biosample, dict) else None,
                "status": item.get("status"),
                "url": f"https://www.encodeproject.org/experiments/{accession}/"
                if accession
                else None,
            }
        )

    df = pd.DataFrame(rows)

    summary = {
        "tf_name": tf_name,
        "cell_type": cell_type,
        "n_results": int(len(df)),
        "warnings": [],
        "pass": True,
    }

    if len(df) == 0:
        summary["warnings"].append("No ENCODE experiment records found.")

    return df, summary


def literature_search_for_motifs(
    motif_table_path: str | Path,
    motif_col: str = "motif",
    target_cell_type: str | None = None,
    off_target_cell_type: str | None = None,
    top_n_motifs: int = 5,
    max_pubmed_results_per_motif: int = 3,
    max_database_results_per_motif: int = 3,
    email: str | None = None,
    ncbi_api_key: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Run actual API-backed literature/database retrieval for enriched TF motifs.

    Inputs:
        motif_table_path:
            Output from motif_enrichment_summary(), or any table with a motif column.

    For each top motif, this queries:
        - PubMed through NCBI E-utilities
        - JASPAR REST API
        - ENCODE REST API

    Output:
        One combined evidence table that can be passed to an LLM for interpretation.
    """

    motif_df = _read_table(motif_table_path)
    _check_cols(motif_df, {motif_col}, str(motif_table_path))

    if "enrichment_ratio" in motif_df.columns:
        motif_df = motif_df.sort_values("enrichment_ratio", ascending=False)

    motifs = [
        str(m)
        for m in motif_df[motif_col].dropna().head(top_n_motifs).tolist()
        if str(m).lower() != "none"
    ]

    all_rows = []
    warnings_list = []

    for motif in motifs:
        if target_cell_type and off_target_cell_type:
            pubmed_query = (
                f'({motif}[Title/Abstract]) AND '
                f'({target_cell_type}[Title/Abstract] OR enhancer[Title/Abstract] '
                f'OR promoter[Title/Abstract] OR MPRA[Title/Abstract] '
                f'OR "reporter assay"[Title/Abstract]) NOT '
                f'({off_target_cell_type}[Title/Abstract])'
            )
        elif target_cell_type:
            pubmed_query = (
                f'({motif}[Title/Abstract]) AND '
                f'({target_cell_type}[Title/Abstract] OR enhancer[Title/Abstract] '
                f'OR promoter[Title/Abstract] OR MPRA[Title/Abstract] '
                f'OR "reporter assay"[Title/Abstract])'
            )
        else:
            pubmed_query = (
                f'({motif}[Title/Abstract]) AND '
                f'(enhancer[Title/Abstract] OR promoter[Title/Abstract] '
                f'OR MPRA[Title/Abstract] OR "reporter assay"[Title/Abstract])'
            )

        pubmed_df, pubmed_summary = search_pubmed(
            query=pubmed_query,
            max_results=max_pubmed_results_per_motif,
            email=email,
            api_key=ncbi_api_key,
        )
        if len(pubmed_df):
            pubmed_df["motif"] = motif
            pubmed_df["evidence_type"] = "literature"
            all_rows.append(pubmed_df)
        warnings_list.extend(pubmed_summary.get("warnings", []))

        jaspar_df, jaspar_summary = search_jaspar_motif(
            tf_name=motif,
            max_results=max_database_results_per_motif,
        )
        if len(jaspar_df):
            jaspar_df["motif"] = motif
            jaspar_df["evidence_type"] = "motif_database"
            all_rows.append(jaspar_df)
        warnings_list.extend(jaspar_summary.get("warnings", []))

        encode_df, encode_summary = search_encode_tf(
            tf_name=motif,
            cell_type=target_cell_type,
            max_results=max_database_results_per_motif,
        )
        if len(encode_df):
            encode_df["motif"] = motif
            encode_df["evidence_type"] = "functional_genomics"
            all_rows.append(encode_df)
        warnings_list.extend(encode_summary.get("warnings", []))

        # Avoid hammering APIs.
        time.sleep(0.25)

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True, sort=False)
    else:
        combined = pd.DataFrame(
            columns=[
                "source",
                "motif",
                "evidence_type",
                "title",
                "journal",
                "pubdate",
                "matrix_id",
                "accession",
                "url",
            ]
        )

    summary = {
        "motifs_searched": motifs,
        "n_motifs": int(len(motifs)),
        "n_evidence_records": int(len(combined)),
        "target_cell_type": target_cell_type,
        "off_target_cell_type": off_target_cell_type,
        "warnings": warnings_list,
        "pass": True,
    }

    return combined, summary


def interpret_literature_evidence(
    evidence_table_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Convert retrieved API records into a simple interpretation-ready summary.

    This does not use an LLM. It creates a structured summary that the MCP agent
    or frontend can display.
    """

    df = _read_table(evidence_table_path)

    if df.empty:
        return df, {
            "n_sources": 0,
            "interpretation": "No literature or database evidence was retrieved.",
            "warnings": ["Evidence table is empty."],
            "pass": False,
        }

    source_counts = df["source"].value_counts().to_dict() if "source" in df.columns else {}

    motif_counts = df["motif"].value_counts().to_dict() if "motif" in df.columns else {}

    interpretation_parts = []

    if motif_counts:
        top_motif = max(motif_counts, key=motif_counts.get)
        interpretation_parts.append(
            f"The retrieved evidence is strongest for {top_motif}, "
            f"which has {motif_counts[top_motif]} supporting records."
        )

    if "PubMed" in source_counts:
        interpretation_parts.append(
            f"PubMed returned {source_counts['PubMed']} literature records."
        )

    if "JASPAR" in source_counts:
        interpretation_parts.append(
            f"JASPAR returned {source_counts['JASPAR']} motif profile records."
        )

    if "ENCODE" in source_counts:
        interpretation_parts.append(
            f"ENCODE returned {source_counts['ENCODE']} functional genomics records."
        )

    result = pd.DataFrame(
        {
            "metric": ["source_counts", "motif_counts"],
            "value": [str(source_counts), str(motif_counts)],
        }
    )

    summary = {
        "n_sources": int(len(source_counts)),
        "source_counts": source_counts,
        "motif_counts": motif_counts,
        "interpretation": " ".join(interpretation_parts),
        "warnings": [],
        "pass": True,
    }

    return result, summary