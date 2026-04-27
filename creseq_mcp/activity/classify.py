"""
creseq_mcp/activity_calling.py
==============================
Full-featured activity calling: classify CRE-seq elements as active vs.
inactive against an empirical null distribution drawn from negative controls.

Compared to the lightweight ``qc.activity.call_activity`` (a quick QC check),
this module:
  - uses median / MAD for the null (robust to outlier controls),
  - reports per-element pvalue, fdr (BH), zscore, fold_over_controls,
  - excludes negative controls from testing (their pvalue/fdr/zscore/fold = NaN),
  - emits a structured summary dict suitable for downstream plotting and
    motif-enrichment tools.

References
----------
- Ashuach et al. 2019 (MPRAnalyze) — robust null estimation via median/MAD.
- Benjamini & Hochberg 1995 — FDR correction.
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation, norm, shapiro
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)

_REQUIRED_INPUT_COLS = {"element_id", "mean_activity"}


def _validate_inputs(
    activity_table: pd.DataFrame,
    negative_control_ids: list[str],
) -> None:
    if len(activity_table) == 0:
        raise ValueError("activity_table is empty.")

    missing = _REQUIRED_INPUT_COLS - set(activity_table.columns)
    if missing:
        raise ValueError(
            f"activity_table missing required columns: {missing}. "
            f"Found: {list(activity_table.columns)}"
        )

    if not negative_control_ids:
        raise ValueError("negative_control_ids is empty — at least one control required.")

    table_ids = set(activity_table["element_id"])
    not_found = [c for c in negative_control_ids if c not in table_ids]
    if len(not_found) == len(negative_control_ids):
        raise ValueError(
            f"None of the supplied negative_control_ids could be matched — "
            f"all IDs not found in activity_table. Examples: {not_found[:5]}"
        )
    if not_found:
        warnings.warn(
            f"{len(not_found)} negative_control_ids not found in activity_table "
            f"(possibly dropped during QC). Examples: {not_found[:5]}",
            UserWarning,
            stacklevel=3,
        )


def call_active_elements_empirical(
    activity_table: pd.DataFrame,
    negative_control_ids: list[str],
    fdr_threshold: float = 0.05,
) -> tuple[pd.DataFrame, dict]:
    """
    Classify elements as active/inactive using an empirical null distribution
    derived from negative controls.

    Parameters
    ----------
    activity_table : DataFrame
        Per-element activity table.  Must contain at least
        ``element_id`` and ``mean_activity``; ``std_activity`` and
        ``n_barcodes`` are propagated to the output if present.
    negative_control_ids : list[str]
        Element IDs treated as the null.  Their rows define the location/scale
        of the null distribution and are excluded from significance testing
        (their pvalue/fdr/zscore/fold_over_controls are NaN).
    fdr_threshold : float, default 0.05
        Benjamini-Hochberg significance level.

    Returns
    -------
    classified_df : DataFrame
        One row per element with columns:
        ``element_id, mean_activity, std_activity, n_barcodes, active,
        pvalue, fdr, fold_over_controls, zscore``.  ``active`` is the BH
        rejection at *fdr_threshold* (one-sided, upper tail).
    summary : dict
        Structured summary with the null distribution parameters, per-class
        counts, and an ``active_summary`` block describing the called set.
    """
    _validate_inputs(activity_table, negative_control_ids)

    df = activity_table.copy()
    # Backfill optional columns so the output schema is stable.
    if "std_activity" not in df.columns:
        df["std_activity"] = np.nan
    if "n_barcodes" not in df.columns:
        df["n_barcodes"] = pd.NA

    is_control = df["element_id"].isin(set(negative_control_ids))
    neg_activities = df.loc[is_control, "mean_activity"].dropna().to_numpy()

    if len(neg_activities) < 3:
        raise ValueError(
            f"Need at least 3 negative controls present in the activity table "
            f"to fit a null; got {len(neg_activities)}."
        )

    if len(neg_activities) < 20:
        warnings.warn(
            f"Only {len(neg_activities)} negative controls present (fewer than "
            f"20 recommended) — the null distribution is underpowered and FDR "
            f"estimates may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Robust location/scale (Ashuach 2019).
    null_center = float(np.median(neg_activities))
    null_scale = float(median_abs_deviation(neg_activities, scale="normal"))
    if null_scale <= 0:
        # Degenerate fallback — every control identical.  Use a tiny floor so
        # downstream z-scores stay finite without inflating false positives.
        null_scale = max(float(np.std(neg_activities, ddof=1)), 1e-9)

    # Normality check on the null (Shapiro requires 3 ≤ n ≤ 5000).
    null_normality_p: float | None = None
    if 3 <= len(neg_activities) <= 5000:
        try:
            null_normality_p = float(shapiro(neg_activities).pvalue)
        except Exception:
            null_normality_p = None

    warnings_list: list[str] = []
    if null_normality_p is not None and null_normality_p < 0.01:
        msg = (
            f"Shapiro–Wilk p={null_normality_p:.3g} < 0.01: negative-control "
            f"distribution is non-normal; the empirical z-test assumes "
            f"normality. Consider the GLM method or more controls."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
        warnings_list.append(msg)

    # One-sided z-test for activity ABOVE background.
    zscores = (df["mean_activity"].to_numpy() - null_center) / null_scale
    pvalues = 1.0 - norm.cdf(zscores)

    # BH on test elements only — controls are excluded from the multiple-testing
    # set (testing them against themselves is circular).
    test_mask = (~is_control).to_numpy()
    test_pvals = pvalues[test_mask]

    fdrs = np.full(len(df), np.nan)
    active = np.zeros(len(df), dtype=bool)
    if len(test_pvals) > 0:
        rejected, fdr_test, _, _ = multipletests(
            test_pvals, alpha=fdr_threshold, method="fdr_bh"
        )
        fdrs[test_mask] = fdr_test
        active[test_mask] = rejected

    fold = np.power(2.0, df["mean_activity"].to_numpy() - null_center)

    # Controls don't get tested → null out their stats (but keep them in the
    # output so callers can still see their raw activity).
    pvalues = np.where(is_control, np.nan, pvalues)
    zscores_out = np.where(is_control, np.nan, zscores)
    fold_out = np.where(is_control, np.nan, fold)

    classified = pd.DataFrame({
        "element_id": df["element_id"].to_numpy(),
        "mean_activity": df["mean_activity"].to_numpy(),
        "std_activity": df["std_activity"].to_numpy(),
        "n_barcodes": df["n_barcodes"].to_numpy(),
        "active": active,
        "pvalue": pvalues,
        "fdr": fdrs,
        "fold_over_controls": fold_out,
        "zscore": zscores_out,
    })

    # ── summary ──────────────────────────────────────────────────────────────
    n_active = int(active.sum())
    n_test = int(test_mask.sum())
    n_inactive = n_test - n_active

    # Putative silencers: significantly *below* background (one-sided lower tail).
    lower_tail_p = norm.cdf(zscores)
    silencer_mask = test_mask & (lower_tail_p < fdr_threshold)
    n_silencer_candidates = int(silencer_mask.sum())

    active_block: dict[str, float | list[float] | None]
    if n_active:
        act_rows = classified.loc[active]
        active_block = {
            "median_activity": float(act_rows["mean_activity"].median()),
            "median_fold_over_controls": float(act_rows["fold_over_controls"].median()),
            "activity_range": [
                float(act_rows["mean_activity"].min()),
                float(act_rows["mean_activity"].max()),
            ],
        }
    else:
        active_block = {
            "median_activity": None,
            "median_fold_over_controls": None,
            "activity_range": None,
        }

    summary = {
        "n_total_elements": int(len(df)),
        "n_negative_controls": int(is_control.sum()),
        "n_test_elements": n_test,
        "n_active": n_active,
        "n_inactive": n_inactive,
        "fdr_threshold": float(fdr_threshold),
        "method": "empirical",
        "null_distribution": {
            "center": null_center,
            "scale": null_scale,
            "estimator": "median/MAD",
            "n_controls": int(len(neg_activities)),
            "shapiro_pvalue": null_normality_p,
        },
        "active_summary": active_block,
        "n_silencer_candidates": n_silencer_candidates,
        "warnings": warnings_list,
    }

    return classified, summary


def call_active_elements_glm(
    count_table: pd.DataFrame,
    negative_control_ids: list[str],
    fdr_threshold: float = 0.05,
) -> tuple[pd.DataFrame, dict]:
    """
    Negative-binomial GLM activity calling on raw barcode counts.

    Not yet implemented — placeholder so the dispatcher contract is stable.
    Will fit ``log(RNA) ~ log(DNA) + element_indicator`` per element using
    statsmodels and compare each element's coefficient to the distribution of
    coefficients from negative controls.
    """
    raise NotImplementedError(
        "GLM-based activity calling is not yet implemented. Use method='empirical'."
    )


def call_active_elements(
    activity_table_path: str | Path,
    negative_controls: list[str],
    fdr_threshold: float = 0.05,
    method: str = "empirical",
    count_table_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict:
    """
    Dispatcher MCP entry point.

    Reads the activity table from disk, runs the requested method, writes the
    classified TSV next to the input (or to *output_path* if given), and
    returns a JSON-safe dict with the output path and summary.
    """
    activity_path = Path(activity_table_path)
    activity_df = pd.read_csv(activity_path, sep="\t")

    if method == "empirical":
        classified_df, summary = call_active_elements_empirical(
            activity_df, negative_controls, fdr_threshold
        )
    elif method == "glm":
        if count_table_path is None:
            raise ValueError(
                "method='glm' requires count_table_path (raw barcode counts)."
            )
        count_df = pd.read_csv(count_table_path, sep="\t")
        classified_df, summary = call_active_elements_glm(
            count_df, negative_controls, fdr_threshold
        )
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'empirical' or 'glm'.")

    if output_path is None:
        suffix = activity_path.suffix or ".tsv"
        out_path = activity_path.with_name(activity_path.stem + "_classified" + suffix)
    else:
        out_path = Path(output_path)

    classified_df.to_csv(out_path, sep="\t", index=False)

    return {
        "classified_elements": str(out_path),
        "summary": summary,
        "summary_json": json.dumps(summary, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
    }
