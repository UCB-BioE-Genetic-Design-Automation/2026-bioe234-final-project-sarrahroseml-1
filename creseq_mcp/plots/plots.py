"""
creseq_mcp/plots/plots.py
=========================
Publication-quality figures for CRE-seq analysis output.

One MCP-facing dispatcher (``plot_creseq``) that routes to five private plot
implementations.  Every plot returns ``{"plot_path", "description"}`` — the
description is a concise natural-language summary so an LLM caller can relay
the figure's contents without seeing the image.

Conventions
-----------
- Headless Agg backend (no GUI windows).
- Consistent palette across plots: red ``#E63946`` for active, grey
  ``#BBBBBB`` for inactive, blue ``#457B9D`` for negative controls, teal
  ``#2A9D8F`` for highlights.
- Title 14 pt, axis labels 12 pt, legend 8 pt; PNG @ 200 DPI; no top/right
  spines.
- ``fig.tight_layout()`` then ``fig.savefig(...)`` then ``plt.close(fig)`` to
  avoid memory leaks under repeated test invocation.
"""
from __future__ import annotations

import logging
from pathlib import Path
import matplotlib

matplotlib.use("Agg")  # headless — must come before pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

logger = logging.getLogger(__name__)

# Palette
_COLOR_ACTIVE = "#E63946"
_COLOR_INACTIVE = "#BBBBBB"
_COLOR_CONTROL = "#457B9D"
_COLOR_HIGHLIGHT = "#2A9D8F"

_TITLE_FS = 14
_LABEL_FS = 12
_LEGEND_FS = 8
_DPI = 200


def _strip_spines(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig: plt.Figure, output_path: str | Path) -> None:
    """Save with tight layout, close to release memory.  SVG inferred from suffix."""
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 1 — volcano
# ---------------------------------------------------------------------------


def _plot_volcano(
    df: pd.DataFrame,
    output_path: str,
    neg_control_ids: list[str] | None = None,
    highlight_ids: list[str] | None = None,
    fdr_threshold: float = 0.05,
) -> dict:
    """Volcano plot: log2(activity) vs −log10(p-value)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    is_control = df["pvalue"].isna()
    is_active = df["active"].astype(bool)
    is_inactive = (~is_control) & (~is_active)

    # Clip to a tiny floor so underflowed (==0) p-values don't blow up −log10.
    _PMIN = 1e-300

    inactive = df[is_inactive]
    if len(inactive):
        ax.scatter(
            inactive["mean_activity"],
            -np.log10(inactive["pvalue"].clip(lower=_PMIN)),
            c=_COLOR_INACTIVE, s=8, alpha=0.5,
            label=f"Inactive (n={len(inactive)})", zorder=1,
        )

    active = df[is_active]
    if len(active):
        ax.scatter(
            active["mean_activity"],
            -np.log10(active["pvalue"].clip(lower=_PMIN)),
            c=_COLOR_ACTIVE, s=12, alpha=0.7,
            label=f"Active (n={len(active)})", zorder=2,
        )

    n_controls_plotted = 0
    if neg_control_ids:
        controls = df[is_control]
        n_controls_plotted = len(controls)
        if n_controls_plotted:
            ax.scatter(
                controls["mean_activity"],
                np.full(n_controls_plotted, 0.1),
                c=_COLOR_CONTROL, s=10, alpha=0.6, marker="D",
                label=f"Neg controls (n={n_controls_plotted})", zorder=1,
            )

    if highlight_ids:
        highlighted = df[df["element_id"].isin(set(highlight_ids))]
        if len(highlighted):
            # Highlights at y=0.1 if they have NaN p-values (i.e. controls).
            y_vals = np.where(
                highlighted["pvalue"].isna(),
                0.1,
                -np.log10(highlighted["pvalue"].fillna(1.0).clip(lower=_PMIN)),
            )
            ax.scatter(
                highlighted["mean_activity"], y_vals,
                c=_COLOR_HIGHLIGHT, s=30, edgecolors="black", linewidth=0.5,
                label=f"Highlighted (n={len(highlighted)})", zorder=3,
            )
            for (_, row), y in zip(highlighted.iterrows(), y_vals):
                ax.annotate(
                    str(row["element_id"]),
                    (row["mean_activity"], y),
                    fontsize=6, ha="left", va="bottom",
                )

    if is_active.any():
        pval_cutoff = float(df.loc[is_active, "pvalue"].max())
        if pval_cutoff > 0:
            ax.axhline(
                -np.log10(max(pval_cutoff, _PMIN)),
                color="black", linestyle="--", linewidth=0.8, alpha=0.5,
                label=f"FDR = {fdr_threshold}",
            )

    ax.set_xlabel("Activity (log2 RNA/DNA)", fontsize=_LABEL_FS)
    ax.set_ylabel("-log10(p-value)", fontsize=_LABEL_FS)
    ax.set_title("CRE-seq Activity Volcano Plot", fontsize=_TITLE_FS)
    ax.legend(fontsize=_LEGEND_FS, loc="upper left", framealpha=0.9)
    _strip_spines(ax)

    _save(fig, output_path)

    n_active = int(is_active.sum())
    n_test = int((~is_control).sum())
    description = (
        f"Volcano plot showing {n_test} test elements. "
        f"{n_active} elements are active (FDR < {fdr_threshold}), shown in red. "
        f"Inactive elements in grey."
    )
    if neg_control_ids:
        description += f" {n_controls_plotted} negative controls shown as blue diamonds."

    return {"plot_path": str(output_path), "description": description}


# ---------------------------------------------------------------------------
# Plot 2 — ranked activity
# ---------------------------------------------------------------------------


def _plot_ranked_activity(
    df: pd.DataFrame,
    output_path: str,
    neg_control_ids: list[str] | None = None,
    highlight_ids: list[str] | None = None,
) -> dict:
    """All elements sorted by activity, with active overlay and control marks."""
    df_sorted = df.sort_values("mean_activity", ascending=True).reset_index(drop=True)
    df_sorted["rank"] = np.arange(len(df_sorted))

    fig, ax = plt.subplots(figsize=(10, 5))

    is_control = df_sorted["pvalue"].isna()
    is_active = df_sorted["active"].astype(bool)

    ax.plot(
        df_sorted["rank"], df_sorted["mean_activity"],
        color=_COLOR_INACTIVE, linewidth=0.5, zorder=1,
    )

    active = df_sorted[is_active]
    ax.scatter(
        active["rank"], active["mean_activity"],
        c=_COLOR_ACTIVE, s=6, alpha=0.7,
        label=f"Active (n={len(active)})", zorder=2,
    )

    if neg_control_ids:
        controls = df_sorted[is_control]
        ax.scatter(
            controls["rank"], controls["mean_activity"],
            c=_COLOR_CONTROL, s=24, alpha=0.7, marker="|",
            label=f"Neg controls (n={len(controls)})", zorder=2,
        )

    if highlight_ids:
        highlighted = df_sorted[df_sorted["element_id"].isin(set(highlight_ids))]
        if len(highlighted):
            ax.scatter(
                highlighted["rank"], highlighted["mean_activity"],
                c=_COLOR_HIGHLIGHT, s=25, edgecolors="black", linewidth=0.5,
                label=f"Highlighted (n={len(highlighted)})", zorder=3,
            )

    ax.set_xlabel("Element rank", fontsize=_LABEL_FS)
    ax.set_ylabel("Activity (log2 RNA/DNA)", fontsize=_LABEL_FS)
    ax.set_title("Ranked CRE Activity", fontsize=_TITLE_FS)
    ax.legend(fontsize=_LEGEND_FS, loc="upper left")
    _strip_spines(ax)

    _save(fig, output_path)

    description = (
        f"Ranked activity plot. {len(df_sorted)} elements sorted by log2(RNA/DNA). "
        f"{int(is_active.sum())} active elements highlighted in red."
    )
    return {"plot_path": str(output_path), "description": description}


# ---------------------------------------------------------------------------
# Plot 3 — replicate correlation
# ---------------------------------------------------------------------------


def _find_replicate_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c.startswith("rep") and c.endswith("_activity")
    ]


def _plot_replicate_correlation(
    df: pd.DataFrame,
    output_path: str,
) -> dict:
    """N×N replicate-correlation grid: scatter (lower), r (upper), histogram (diag)."""
    rep_cols = _find_replicate_columns(df)
    if len(rep_cols) < 2:
        raise ValueError(
            f"Need at least 2 replicate columns (named rep*_activity); "
            f"found: {rep_cols}"
        )

    n = len(rep_cols)
    fig, axes = plt.subplots(n, n, figsize=(4 * n, 4 * n))
    if n == 1:
        axes = np.array([[axes]])

    correlations: list[float] = []
    for i in range(n):
        for j in range(n):
            ax = axes[i][j]
            x = df[rep_cols[j]].dropna()
            y = df[rep_cols[i]].dropna()
            common_idx = x.index.intersection(y.index)
            x = x.loc[common_idx].to_numpy()
            y = y.loc[common_idx].to_numpy()

            if i == j:
                ax.hist(x, bins=40, color=_COLOR_CONTROL, alpha=0.7, edgecolor="none")
                ax.set_title(rep_cols[i].replace("_activity", ""), fontsize=10)
            elif i > j:
                ax.scatter(x, y, s=3, alpha=0.3, c="#333333")
                r = float(np.corrcoef(x, y)[0, 1]) if len(x) >= 2 else float("nan")
                correlations.append(r)
                ax.annotate(
                    f"r = {r:.3f}", xy=(0.05, 0.95), xycoords="axes fraction",
                    fontsize=10, fontweight="bold", va="top",
                )
            else:
                r = float(np.corrcoef(x, y)[0, 1]) if len(x) >= 2 else float("nan")
                color = _COLOR_HIGHLIGHT if (not np.isnan(r) and r >= 0.85) else _COLOR_ACTIVE
                ax.text(
                    0.5, 0.5, f"{r:.3f}",
                    ha="center", va="center", fontsize=20, fontweight="bold",
                    color=color, transform=ax.transAxes,
                )
                ax.set_xticks([])
                ax.set_yticks([])

            if j > 0 and i != j:
                ax.set_yticks([])
            if i < n - 1:
                ax.set_xticks([])
            _strip_spines(ax)

    fig.suptitle("Replicate Correlation", fontsize=_TITLE_FS, y=1.02)
    _save(fig, output_path)

    if correlations:
        min_r = float(min(correlations))
        mean_r = float(np.mean(correlations))
    else:
        min_r = mean_r = float("nan")

    description = (
        f"Replicate correlation matrix ({n} replicates). "
        f"Mean pairwise Pearson r = {mean_r:.3f}, min r = {min_r:.3f}."
    )
    if not np.isnan(min_r) and min_r < 0.85:
        description += " Warning: some replicate pairs show low correlation."

    return {"plot_path": str(output_path), "description": description}


# ---------------------------------------------------------------------------
# Plot 4 — annotation boxplot
# ---------------------------------------------------------------------------


def _plot_annotation_boxplot(
    df: pd.DataFrame,
    annotation_file: str,
    output_path: str,
) -> dict:
    """Boxplot of activity grouped by annotation, ordered by median."""
    annot = pd.read_csv(annotation_file, sep="\t")
    if "element_id" not in annot.columns or "annotation" not in annot.columns:
        raise ValueError(
            "Annotation file must have 'element_id' and 'annotation' columns; "
            f"got {list(annot.columns)}"
        )

    merged = df.merge(annot, on="element_id", how="inner")
    if merged.empty:
        raise ValueError(
            "No elements matched between activity table and annotation file. "
            "Check element_id values agree between the two inputs."
        )

    category_order = (
        merged.groupby("annotation")["mean_activity"]
        .median()
        .sort_values()
        .index.tolist()
    )

    fig, ax = plt.subplots(figsize=(max(8, len(category_order) * 1.2), 6))

    sns.boxplot(
        data=merged, x="annotation", y="mean_activity",
        order=category_order, hue="annotation", hue_order=category_order,
        palette="Set2", legend=False,
        fliersize=2, linewidth=0.8, ax=ax,
    )

    counts = merged.groupby("annotation").size()
    y_top = ax.get_ylim()[1]
    for i, cat in enumerate(category_order):
        ax.text(
            i, y_top, f"n={counts[cat]}",
            ha="center", va="bottom", fontsize=8, color="#555555",
        )

    ax.set_xlabel("Genomic Annotation", fontsize=_LABEL_FS)
    ax.set_ylabel("Activity (log2 RNA/DNA)", fontsize=_LABEL_FS)
    ax.set_title("Activity by Genomic Annotation", fontsize=_TITLE_FS)
    ax.tick_params(axis="x", rotation=45)
    _strip_spines(ax)

    _save(fig, output_path)

    top_cat = category_order[-1]
    bot_cat = category_order[0]
    description = (
        f"Activity distributions across {len(category_order)} annotation categories "
        f"({len(merged)} elements). Highest median: {top_cat}. Lowest median: {bot_cat}."
    )
    return {"plot_path": str(output_path), "description": description}


# ---------------------------------------------------------------------------
# Plot 5 — motif dotplot
# ---------------------------------------------------------------------------


def _plot_motif_dotplot(
    df: pd.DataFrame,
    output_path: str,
    max_motifs: int = 20,
) -> dict:
    """Dot plot of enriched TF motifs: OR on x, FDR as color, hits as size."""
    required = {"tf_name", "odds_ratio", "fdr", "n_active_hits"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Motif table must have columns {required}; got {set(df.columns)}"
        )

    sig = df[df["fdr"] < 0.05].copy()
    note = ""
    if sig.empty:
        sig = df.nlargest(max_motifs, "odds_ratio").copy()
        note = "No motifs reached FDR < 0.05. Showing top by odds ratio."
    else:
        sig = sig.nlargest(max_motifs, "odds_ratio")

    sig = sig.sort_values("odds_ratio", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, max(4, len(sig) * 0.4)))

    color_vals = -np.log10(sig["fdr"].clip(lower=1e-50))
    scatter = ax.scatter(
        sig["odds_ratio"],
        np.arange(len(sig)),
        s=sig["n_active_hits"] * 3,
        c=color_vals, cmap="YlOrRd",
        edgecolors="black", linewidth=0.3, alpha=0.85,
    )

    ax.set_yticks(np.arange(len(sig)))
    ax.set_yticklabels(sig["tf_name"], fontsize=10)
    ax.set_xlabel("Odds Ratio (active vs. background)", fontsize=_LABEL_FS)
    ax.set_title("Motif Enrichment", fontsize=_TITLE_FS)
    ax.axvline(1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    _strip_spines(ax)

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("-log10(FDR)", fontsize=10)

    for size_val in (5, 20, 50):
        ax.scatter(
            [], [], s=size_val * 3, c="grey",
            edgecolors="black", linewidth=0.3, label=f"{size_val} hits",
        )
    ax.legend(
        title="Hits in active set", fontsize=_LEGEND_FS, title_fontsize=9,
        loc="lower right", framealpha=0.9,
    )

    _save(fig, output_path)

    top_motif = sig.iloc[-1]
    description = (
        f"Motif enrichment dot plot showing top {len(sig)} motifs. "
        f"Top enriched: {top_motif['tf_name']} "
        f"(OR={float(top_motif['odds_ratio']):.1f}, "
        f"FDR={float(top_motif['fdr']):.1e}). "
        f"Dot size = number of hits in active elements, color = significance."
    )
    if note:
        description = note + " " + description

    return {"plot_path": str(output_path), "description": description}


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


_PLOT_TYPES: tuple[str, ...] = (
    "volcano",
    "ranked_activity",
    "replicate_correlation",
    "annotation_boxplot",
    "motif_dotplot",
)


def plot_creseq(
    data_file: str,
    plot_type: str,
    output_path: str = "plot.png",
    highlight_ids: list[str] | None = None,
    neg_control_ids: list[str] | None = None,
    annotation_file: str | None = None,
) -> dict:
    """
    MCP-facing dispatcher.  Reads *data_file* (TSV) and routes to the requested
    plot.  Returns ``{"plot_path", "description"}``.
    """
    if plot_type not in _PLOT_TYPES:
        raise ValueError(
            f"Unknown plot_type {plot_type!r}. Options: {list(_PLOT_TYPES)}"
        )

    if plot_type == "annotation_boxplot" and annotation_file is None:
        raise ValueError("annotation_boxplot requires annotation_file")

    df = pd.read_csv(data_file, sep="\t")

    if plot_type == "volcano":
        return _plot_volcano(df, output_path, neg_control_ids, highlight_ids)
    if plot_type == "ranked_activity":
        return _plot_ranked_activity(df, output_path, neg_control_ids, highlight_ids)
    if plot_type == "replicate_correlation":
        return _plot_replicate_correlation(df, output_path)
    if plot_type == "annotation_boxplot":
        return _plot_annotation_boxplot(df, annotation_file, output_path)  # type: ignore[arg-type]
    if plot_type == "motif_dotplot":
        return _plot_motif_dotplot(df, output_path)

    # Unreachable — guarded above.
    raise AssertionError("unreachable")
