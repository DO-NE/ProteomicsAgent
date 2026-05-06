"""Matrix visualization and export for EM degeneracy analysis.

Produces TSV exports and publication-quality figures for the mapping matrix
(A), uniform emission matrix (M), and detectability-weighted emission matrix
(W) to analyse peptide degeneracy patterns across taxa.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib

# Headless-safe backend — must be set before pyplot is imported.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse  # noqa: E402

logger = logging.getLogger(__name__)

_EPS = 1e-12


# ------------------------------------------------------------------ public API


def visualize_and_export_matrices(
    A: np.ndarray,
    M: np.ndarray,
    W: np.ndarray,
    peptide_list: list,
    taxon_labels: list,
    y: np.ndarray,
    output_dir: str,
    top_n_taxa: Optional[int] = None,
    pi: Optional[np.ndarray] = None,
) -> None:
    """Visualize and export mapping / emission matrices for degeneracy analysis.

    All outputs go to ``{output_dir}/diagnostics/matrices/``.

    Parameters
    ----------
    A : np.ndarray, shape (P, T)
        Binary mapping matrix.
    M : np.ndarray, shape (P, T)
        Uniform emission matrix (A / n_t per column).
    W : np.ndarray, shape (P, T)
        Detectability-weighted emission matrix used in EM (equals M when
        detectability_mode is 'uniform').
    peptide_list : list of str
        Row labels (peptide sequences).
    taxon_labels : list of str
        Column labels, typically ``"<taxon_id>|<taxon_name>"`` format.
    y : np.ndarray, shape (P,)
        Observed PSM counts.
    output_dir : str
        Base output directory; outputs go into ``diagnostics/matrices/``.
    top_n_taxa : int or None, optional
        Restrict analysis to the top-N taxa by repertoire size.
    pi : np.ndarray or None, optional
        Converged EM abundance vector ``(T,)``.  Used to identify the dominant
        taxon in Visualization (e); falls back to PSM-mass proxy when None.
    """
    out_dir = Path(str(output_dir)) / "diagnostics" / "matrices"
    out_dir.mkdir(parents=True, exist_ok=True)

    A_arr = np.asarray(A, dtype=np.float64)
    W_arr = np.asarray(W, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    taxon_labels = list(taxon_labels)
    peptide_list = list(peptide_list)

    P, T = A_arr.shape

    # Optionally restrict to top-N taxa by repertoire size.
    if top_n_taxa is not None and top_n_taxa < T:
        n_t_all = A_arr.sum(axis=0)
        keep = np.argsort(-n_t_all)[:top_n_taxa]
        A_arr = A_arr[:, keep]
        W_arr = W_arr[:, keep]
        taxon_labels = [taxon_labels[i] for i in keep]
        if pi is not None:
            pi = pi[keep]
        T = top_n_taxa

    # Short display names (strip "id|" prefix).
    short_labels = [
        lbl.split("|", 1)[-1] if "|" in lbl else lbl for lbl in taxon_labels
    ]

    # Row-level degeneracy: number of taxa each peptide maps to.
    n_taxa_mapped = A_arr.sum(axis=1)  # (P,)

    # Sort peptide rows: n_taxa_mapped desc, then y_p desc.
    sort_idx = np.lexsort((-y_arr, -n_taxa_mapped))

    logger.info("Matrix visualization: P=%d, T=%d, output=%s", P, T, out_dir)

    # ------------------------------------------------------------------ TSVs

    try:
        _write_matrix_tsv(
            out_dir / "mapping_matrix_A.tsv",
            A_arr,
            peptide_list,
            short_labels,
            n_taxa_mapped,
            y_arr,
            sort_idx,
            float_fmt=None,
        )
    except Exception as exc:
        logger.warning("mapping_matrix_A.tsv failed: %s", exc)

    try:
        _write_matrix_tsv(
            out_dir / "emission_matrix_W.tsv",
            W_arr,
            peptide_list,
            short_labels,
            n_taxa_mapped,
            y_arr,
            sort_idx,
            float_fmt=6,
        )
    except Exception as exc:
        logger.warning("emission_matrix_W.tsv failed: %s", exc)

    # Precompute pairwise similarity matrices (used by multiple TSVs and plots).
    jaccard = _compute_jaccard(A_arr)
    cosine = _compute_cosine(W_arr)

    try:
        _write_square_tsv(out_dir / "jaccard_similarity.tsv", jaccard, short_labels)
    except Exception as exc:
        logger.warning("jaccard_similarity.tsv failed: %s", exc)

    try:
        _write_shared_peptides_summary(
            out_dir / "shared_peptides_summary.tsv",
            A_arr,
            W_arr,
            jaccard,
            cosine,
            peptide_list,
            short_labels,
            y_arr,
        )
    except Exception as exc:
        logger.warning("shared_peptides_summary.tsv failed: %s", exc)

    try:
        _write_degeneracy_profile(
            out_dir / "degeneracy_profile.tsv",
            A_arr,
            short_labels,
        )
    except Exception as exc:
        logger.warning("degeneracy_profile.tsv failed: %s", exc)

    # -------------------------------------------------------- compressed saves

    try:
        scipy.sparse.save_npz(
            str(out_dir / "mapping_matrix_A.npz"),
            scipy.sparse.csc_matrix(A_arr),
        )
        np.savez_compressed(
            str(out_dir / "emission_matrix_W.npz"),
            W=W_arr,
            peptides=np.array(peptide_list, dtype=object),
            taxa=np.array(taxon_labels, dtype=object),
        )
        logger.info("Saved compressed matrices to %s", out_dir)
    except Exception as exc:
        logger.warning("Compressed matrix save failed: %s", exc)

    # ------------------------------------------------------- visualizations

    try:
        _plot_clustered_heatmap(
            out_dir / "jaccard_heatmap",
            jaccard,
            short_labels,
            title="Peptide Sharing (Jaccard Similarity) Between Taxa",
            cmap="Blues",
        )
    except Exception as exc:
        logger.warning("jaccard_heatmap failed: %s", exc)

    try:
        _plot_clustered_heatmap(
            out_dir / "cosine_heatmap",
            cosine,
            short_labels,
            title="Emission Profile Similarity (Cosine) Between Taxa",
            cmap="Blues",
        )
    except Exception as exc:
        logger.warning("cosine_heatmap failed: %s", exc)

    try:
        _plot_shared_peptide_heatmap(
            out_dir / "shared_peptide_heatmap",
            W_arr,
            n_taxa_mapped,
            y_arr,
            short_labels,
        )
    except Exception as exc:
        logger.warning("shared_peptide_heatmap failed: %s", exc)

    try:
        _plot_degeneracy_profile(
            out_dir / "degeneracy_profile",
            A_arr,
            short_labels,
        )
    except Exception as exc:
        logger.warning("degeneracy_profile plot failed: %s", exc)

    try:
        _plot_dominant_taxon_sharing(
            out_dir / "dominant_taxon_sharing",
            A_arr,
            short_labels,
            pi=pi,
            y=y_arr,
        )
    except Exception as exc:
        logger.warning("dominant_taxon_sharing plot failed: %s", exc)

    logger.info("Matrix visualization complete → %s", out_dir)


# ---------------------------------------------------------------- computation


def _compute_jaccard(A: np.ndarray) -> np.ndarray:
    """Return T×T pairwise Jaccard similarity matrix."""
    A_sp = scipy.sparse.csc_matrix(A, dtype=np.float64)
    intersection = (A_sp.T @ A_sp).toarray()  # (T, T) — |A∩B| counts
    n_t = A.sum(axis=0)  # (T,) repertoire sizes
    union = n_t[:, None] + n_t[None, :] - intersection
    with np.errstate(invalid="ignore", divide="ignore"):
        J = np.where(union > 0, intersection / union, 0.0)
    np.fill_diagonal(J, 1.0)
    return J


def _compute_cosine(W: np.ndarray) -> np.ndarray:
    """Return T×T pairwise cosine similarity matrix from W columns."""
    norms = np.linalg.norm(W, axis=0)  # (T,)
    norms_safe = np.where(norms > _EPS, norms, 1.0)
    W_norm = W / norms_safe[np.newaxis, :]
    cosine = W_norm.T @ W_norm
    np.fill_diagonal(cosine, 1.0)
    return np.clip(cosine, 0.0, 1.0)


# ---------------------------------------------------------------- TSV writers


def _write_matrix_tsv(
    path: Path,
    matrix: np.ndarray,
    peptide_list: list,
    taxon_names: list,
    n_taxa_mapped: np.ndarray,
    y: np.ndarray,
    sort_idx: np.ndarray,
    float_fmt: Optional[int],
) -> None:
    """Write a wide-format peptide × taxon TSV."""
    T = matrix.shape[1]
    with path.open("w", encoding="utf-8") as fh:
        fh.write(
            "\t".join(["peptide_sequence"] + taxon_names + ["n_taxa_mapped", "y_p"])
            + "\n"
        )
        for i in sort_idx:
            row = matrix[i]
            if float_fmt is None:
                vals = "\t".join(str(int(v)) for v in row)
            else:
                fmt = f"{{:.{float_fmt}f}}"
                vals = "\t".join(fmt.format(v) for v in row)
            fh.write(
                f"{peptide_list[i]}\t{vals}\t{int(n_taxa_mapped[i])}\t{int(y[i])}\n"
            )


def _write_square_tsv(path: Path, matrix: np.ndarray, labels: list) -> None:
    """Write a T×T similarity matrix with row and column headers."""
    T = len(labels)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\t" + "\t".join(labels) + "\n")
        for i in range(T):
            row = "\t".join(f"{matrix[i, j]:.6f}" for j in range(T))
            fh.write(f"{labels[i]}\t{row}\n")


def _write_shared_peptides_summary(
    path: Path,
    A: np.ndarray,
    W: np.ndarray,
    jaccard: np.ndarray,
    cosine: np.ndarray,
    peptide_list: list,
    taxon_names: list,
    y: np.ndarray,
) -> None:
    """Write one row per taxon pair that shares at least one peptide."""
    A_sp = scipy.sparse.csc_matrix(A, dtype=np.float64)
    intersection = (A_sp.T @ A_sp).toarray().astype(int)
    T = A.shape[1]

    rows = []
    for i in range(T):
        for j in range(i + 1, T):
            n_shared = int(intersection[i, j])
            if n_shared == 0:
                continue
            shared_mask = (A[:, i] > 0) & (A[:, j] > 0)
            shared_idx = np.where(shared_mask)[0]
            top_k = shared_idx[np.argsort(-y[shared_idx])[:10]]
            top_peps = ";".join(peptide_list[k] for k in top_k)
            rows.append((
                taxon_names[i],
                taxon_names[j],
                n_shared,
                jaccard[i, j],
                cosine[i, j],
                top_peps,
            ))

    rows.sort(key=lambda r: r[2], reverse=True)

    with path.open("w", encoding="utf-8") as fh:
        fh.write(
            "taxon_1\ttaxon_2\tn_shared\tjaccard\tcosine_W\ttop_shared_peptides\n"
        )
        for r in rows:
            fh.write(
                f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]:.6f}\t{r[4]:.6f}\t{r[5]}\n"
            )


def _write_degeneracy_profile(
    path: Path,
    A: np.ndarray,
    taxon_names: list,
) -> None:
    """Write per-taxon unique vs shared peptide counts."""
    n_taxa_mapped = A.sum(axis=1)
    T = A.shape[1]

    with path.open("w", encoding="utf-8") as fh:
        fh.write(
            "taxon\tn_total\tn_unique\tn_shared\tfrac_shared\trepertoire_size\n"
        )
        for t in range(T):
            col_mask = A[:, t] > 0
            n_total = int(col_mask.sum())
            n_unique = int((col_mask & (n_taxa_mapped == 1)).sum())
            n_shared = n_total - n_unique
            frac = n_shared / n_total if n_total > 0 else 0.0
            fh.write(
                f"{taxon_names[t]}\t{n_total}\t{n_unique}\t{n_shared}\t"
                f"{frac:.6f}\t{n_total}\n"
            )


# ------------------------------------------------------------ visualization


def _save_fig(fig: plt.Figure, path_stem: Path) -> None:
    """Save figure as both PNG and PDF at 300 dpi, white background."""
    for ext in (".png", ".pdf"):
        fig.savefig(
            str(path_stem) + ext,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )


def _plot_clustered_heatmap(
    path_stem: Path,
    matrix: np.ndarray,
    labels: list,
    title: str,
    cmap: str = "Blues",
) -> None:
    """Clustered T×T similarity heatmap (Jaccard or cosine)."""
    import pandas as pd

    T = len(labels)
    if T < 2:
        logger.debug("Skipping %s: T=%d < 2", path_stem.name, T)
        return

    df = pd.DataFrame(matrix, index=labels, columns=labels)
    figsize = (max(8.0, T * 0.7), max(6.0, T * 0.6))
    annot = T <= 25
    annot_fmt = ".2f" if T <= 15 else ".1f"
    annot_kw = {"size": max(4, 9 - max(0, T - 10) // 3)}

    try:
        import seaborn as sns
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import squareform

        dist = np.clip(1.0 - matrix, 0.0, None)
        np.fill_diagonal(dist, 0.0)
        link = linkage(squareform(dist, checks=False), method="average")

        g = sns.clustermap(
            df,
            row_linkage=link,
            col_linkage=link,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            annot=annot,
            fmt=annot_fmt if annot else "",
            annot_kws=annot_kw if annot else {},
            figsize=figsize,
            linewidths=0.5 if T <= 40 else 0.0,
        )
        g.fig.suptitle(title, y=1.02, fontsize=11, fontweight="bold")
        g.fig.patch.set_facecolor("white")
        _save_fig(g.fig, path_stem)
        plt.close(g.fig)

    except ImportError:
        # Fallback: plain imshow without clustering.
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(T))
        ax.set_yticks(range(T))
        tick_fs = max(4, 9 - T // 5)
        ax.set_xticklabels(labels, rotation=90, fontsize=tick_fs)
        ax.set_yticklabels(labels, fontsize=tick_fs)
        plt.colorbar(im, ax=ax)
        ax.set_title(title, fontweight="bold")
        fig.patch.set_facecolor("white")
        plt.tight_layout()
        _save_fig(fig, path_stem)
        plt.close(fig)


def _plot_shared_peptide_heatmap(
    path_stem: Path,
    W: np.ndarray,
    n_taxa_mapped: np.ndarray,
    y: np.ndarray,
    labels: list,
) -> None:
    """Filtered P×T heatmap of W for peptides shared across ≥2 taxa."""
    import pandas as pd

    T = W.shape[1]
    shared_mask = n_taxa_mapped >= 2
    W_shared = W[shared_mask]
    y_shared = y[shared_mask]

    if W_shared.shape[0] == 0:
        logger.debug("No shared peptides — skipping shared_peptide_heatmap")
        return

    if W_shared.shape[0] > 200:
        top_idx = np.argsort(-y_shared)[:200]
        W_shared = W_shared[top_idx]
        y_shared = y_shared[top_idx]

    P_plot = W_shared.shape[0]

    # Decide color scale: log1p when values span >2 orders of magnitude.
    nonzero = W_shared[W_shared > _EPS]
    use_log = nonzero.size > 1 and (nonzero.max() / nonzero.min() > 100.0)
    plot_data = np.log1p(W_shared) if use_log else W_shared
    cbar_label = "log1p(W)" if use_log else "W"

    figsize = (max(8.0, T * 0.7), max(6.0, P_plot * 0.15))
    row_labels = [f"p{i}" for i in range(P_plot)]
    df = pd.DataFrame(plot_data, index=row_labels, columns=labels)

    can_cluster = P_plot >= 2 and T >= 2

    try:
        import seaborn as sns
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import pdist

        if can_cluster:
            col_dist = pdist(plot_data.T + _EPS, metric="cosine")
            col_link = linkage(col_dist, method="average")
            row_dist = pdist(plot_data + _EPS, metric="euclidean")
            row_link = linkage(row_dist, method="average")
            g = sns.clustermap(
                df,
                row_linkage=row_link,
                col_linkage=col_link,
                cmap="viridis",
                figsize=figsize,
                linewidths=0.0,
                yticklabels=False,
                cbar_kws={"label": cbar_label},
            )
        else:
            g = sns.clustermap(
                df,
                row_cluster=False,
                col_cluster=False,
                cmap="viridis",
                figsize=figsize,
                linewidths=0.0,
                yticklabels=False,
                cbar_kws={"label": cbar_label},
            )

        g.fig.suptitle(
            "Emission Weights for Shared Peptides",
            y=1.02,
            fontsize=11,
            fontweight="bold",
        )
        g.fig.patch.set_facecolor("white")
        _save_fig(g.fig, path_stem)
        plt.close(g.fig)

    except ImportError:
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(plot_data, cmap="viridis", aspect="auto")
        ax.set_xticks(range(T))
        ax.set_xticklabels(labels, rotation=90, fontsize=max(4, 9 - T // 5))
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, label=cbar_label)
        ax.set_title("Emission Weights for Shared Peptides", fontweight="bold")
        fig.patch.set_facecolor("white")
        plt.tight_layout()
        _save_fig(fig, path_stem)
        plt.close(fig)


def _plot_degeneracy_profile(
    path_stem: Path,
    A: np.ndarray,
    labels: list,
) -> None:
    """Stacked horizontal bar chart: unique vs shared peptides per taxon."""
    T = A.shape[1]
    n_taxa_mapped = A.sum(axis=1)

    n_total = np.zeros(T, dtype=int)
    n_unique = np.zeros(T, dtype=int)

    for t in range(T):
        col = A[:, t] > 0
        n_total[t] = int(col.sum())
        n_unique[t] = int((col & (n_taxa_mapped == 1)).sum())

    n_shared = n_total - n_unique
    frac_shared = np.where(n_total > 0, n_shared / n_total, 0.0)

    # Sort by fraction shared descending.
    order = np.argsort(-frac_shared)
    sorted_labels = [labels[i] for i in order]
    sorted_unique = n_unique[order]
    sorted_shared = n_shared[order]

    figsize = (10.0, max(4.0, T * 0.4))
    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(T)
    ax.barh(y_pos, sorted_unique, color="#4472C4", label="Unique peptides")
    ax.barh(
        y_pos, sorted_shared, left=sorted_unique,
        color="#ED7D31", label="Shared peptides",
    )

    tick_fs = max(5, min(9, 200 // max(T, 1)))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_labels, fontsize=tick_fs)
    ax.set_xlabel("Number of observed peptides")
    ax.set_title("Per-taxon Peptide Degeneracy Profile", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    _save_fig(fig, path_stem)
    plt.close(fig)


def _plot_dominant_taxon_sharing(
    path_stem: Path,
    A: np.ndarray,
    labels: list,
    pi: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
) -> None:
    """Bar chart: shared peptide count between dominant taxon and all others."""
    T = A.shape[1]
    if T < 2:
        return

    if pi is not None and len(pi) == T:
        dominant = int(np.argmax(pi))
    elif y is not None:
        psm_proxy = (A * y[:, np.newaxis]).sum(axis=0)
        dominant = int(np.argmax(psm_proxy))
    else:
        dominant = 0

    dom_col = A[:, dominant] > 0
    others = [t for t in range(T) if t != dominant]
    shared_counts = [int((dom_col & (A[:, t] > 0)).sum()) for t in others]

    order = np.argsort(-np.array(shared_counts))
    sorted_labels = [labels[others[i]] for i in order]
    sorted_counts = [shared_counts[i] for i in order]

    figsize = (10.0, max(4.0, (T - 1) * 0.4))
    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(others))
    ax.barh(y_pos, sorted_counts, color="#4472C4")

    tick_fs = max(5, min(9, 200 // max(T, 1)))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_labels, fontsize=tick_fs)
    ax.set_xlabel("Number of shared peptides")
    ax.set_title(
        f"Shared Peptides: {labels[dominant]} vs Others",
        fontweight="bold",
    )
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    _save_fig(fig, path_stem)
    plt.close(fig)
