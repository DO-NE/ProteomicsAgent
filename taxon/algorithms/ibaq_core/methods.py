"""Five iBAQ quantification methods for species-level abundance estimation.

All public functions accept a PSM-level DataFrame with columns
``[peptide, protein_acc, species, intensity]`` and return
``dict[str, float]`` mapping species name to relative abundance
(proportions summing to 1.0).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── public API ────────────────────────────────────────────────────────────────


def raw_sum(df: pd.DataFrame) -> dict[str, float]:
    """Baseline: sum intensities per species, no iBAQ normalization."""
    # Deduplicate to one row per (peptide, species) so that peptides
    # mapping to multiple proteins within the same species are not
    # counted multiple times.
    deduped = df.groupby(["peptide", "species"])["intensity"].first().reset_index()
    species_intensity = deduped.groupby("species")["intensity"].sum()
    return _normalize(species_intensity)


def raw_ibaq(df: pd.DataFrame) -> dict[str, float]:
    """iBAQ with observed peptide count normalization.

    Per protein: iBAQ = total_intensity / n_unique_observed_peptides.
    Species abundance = sum of per-protein iBAQ scores, normalized.
    """
    protein_stats = (
        df.groupby(["protein_acc", "species"])
        .agg(
            total_intensity=("intensity", "sum"),
            num_peptides=("peptide", "nunique"),
        )
        .reset_index()
    )
    protein_stats = protein_stats[protein_stats["num_peptides"] > 0].copy()
    if protein_stats.empty:
        return {}

    protein_stats["ibaq_score"] = (
        protein_stats["total_intensity"] / protein_stats["num_peptides"]
    )
    species_ibaq = protein_stats.groupby("species")["ibaq_score"].sum()
    return _normalize(species_ibaq)


def ibaq_theoretical(
    df: pd.DataFrame,
    theoretical_counts: dict[str, int],
    fallback_count: int | None = None,
) -> dict[str, float]:
    """iBAQ with theoretical tryptic peptide count normalization.

    Per protein: iBAQ = total_intensity / n_theoretical_peptides.
    Uses *theoretical_counts* (accession -> count from in-silico digestion).
    Proteins absent from the map use *fallback_count* (median if ``None``).
    """
    protein_stats = (
        df.groupby(["protein_acc", "species"])
        .agg(total_intensity=("intensity", "sum"))
        .reset_index()
    )
    if protein_stats.empty:
        return {}

    valid_counts = [v for v in theoretical_counts.values() if v > 0]
    if fallback_count is None:
        fallback_count = max(int(np.median(valid_counts)), 1) if valid_counts else 1

    protein_stats["n_theo"] = (
        protein_stats["protein_acc"]
        .map(theoretical_counts)
        .fillna(fallback_count)
        .clip(lower=1)
    )
    protein_stats["ibaq_score"] = (
        protein_stats["total_intensity"] / protein_stats["n_theo"]
    )
    species_ibaq = protein_stats.groupby("species")["ibaq_score"].sum()
    return _normalize(species_ibaq)


def top_n_proteins(df: pd.DataFrame, n: int = 3) -> dict[str, float]:
    """Top-N protein approach: use only the N most intense proteins per species.

    These high-abundance proteins are least affected by detection bias.
    """
    protein_stats = (
        df.groupby(["protein_acc", "species"])
        .agg(total_intensity=("intensity", "sum"))
        .reset_index()
    )
    if protein_stats.empty:
        return {}

    top = (
        protein_stats
        .sort_values(["species", "total_intensity"], ascending=[True, False])
        .groupby("species")
        .head(n)
        .groupby("species")["total_intensity"]
        .sum()
    )
    return _normalize(top)


def ibaq_em(
    df: pd.DataFrame,
    max_iter: int = 500,
    tol: float = 1e-8,
    epsilon: float = 1e-12,
) -> dict[str, float]:
    """EM shared-peptide redistribution followed by iBAQ.

    1. Identify unique (single-species) vs shared (multi-species) peptides.
    2. Initialize species proportions (pi) from unique-peptide intensities.
    3. EM loop: E-step assigns shared peptide intensities proportionally,
       M-step recomputes pi.
    4. After convergence, compute iBAQ from EM-reassigned intensities.
    """
    em_df = _build_em_reassigned_df(df, max_iter=max_iter, tol=tol, epsilon=epsilon)
    if em_df.empty or "em_intensity" not in em_df.columns:
        return {}

    protein_stats = (
        em_df.groupby(["protein_acc", "species"])
        .agg(
            total_intensity=("em_intensity", "sum"),
            num_peptides=("peptide", "nunique"),
        )
        .reset_index()
    )
    protein_stats = protein_stats[protein_stats["num_peptides"] > 0].copy()
    if protein_stats.empty:
        return {}

    protein_stats["ibaq_score"] = (
        protein_stats["total_intensity"] / protein_stats["num_peptides"]
    )
    species_ibaq = protein_stats.groupby("species")["ibaq_score"].sum()
    return _normalize(species_ibaq)


# ── internals ─────────────────────────────────────────────────────────────────


def _normalize(series: pd.Series) -> dict[str, float]:
    """Normalize a pandas Series to proportions summing to 1.0."""
    total = float(series.sum())
    if total <= 0:
        return {}
    return (series / total).to_dict()


def _build_em_reassigned_df(
    df: pd.DataFrame,
    max_iter: int = 500,
    tol: float = 1e-8,
    epsilon: float = 1e-12,
) -> pd.DataFrame:
    """EM redistribution of shared-peptide intensities across species.

    Returns a copy of *df* with an added ``em_intensity`` column where
    shared peptides have their intensity allocated proportionally.
    """
    working = df[df["intensity"].notna()].copy()
    if working.empty:
        out = df.copy()
        out["em_intensity"] = np.nan
        return out

    # Group by peptide to find which species each peptide maps to.
    grouped = (
        working.groupby("peptide", dropna=False)
        .agg(
            intensity=("intensity", "first"),
            species_set=("species", lambda x: tuple(sorted(set(x)))),
        )
        .reset_index()
    )
    grouped["n_species"] = grouped["species_set"].apply(len)
    grouped = grouped[grouped["n_species"] > 0].copy()
    if grouped.empty:
        out = df.copy()
        out["em_intensity"] = np.nan
        return out

    species_list = sorted({sp for sp_set in grouped["species_set"] for sp in sp_set})
    s_to_idx = {sp: i for i, sp in enumerate(species_list)}
    n_species = len(species_list)

    unique = grouped[grouped["n_species"] == 1]
    shared = grouped[grouped["n_species"] > 1]

    # Initialize pi from unique-peptide intensities.
    unique_intensity = np.zeros(n_species, dtype=np.float64)
    for _, row in unique.iterrows():
        sp = row["species_set"][0]
        unique_intensity[s_to_idx[sp]] += float(row["intensity"])

    if unique_intensity.sum() > 0:
        pi = unique_intensity / unique_intensity.sum()
    else:
        pi = np.full(n_species, 1.0 / n_species, dtype=np.float64)

    # Floor zeros.
    zero_mask = pi <= 0
    if zero_mask.any():
        pi[zero_mask] = epsilon
        pi /= pi.sum()

    # Pre-compute shared peptide data.
    shared_payload: list[tuple[float, np.ndarray]] = []
    for _, row in shared.iterrows():
        cand_idx = np.array([s_to_idx[sp] for sp in row["species_set"]], dtype=int)
        shared_payload.append((float(row["intensity"]), cand_idx))

    # EM iterations.
    for it in range(max_iter):
        reassigned = unique_intensity.copy()

        for intensity, cand_idx in shared_payload:
            denom = pi[cand_idx].sum()
            if denom <= 0:
                weights = np.full(len(cand_idx), 1.0 / len(cand_idx))
            else:
                weights = pi[cand_idx] / denom
            reassigned[cand_idx] += intensity * weights

        total = reassigned.sum()
        if total <= 0:
            break

        new_pi = reassigned / total
        if np.max(np.abs(new_pi - pi)) < tol:
            logger.debug("EM converged at iteration %d", it + 1)
            pi = new_pi
            break
        pi = new_pi

    # Allocate intensities back to rows.
    alloc: dict[tuple[str, str], float] = {}
    for _, row in grouped.iterrows():
        pep = row["peptide"]
        intensity = float(row["intensity"])
        species_present = list(row["species_set"])

        if len(species_present) == 1:
            alloc[(pep, species_present[0])] = intensity
        else:
            cand_idx = np.array([s_to_idx[sp] for sp in species_present], dtype=int)
            denom = pi[cand_idx].sum()
            if denom <= 0:
                weights = np.full(len(cand_idx), 1.0 / len(cand_idx))
            else:
                weights = pi[cand_idx] / denom
            for sp, w in zip(species_present, weights):
                alloc[(pep, sp)] = intensity * float(w)

    # Map back to the working DataFrame.
    working["em_intensity"] = working.apply(
        lambda r: alloc.get((r["peptide"], r["species"]), np.nan), axis=1,
    )

    # Divide evenly among rows for the same (peptide, species) pair
    # to avoid double-counting when multiple proteins share a peptide.
    counts = working.groupby(["peptide", "species"]).transform("size")
    working["em_intensity"] = working["em_intensity"] / counts

    return working
