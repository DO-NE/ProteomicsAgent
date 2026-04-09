"""Identifiability diagnostics for the peptide-taxon mapping matrix.

These checks tell you whether the abundance vector can be recovered from the
data in principle, before any fit is attempted. They are cheap and worth
running whenever a database changes.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy import linalg as sla

logger = logging.getLogger(__name__)

_COLLINEAR_THRESHOLD = 0.95


def identifiability_report(
    A: np.ndarray,
    taxon_names: Optional[list] = None,
) -> dict:
    """Diagnose identifiability of the inference problem implied by ``A``.

    Parameters
    ----------
    A : np.ndarray, shape ``(P, T)``
        Binary peptide-to-taxon mapping matrix.
    taxon_names : list of str or None, optional
        Optional taxon labels for human-readable output. Defaults to
        ``["taxon_0", "taxon_1", ...]``.

    Returns
    -------
    dict
        Keys: ``rank``, ``n_taxa``, ``identifiable``, ``condition_number``,
        ``unique_peptide_counts``, ``at_risk_taxa``, ``collinear_groups``,
        ``cramer_rao_bounds``, ``warnings``.

    Notes
    -----
    The Cramer-Rao bound is computed assuming a uniform abundance vector and
    a notional total PSM count of 10000. It is meant for relative comparison
    between taxa, not as an absolute uncertainty quantifier — that role
    belongs to the standard errors returned by :class:`AbundanceEM`.
    """
    A_arr = np.asarray(A)
    if A_arr.ndim != 2:
        raise ValueError("A must be 2-D")

    P, T = A_arr.shape
    if taxon_names is None:
        taxon_names = [f"taxon_{i}" for i in range(T)]
    if len(taxon_names) != T:
        raise ValueError(
            f"taxon_names length ({len(taxon_names)}) does not match "
            f"number of columns in A ({T})"
        )

    A_bin = (A_arr != 0).astype(np.float64)
    n_t = A_bin.sum(axis=0)

    warnings_list: list = []

    # ---- normalized mapping matrix M = A / n_t (column-wise) -------------
    n_t_safe = np.where(n_t == 0, 1.0, n_t)
    M = A_bin / n_t_safe[np.newaxis, :]
    # zero-repertoire columns are forced to zero so they cannot contribute
    if (n_t == 0).any():
        M[:, n_t == 0] = 0.0
        empty_taxa = [taxon_names[i] for i in np.where(n_t == 0)[0]]
        warnings_list.append(
            f"{len(empty_taxa)} taxon(a) have empty peptide repertoires "
            f"and cannot be inferred: {empty_taxa}"
        )

    # ---- rank and condition number of M^T M ------------------------------
    rank = int(np.linalg.matrix_rank(M))
    identifiable = rank == T

    # Use singular values for a stable condition number; cap to a finite
    # sentinel when the smallest singular value underflows.
    try:
        sv = sla.svdvals(M)
    except sla.LinAlgError:
        sv = np.linalg.svd(M, compute_uv=False)
    if sv.size == 0 or sv[0] == 0:
        condition_number = float("inf")
    else:
        smallest = float(sv[-1]) if sv.size >= T else 0.0
        if smallest <= 0:
            condition_number = float("inf")
        else:
            # cond(M^T M) = (sigma_max / sigma_min)^2
            condition_number = float((sv[0] / smallest) ** 2)

    if not identifiable:
        warnings_list.append(
            f"Mapping matrix M is rank-deficient (rank {rank} < T={T}). "
            "Some taxon abundances are not uniquely identifiable from the data."
        )
    if condition_number > 1e8:
        warnings_list.append(
            f"Condition number of M^T M is very large ({condition_number:.2e}); "
            "estimates will be unstable for at least some taxa."
        )

    # ---- per-taxon unique peptide counts ---------------------------------
    membership_count = A_bin.sum(axis=1)  # how many taxa each peptide belongs to
    unique_mask = membership_count == 1
    unique_peptide_counts: dict = {}
    at_risk: list = []
    for t in range(T):
        unique_t = int(np.sum(A_bin[unique_mask, t] > 0)) if unique_mask.any() else 0
        unique_peptide_counts[taxon_names[t]] = unique_t
        if unique_t == 0:
            at_risk.append(taxon_names[t])
    if at_risk:
        warnings_list.append(
            f"{len(at_risk)} taxon(a) have no unique peptides; abundance "
            f"can only be inferred from sharing patterns: {at_risk}"
        )

    # ---- collinear groups (cosine similarity > threshold) ----------------
    collinear_groups: list = []
    norms = np.linalg.norm(M, axis=0)
    nonzero_cols = norms > 0
    if nonzero_cols.sum() >= 2:
        Mn = np.zeros_like(M)
        Mn[:, nonzero_cols] = M[:, nonzero_cols] / norms[nonzero_cols]
        cos_sim = Mn.T @ Mn
        # Only check upper triangle to avoid double counting.
        seen = [False] * T
        for i in range(T):
            if seen[i] or not nonzero_cols[i]:
                continue
            group = [i]
            for j in range(i + 1, T):
                if seen[j] or not nonzero_cols[j]:
                    continue
                if cos_sim[i, j] > _COLLINEAR_THRESHOLD:
                    group.append(j)
                    seen[j] = True
            if len(group) > 1:
                seen[i] = True
                collinear_groups.append([taxon_names[k] for k in group])
    if collinear_groups:
        warnings_list.append(
            f"{len(collinear_groups)} group(s) of near-collinear taxa "
            f"(cosine similarity > {_COLLINEAR_THRESHOLD}): {collinear_groups}"
        )

    # ---- Cramer-Rao bounds at uniform pi, N=10000 ------------------------
    crb: dict = {}
    notional_N = 10000.0
    pi_uniform = np.full(T, 1.0 / T)
    phi = M @ pi_uniform
    phi = np.maximum(phi, 1e-12)
    # Per-PSM information density: weights_p = phi_p (so the expected count
    # per peptide is N * phi_p, and the Fisher information element is
    # sum_p (M_pt M_pt' * N * phi_p) / phi_p^2).
    weights = notional_N / phi  # since y_p ~ N * phi_p in expectation
    F = (M * weights[:, np.newaxis]).T @ M
    try:
        F_inv = sla.pinvh(F + 1e-10 * np.eye(T))
        diag = np.diag(F_inv)
        diag = np.maximum(diag, 0.0)
        crb_values = np.sqrt(diag)
    except (sla.LinAlgError, ValueError):
        crb_values = np.full(T, float("nan"))
    for t in range(T):
        crb[taxon_names[t]] = float(crb_values[t])

    if warnings_list:
        for w in warnings_list:
            logger.warning("[identifiability] %s", w)
    else:
        logger.info("[identifiability] no issues detected for %d taxa", T)

    return {
        "rank": rank,
        "n_taxa": T,
        "identifiable": identifiable,
        "condition_number": condition_number,
        "unique_peptide_counts": unique_peptide_counts,
        "at_risk_taxa": at_risk,
        "collinear_groups": collinear_groups,
        "cramer_rao_bounds": crb,
        "warnings": warnings_list,
    }
