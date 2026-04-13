"""Peptide detectability weight estimation.

Peptide detectability varies enormously across peptides due to differences in
ionization efficiency, chromatographic behavior, and fragmentation properties.
The default uniform-emission model (``M_{pt} = A_{pt} / n_t``) assumes every
peptide from a taxon is equally likely to be observed, which is biologically
wrong.

This module provides methods to estimate per-peptide detectability weights
``d_p`` that correct the emission probabilities:

    M_{pt} = (A_{pt} * d_p) / sum_{p'} (A_{p't} * d_{p'})

Sources for detectability weights (in decreasing reliability):

1. **Pure culture controls** (empirical) — observed spectral counts in
   single-organism experiments directly reflect peptide detectability under the
   same LC-MS/MS conditions.  This is the gold standard when available.
2. **Flat prior** (default) — all weights equal to 1, reducing to the uniform
   emission baseline.

Future extensions may add sequence-based prediction (hydrophobicity,
charge, length, etc.) as a fallback when pure culture data is unavailable.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class DetectabilityWeights:
    """Container for per-peptide detectability weights.

    The weight vector ``d`` has one entry per peptide in the observed peptide
    list.  Weights are non-negative and need not sum to 1 — the EM model
    normalizes each taxon column of the emission matrix independently.

    Parameters
    ----------
    weights : dict[str, float]
        Mapping from peptide sequence to detectability weight.
    default_weight : float
        Weight assigned to peptides absent from the ``weights`` dict
        (default ``1.0``).

    Attributes
    ----------
    weights : dict[str, float]
        The underlying peptide -> weight mapping (frozen after construction).
    default_weight : float
        Fallback weight for unseen peptides.
    source : str
        Human-readable label describing how the weights were estimated.
    n_peptides_estimated : int
        Number of peptides with empirical weight estimates.
    """

    def __init__(
        self,
        weights: dict[str, float],
        default_weight: float = 1.0,
        source: str = "manual",
    ) -> None:
        if default_weight < 0:
            raise ValueError("default_weight must be non-negative")
        self.weights = dict(weights)
        self.default_weight = float(default_weight)
        self.source = source
        self.n_peptides_estimated = len(self.weights)

    def get_weight_vector(self, peptide_list: list[str]) -> np.ndarray:
        """Return a weight array aligned with ``peptide_list``.

        Parameters
        ----------
        peptide_list : list of str
            Observed peptide sequences in the order matching the mapping
            matrix rows.

        Returns
        -------
        np.ndarray, shape ``(P,)``
            Non-negative weight vector.  Peptides present in ``self.weights``
            get their estimated value; all others get ``self.default_weight``.
        """
        d = np.full(len(peptide_list), self.default_weight, dtype=np.float64)
        n_hits = 0
        for i, pep in enumerate(peptide_list):
            if pep in self.weights:
                d[i] = self.weights[pep]
                n_hits += 1
        logger.info(
            "Detectability weights (%s): %d / %d peptides had empirical "
            "weights; %d used default=%.4g",
            self.source,
            n_hits,
            len(peptide_list),
            len(peptide_list) - n_hits,
            self.default_weight,
        )
        return d

    # ---------------------------------------------------------------- factories

    @classmethod
    def uniform(cls) -> "DetectabilityWeights":
        """Return a trivial weight set where every peptide has weight 1.

        This reproduces the uniform-emission baseline of the original model.
        """
        return cls(weights={}, default_weight=1.0, source="uniform")

    @classmethod
    def from_pure_cultures(
        cls,
        pure_culture_counts: dict[str, dict[str, int]],
        pseudocount: float = 0.5,
        min_total_psms: int = 100,
    ) -> "DetectabilityWeights":
        """Estimate detectability weights from pure culture experiments.

        In a pure culture of taxon *t*, the observed spectral count of peptide
        *p* is proportional to its detectability (all abundance goes to one
        organism, so variation in counts reflects purely peptide-level effects).

        For each peptide observed in at least one pure culture, the weight is::

            d_p = mean over cultures c in which p was seen of:
                  (y_p^c + pseudocount) / (N^c + P^c * pseudocount)

        where ``N^c`` is total PSMs and ``P^c`` is total unique peptides in
        culture *c*.  The pseudocount prevents zero weights and provides
        Bayesian shrinkage.  The default weight for peptides not in any pure
        culture is the median of the estimated weights.

        Parameters
        ----------
        pure_culture_counts : dict[str, dict[str, int]]
            Mapping ``culture_label -> {peptide_sequence: spectral_count}``.
            The culture label is typically the taxon name but need not be — the
            estimation only uses the count distributions.
        pseudocount : float
            Additive pseudocount for Bayesian smoothing (default ``0.5``).
        min_total_psms : int
            Minimum total PSMs for a culture to be used; cultures below this
            threshold are skipped with a warning (default ``100``).

        Returns
        -------
        DetectabilityWeights
            Weights estimated from the pure culture data.  The ``source``
            attribute is set to ``"pure_culture"``.

        Raises
        ------
        ValueError
            If no usable cultures remain after filtering.
        """
        if pseudocount < 0:
            raise ValueError("pseudocount must be non-negative")

        # Compute per-culture normalized frequencies.
        peptide_freq_sums: dict[str, list[float]] = {}
        n_cultures_used = 0

        for label, counts in pure_culture_counts.items():
            total = sum(counts.values())
            if total < min_total_psms:
                logger.warning(
                    "Pure culture '%s' has only %d PSMs (< %d); skipping",
                    label,
                    total,
                    min_total_psms,
                )
                continue
            n_cultures_used += 1
            n_unique = len(counts)
            denom = total + n_unique * pseudocount
            for pep, count in counts.items():
                freq = (count + pseudocount) / denom
                if pep not in peptide_freq_sums:
                    peptide_freq_sums[pep] = []
                peptide_freq_sums[pep].append(freq)

        if n_cultures_used == 0:
            raise ValueError(
                "No pure cultures had sufficient PSMs "
                f"(min_total_psms={min_total_psms})"
            )

        # Average across cultures for peptides seen in multiple.
        weights: dict[str, float] = {}
        all_values = []
        for pep, freqs in peptide_freq_sums.items():
            w = float(np.mean(freqs))
            weights[pep] = w
            all_values.append(w)

        # Default weight = median of estimated weights.  This is a principled
        # choice: peptides not seen in any pure culture are assumed to have
        # "typical" detectability rather than the extremes.
        default_weight = float(np.median(all_values)) if all_values else 1.0

        logger.info(
            "Estimated detectability weights from %d pure culture(s): "
            "%d unique peptides, default_weight=%.6g, "
            "weight range=[%.6g, %.6g]",
            n_cultures_used,
            len(weights),
            default_weight,
            min(all_values) if all_values else 0,
            max(all_values) if all_values else 0,
        )

        return cls(
            weights=weights,
            default_weight=default_weight,
            source="pure_culture",
        )

    @classmethod
    def from_pure_culture_pepxml(
        cls,
        pepxml_paths: dict[str, str],
        exclude_prefixes: list[str] | None = None,
        pseudocount: float = 0.5,
        min_total_psms: int = 100,
    ) -> "DetectabilityWeights":
        """Convenience: parse pepXML files, then estimate weights.

        Parameters
        ----------
        pepxml_paths : dict[str, str]
            Mapping ``culture_label -> pepxml_file_path``.
        exclude_prefixes : list[str] or None
            Prefixes to exclude during pepXML parsing (default
            ``["DECOY", "contag"]``).
        pseudocount : float
            Passed to :meth:`from_pure_cultures`.
        min_total_psms : int
            Passed to :meth:`from_pure_cultures`.

        Returns
        -------
        DetectabilityWeights
        """
        from .pepxml_parser import parse_pepxml

        if exclude_prefixes is None:
            exclude_prefixes = ["DECOY", "contag"]

        pure_culture_counts: dict[str, dict[str, int]] = {}
        for label, path in pepxml_paths.items():
            logger.info("Parsing pure culture pepXML for '%s': %s", label, path)
            counts, _ = parse_pepxml(path, exclude_prefixes=exclude_prefixes)
            pure_culture_counts[label] = counts

        return cls.from_pure_cultures(
            pure_culture_counts,
            pseudocount=pseudocount,
            min_total_psms=min_total_psms,
        )


def build_weighted_emission_matrix(
    A: np.ndarray,
    detectability_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Build the emission matrix M from the mapping matrix A and optional weights.

    Parameters
    ----------
    A : np.ndarray, shape ``(P, T)``
        Binary peptide-to-taxon mapping matrix.
    detectability_weights : np.ndarray or None, shape ``(P,)``
        Per-peptide detectability weights.  If ``None``, uniform weights
        (all ones) are used, reproducing the original model.

    Returns
    -------
    M : np.ndarray, shape ``(P, T)``
        Emission probability matrix.  Each column sums to 1 (for columns
        with at least one nonzero entry in A).

    Notes
    -----
    The weighted emission probability is::

        M_{pt} = (A_{pt} * d_p) / sum_{p'} (A_{p't} * d_{p'})

    When ``detectability_weights`` is ``None`` (or all ones), this reduces
    to the original ``M_{pt} = A_{pt} / n_t``.
    """
    P, T = A.shape
    A_float = A.astype(np.float64) if A.dtype != np.float64 else A.copy()

    if detectability_weights is not None:
        d = np.asarray(detectability_weights, dtype=np.float64)
        if d.shape != (P,):
            raise ValueError(
                f"detectability_weights must have shape ({P},), got {d.shape}"
            )
        if (d < 0).any():
            raise ValueError("detectability_weights must be non-negative")
        # Weight each row of A by its detectability.
        A_weighted = A_float * d[:, np.newaxis]
    else:
        A_weighted = A_float

    # Column sums = effective repertoire sizes.
    col_sums = A_weighted.sum(axis=0)
    # Guard against zero columns (taxon with no peptides or all-zero weights).
    col_sums_safe = np.where(col_sums == 0, 1.0, col_sums)
    M = A_weighted / col_sums_safe[np.newaxis, :]
    # Force zero columns back to zero.
    if (col_sums == 0).any():
        M[:, col_sums == 0] = 0.0

    return M
